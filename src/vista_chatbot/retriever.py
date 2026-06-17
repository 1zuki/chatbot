from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .bm25 import BM25Index, tokenize
from .chunking import Chunk, load_wiki_chunks, read_chunks_jsonl, write_chunks_jsonl
from .config import RetrievalConfig
from .text import compact_for_match, normalize_text, safe_truncate

UNKNOWN_WIKI_REPLY = "I don't know from the wiki yet. Try /wiki or ask staff for further assistance."

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "cant",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "use",
    "was",
    "were",
    "what",
    "wat",
    "whats",
    "wats",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ExtractiveCandidate:
    overlap: int
    intent: float
    score: float
    text: str
    has_command: bool
    has_requirement: bool
    warning_like: bool
    key_match: bool = False


class EmbeddingIndex:
    """Small, dependency-light cosine-similarity index backed by NumPy.

    FAISS is faster at huge scale, but for a server wiki this is easier to deploy
    inside a Minescript workflow and avoids native index incompatibilities.
    """

    def __init__(self, index_dir: Path, embedding_model: str):
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        self.chunks_path = index_dir / "chunks.jsonl"
        self.embeddings_path = index_dir / "embeddings.npy"
        self.meta_path = index_dir / "meta.json"
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self._model = None
        self._bm25: BM25Index | None = None
        self._table_groups: dict[tuple[str, tuple[str, ...]], list[int]] | None = None

    @classmethod
    def build(
        cls,
        *,
        wiki_dir: Path,
        index_dir: Path,
        config: RetrievalConfig,
    ) -> "EmbeddingIndex":
        index = cls(index_dir, config.embedding_model)
        chunks = load_wiki_chunks(
            wiki_dir,
            globs=config.wiki_globs,
            chunk_chars=config.chunk_chars,
            chunk_overlap=config.chunk_overlap,
        )
        if not chunks:
            raise RuntimeError(f"No wiki chunks found in {wiki_dir}. Add .md/.mdx files first.")
        model = index._load_model()
        texts = [format_chunk_for_embedding(c) for c in chunks]
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        index_dir.mkdir(parents=True, exist_ok=True)
        write_chunks_jsonl(chunks, index.chunks_path)
        np.save(index.embeddings_path, embeddings)
        index.meta_path.write_text(
            json.dumps(
                {
                    "embedding_model": config.embedding_model,
                    "chunk_count": len(chunks),
                    "embedding_dim": int(embeddings.shape[1]),
                    "wiki_dir": str(wiki_dir),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        index.chunks = chunks
        index.embeddings = embeddings
        return index

    @classmethod
    def load(cls, index_dir: Path, embedding_model: str) -> "EmbeddingIndex":
        index = cls(index_dir, embedding_model)
        if not index.chunks_path.exists() or not index.embeddings_path.exists():
            raise FileNotFoundError(
                f"Retriever index not found in {index_dir}. Run scripts/build_wiki_index.py first."
            )
        index.chunks = read_chunks_jsonl(index.chunks_path)
        index.embeddings = np.load(index.embeddings_path).astype("float32")
        if len(index.chunks) != index.embeddings.shape[0]:
            raise RuntimeError("Retriever index is corrupted: chunk count != embedding count")
        return index

    def search(self, query: str, *, top_k: int, min_score: float) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Index is not loaded")
        if not query.strip():
            return []
        model = self._load_model()
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
        dense = self.embeddings @ q
        if dense.size == 0:
            return []

        # Dense gate stays the hard inclusion criterion: a chunk is only eligible
        # if its cosine similarity clears min_score. Lexical BM25 then reorders
        # within that gated set, so exact command/term matches (e.g. "/town set
        # taxes") can be pulled up without ever injecting a semantically
        # unrelated chunk that the embedder already rejected.
        eligible = [i for i in range(dense.size) if float(dense[i]) >= min_score]

        out: list[SearchResult] = []
        out_indices: list[int] = []
        # An empty gated set normally means "no answer", but a rare query anchor
        # (e.g. "detonate", df=5) can name a chunk that sits just under the gate
        # — the rare-term rescue below reaches those. So fall through to it rather
        # than early-returning; if no rare anchor matches, the rescue is a no-op
        # and we still return [].
        if eligible:
            dense_rank = sorted(eligible, key=lambda i: float(dense[i]), reverse=True)
            order = self._fuse_rankings(query, dense, dense_rank)

            seen_sources: set[tuple[str, tuple[str, ...]]] = set()
            for idx in order:
                chunk = self.chunks[idx]
                key = (chunk.source_path, tuple(chunk.heading_path))
                # Keep result diversity, but still allow another chunk if few results.
                if key in seen_sources and len(out) >= math.ceil(top_k / 2):
                    continue
                seen_sources.add(key)
                # Score stays the dense cosine value: downstream confidence gating
                # (_low_confidence_match) and context display depend on that scale.
                out.append(SearchResult(chunk=chunk, score=float(dense[idx])))
                out_indices.append(idx)
                if len(out) >= top_k:
                    break
        out, out_indices = self._rescue_rare_term_chunks(
            out, out_indices, dense, query, min_score
        )
        return self._complete_table_siblings(out, out_indices, dense)

    def _fuse_rankings(self, query: str, dense: np.ndarray, dense_rank: list[int]) -> list[int]:
        """Reciprocal Rank Fusion of dense and BM25 rankings over the gated set.

        RRF (k=60) needs no score normalization between the two very different
        score scales. Ties and a missing lexical signal both degrade gracefully
        to the dense order, so behavior is unchanged when a query has no term
        overlap with any chunk.
        """
        rrf_k = 60
        scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        bm25 = self._get_bm25()
        query_terms = tokenize(query)
        if bm25 is not None and query_terms:
            lex_scores = bm25.scores(query_terms)
            lex_eligible = [i for i in dense_rank if lex_scores[i] > 0.0]
            lex_rank = sorted(lex_eligible, key=lambda i: lex_scores[i], reverse=True)
            for rank, idx in enumerate(lex_rank):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Recency as a third RRF signal, but only for "whats new / latest /
        # changed" queries: rank the dated chunks (changelog/announcement/roadmap,
        # named DD-MM-YY) newest-first and add their RRF contribution. This lifts
        # the most recent entry above an older one that happens to out-cosine it,
        # without ever pulling a changelog into a topical question (the gate) or
        # touching the non-dated wiki (no parsed date -> not in the ranking).
        #
        # The recency component uses a much smaller k than dense/BM25 (recency_k
        # vs rrf_k=60). A pure "what changed recently" query has no real topical
        # signal — every changelog out-cosines the others by noise — so recency
        # must dominate, not just nudge: at k=60 adjacent dates differ by ~0.0003,
        # far too little to reorder. At k=2 the newest dated chunk gets +0.33 and
        # the next +0.25, a gap wider than the whole dense+BM25 budget (~0.03), so
        # among the gated dated chunks the order is effectively newest-first. The
        # gate still confines this to recency queries, and non-dated wiki pages
        # (no parsed date) never enter the recency ranking at all.
        if _looks_recency_query(query):
            recency_k = 2
            dated = [
                (i, d)
                for i in dense_rank
                if (d := _parse_source_date(self.chunks[i].source_path)) is not None
            ]
            date_rank = [i for i, _ in sorted(dated, key=lambda t: t[1], reverse=True)]
            for rank, idx in enumerate(date_rank):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (recency_k + rank + 1)
        else:
            # Mirror image: on a *topical* (non-recency) question, dated content
            # is a historical delta, not the canonical answer. A changelog chunk
            # can out-cosine the evergreen wiki page on its own topic ("star
            # rank") and then win at the extraction layer on raw term overlap, so
            # demoting its rank order is not enough — it has to fall out of the
            # top_k entirely so the extractor never sees it. We subtract a penalty
            # larger than the whole RRF budget (max fused score ~0.033) from every
            # dated chunk, sinking all of them below every non-dated wiki chunk.
            # They stay in dense_rank (and keep their relative order by the dense
            # tiebreak), so if dated content is the *only* match it is still
            # returned as a fallback rather than dropped.
            for idx in dense_rank:
                if _parse_source_date(self.chunks[idx].source_path) is not None:
                    scores[idx] = scores.get(idx, 0.0) - 1.0

        # Sort by fused score; break ties by dense cosine so the ordering is
        # deterministic and never worse than dense-only.
        return sorted(dense_rank, key=lambda i: (scores.get(i, 0.0), float(dense[i])), reverse=True)

    def _get_bm25(self) -> "BM25Index | None":
        if self._bm25 is None and self.chunks:
            self._bm25 = BM25Index.build(
                [tokenize(format_chunk_for_embedding(c)) for c in self.chunks]
            )
        return self._bm25

    # A query term appearing in <= this many chunks is a rare, discriminating
    # anchor (e.g. an enchant name like "slingshot" df=2) as opposed to a common
    # topic word ("price" df=14, "town" df=61). Chosen from the live corpus
    # (N=362): every specific item/command name sits at df<=6, every generic
    # topic word at df>=12, so the gap is wide and the threshold is not delicate.
    _RARE_TERM_MAX_DF = 8
    # Cap rescued chunks so a rare term that genuinely appears in many places
    # can't flood the result set past the diversity-limited main list.
    _MAX_RARE_RESCUE = 3
    # A rare exact-term anchor the query explicitly named is itself strong
    # relevance evidence, so the rescue may reach chunks *below* the dense gate
    # down to this relaxed floor. Kept well above zero so a chunk the embedder
    # scored as truly unrelated still can't enter on an incidental term hit.
    _RARE_RESCUE_MIN_SCORE = 0.10

    def _rescue_rare_term_chunks(
        self,
        out: list[SearchResult],
        out_indices: list[int],
        dense: np.ndarray,
        query: str,
        min_score: float,
    ) -> tuple[list[SearchResult], list[int]]:
        """Rescue a chunk that uniquely contains a rare query term.

        RRF fuses dense and BM25 by *rank position*, which discards IDF
        magnitude. So a query like "price for slingshot" — a common word
        ("price", in every pricing tier) plus a rare anchor ("slingshot", in one
        chunk) — lets the dozen generic "Baseline Orb Price" intros crowd the
        single answer-bearing chunk out of the top_k. We detect rare anchors by
        document frequency, and if no chunk already in the result list contains
        one, we splice in the chunks that do (best BM25 first).

        A rare exact-term anchor the query explicitly named is itself strong
        relevance evidence, so the search reaches chunks *below* the dense gate
        down to ``_RARE_RESCUE_MIN_SCORE`` (e.g. the "Atomic Detonate" price row
        sits at dense 0.156, under the 0.18 gate, but "atomic" df=2 names it
        exactly). The floor stays well above zero so a chunk the embedder scored
        as truly unrelated can't enter on an incidental term hit. Spliced chunks
        keep their own dense score; the main list is untouched. Inert for queries
        whose terms are all common, which includes every command/how-to query
        that the other ranking paths handle.
        """
        bm25 = self._get_bm25()
        if bm25 is None:
            return out, out_indices
        query_terms = [t for t in tokenize(query) if t in bm25.df]
        rare_terms = [t for t in query_terms if bm25.df.get(t, 0) <= self._RARE_TERM_MAX_DF]
        if not rare_terms:
            return out, out_indices

        def contains(idx: int, term: str) -> bool:
            return bm25.doc_freqs[idx].get(term, 0) > 0

        uncovered = [t for t in rare_terms if not any(contains(i, t) for i in out_indices)]
        if not uncovered:
            return out, out_indices

        present = set(out_indices)
        floor = min(min_score, self._RARE_RESCUE_MIN_SCORE)
        lex = bm25.scores(query_terms)
        cands: set[int] = set()
        for term in uncovered:
            for i in range(dense.size):
                if i not in present and float(dense[i]) >= floor and contains(i, term):
                    cands.add(i)
        if not cands:
            return out, out_indices
        for i in sorted(cands, key=lambda j: lex[j], reverse=True)[: self._MAX_RARE_RESCUE]:
            out.append(SearchResult(chunk=self.chunks[i], score=float(dense[i])))
            out_indices.append(i)
        return out, out_indices

    # Cap how many sibling fragments a single query may pull in, so a
    # pathologically long table can't flood the result set. A server wiki's
    # largest table (Stars, 30 rows) fragments into ~6 pieces at the configured
    # chunk size, well under this bound.
    _MAX_SIBLINGS = 12

    @staticmethod
    def _is_table_fragment(chunk: Chunk) -> bool:
        """True when a chunk is mostly markdown table rows.

        ``_chunk_table`` emits ``"<context>\\n| row |\\n| row |"`` — a single
        context word followed by ``|``-rows. A chunk with two or more row lines
        that dominate its content is a fragment of a (possibly split) table.
        """
        row_lines = 0
        content_lines = 0
        for line in chunk.text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            content_lines += 1
            if stripped.startswith("|"):
                row_lines += 1
        return row_lines >= 2 and row_lines >= content_lines - 1

    def _get_table_groups(self) -> dict[tuple[str, tuple[str, ...]], list[int]]:
        """Map each ``(source_path, heading_path)`` to its table chunk indices.

        Two distinct retrieval failures motivate this:

        * A long table the chunker spread across several fragments (e.g. Stars),
          where the fragment holding the queried row loses the ranking race to a
          near-identical sibling.
        * A table preceded by a prose intro under the same heading (e.g. each
          pricing-guide tier's "Baseline Orb Price" bullets sit above the price
          table). The intro matches a "price" query strongly while the longer
          table chunk holding the actual row gets length-penalised below it, so
          the answer-bearing chunk never surfaces.

        Both are fixed by completing a heading's table chunks once any chunk
        under that heading is retrieved, so single-table-chunk headings are kept
        too (the prose intro is the trigger, the table chunk is the completion).
        Built once and cached; the index is immutable after load.
        """
        if self._table_groups is not None:
            return self._table_groups
        groups: dict[tuple[str, tuple[str, ...]], list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            if not self._is_table_fragment(chunk):
                continue
            key = (chunk.source_path, tuple(chunk.heading_path))
            groups.setdefault(key, []).append(i)
        for key in groups:
            groups[key] = sorted(groups[key], key=lambda i: self.chunks[i].start_char)
        self._table_groups = groups
        return groups

    def _complete_table_siblings(
        self,
        out: list[SearchResult],
        out_indices: list[int],
        dense: np.ndarray,
    ) -> list[SearchResult]:
        """Complete a heading's table chunks once any chunk under it is retrieved.

        Two failure modes are covered (see ``_get_table_groups``):

        * Split table — a fragment of e.g. the Stars table surfaces but the
          fragment holding the queried row (rank 3) lost the ranking race to a
          near-identical sibling. We splice the missing siblings in.
        * Prose-intro trigger — a pricing tier's "Baseline Orb Price" intro
          matches a "price" query and surfaces, but the longer table chunk under
          the same heading (holding ``| Detonate | 1,000,000$ |``) got
          length-penalised out. The intro is the trigger; the table is spliced.

        Either way the full table reaches the row-aware extractor, which matches
        the specific row by its key column. Only *table* chunks are ever spliced
        (prose neighbours are never pulled in), and each spliced chunk keeps its
        own (often sub-gate) cosine score; the trigger's score is untouched.
        """
        groups = self._get_table_groups()
        if not groups:
            return out
        present = set(out_indices)
        added = 0
        result: list[SearchResult] = []
        completed_groups: set[tuple[str, tuple[str, ...]]] = set()
        for res, idx in zip(out, out_indices):
            result.append(res)
            chunk = self.chunks[idx]
            key = (chunk.source_path, tuple(chunk.heading_path))
            group = groups.get(key)
            if group is None or key in completed_groups:
                continue
            completed_groups.add(key)
            for sib in group:
                if sib in present or added >= self._MAX_SIBLINGS:
                    continue
                present.add(sib)
                result.append(
                    SearchResult(chunk=self.chunks[sib], score=float(dense[sib]))
                )
                added += 1
        return result

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency environment
            raise RuntimeError(
                "sentence-transformers is required for retrieval. Install requirements.txt first."
            ) from exc
        self._model = SentenceTransformer(self.embedding_model_name)
        return self._model


def format_chunk_for_embedding(chunk: Chunk) -> str:
    heading = " > ".join(chunk.heading_path)
    return f"Title: {chunk.title}\nPath: {chunk.source_path}\nHeading: {heading}\n\n{chunk.text}"


def build_context(results: Iterable[SearchResult], *, max_chars: int) -> str:
    blocks: list[str] = []
    total = 0
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title
        block = (
            f"[Wiki chunk {i}] source={chunk.source_path} score={result.score:.3f}\n"
            f"Title: {chunk.title}\nHeading: {heading}\n"
            f"Text: {chunk.text.strip()}"
        )
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining <= 120:
                break
            block = safe_truncate(block, remaining)
        blocks.append(block)
        total += len(block) + 2
        if total >= max_chars:
            break
    return "\n\n".join(blocks)


WIKI_REPLY_PREFIX = "Wiki says: "
# Per server-owner request: item prices on this wiki are community-sourced
# averages, not official rates, so a price question gets a disclaiming prefix
# instead of "Wiki says:" to set expectations.
COMMUNITY_PRICE_PREFIX = "The pricing is set by the community (approx): "


def _looks_price_query(query: str) -> bool:
    """True when the user is asking the going price of something.

    Deliberately narrow to ``price``/``prices``/``pricing``: those are the
    community-sourced market values. It does NOT match server-set costs like
    ``upkeep``/``tax``/``fee`` (covered by the broader ``_looks_cost_query``),
    which are official rates, not community pricing.
    """
    return bool(re.search(r"\bpric(?:e|es|ing)\b", compact_for_match(query)))


def _answer_prefix(query: str) -> str:
    return COMMUNITY_PRICE_PREFIX if _looks_price_query(query) else WIKI_REPLY_PREFIX


def extractive_answer(query: str, results: list[SearchResult], *, max_chars: int) -> str:
    if not results:
        return UNKNOWN_WIKI_REPLY
    ranked = _rank_extractive_candidates(query, results)
    if not ranked:
        return UNKNOWN_WIKI_REPLY
    if _low_confidence_match(query, ranked[0]):
        return UNKNOWN_WIKI_REPLY
    answer = _compose_answer_from_candidates(query, ranked)
    answer = reformat_extractive(answer)
    return safe_truncate(f"{_answer_prefix(query)}{answer}", max_chars)


def extractive_candidates(
    query: str,
    results: list[SearchResult],
    *,
    max_candidates: int,
) -> list[str]:
    """Return ranked extractive sentence candidates for optional LLM reranking."""
    if not results or max_candidates <= 0:
        return []
    ranked = _rank_extractive_candidates(query, results)
    out = [c.text for c in ranked[:max_candidates]]
    if out:
        return out
    # If sentence splitting fails on a noisy chunk, still offer one fallback candidate.
    fallback = reformat_extractive(results[0].chunk.text.replace("\n", " ").strip())
    return [fallback] if fallback else []


def debug_extractive_candidates(
    query: str,
    results: list[SearchResult],
    *,
    max_candidates: int = 10,
) -> list[dict[str, object]]:
    ranked = _rank_extractive_candidates(query, results)
    out: list[dict[str, object]] = []
    for i, c in enumerate(ranked[:max_candidates], start=1):
        out.append(
            {
                "rank": i,
                "overlap": c.overlap,
                "intent": round(c.intent, 3),
                "retrieval_score": round(c.score, 3),
                "has_command": c.has_command,
                "has_requirement": c.has_requirement,
                "warning_like": c.warning_like,
                "text": c.text,
            }
        )
    return out


def _rank_extractive_candidates(query: str, results: list[SearchResult]) -> list[ExtractiveCandidate]:
    query_terms = _query_overlap_terms(query)
    howto_query = _looks_howto_query(query)
    definition_query = _looks_definition_query(query)
    wants_cost = _looks_cost_query(query)
    candidates: list[tuple[float, int, int, float, float, int, str, bool, bool, bool, bool]] = []
    for result in results:
        sentences = _candidate_units(result.chunk.text)
        # Cap the units considered per chunk to bound work on noisy chunks, but
        # always keep a later unit that shares a query term. Otherwise a wanted
        # table row deep in a long table (e.g. "Rush / Slingshot | 250,000$" at
        # row 15 of the Mythical tier) is silently dropped before ranking, which
        # would defeat the rare-anchor retrieval rescue upstream.
        considered = sentences[:12]
        if len(sentences) > 12 and query_terms:
            for extra in sentences[12:]:
                if query_terms & set(compact_for_match(extra).split()):
                    considered.append(extra)
        for sentence in considered:
            cleaned = reformat_extractive(sentence)
            if len(cleaned) < 20:
                continue
            words = set(compact_for_match(cleaned).split())
            key_cover = 0.0
            if _is_data_table_row(sentence) and not _table_first_cell_is_command(sentence):
                # A table row's key is its first cell ("5🌟", "Metropolis",
                # "Detonate"). key_cover is the fraction of that key the query
                # names: "metropolis" fully covers the Metropolis row's key (1.0)
                # but "town" only half-covers "Large Town" (0.5), so the row the
                # query specifically names wins. Digits are scoped to the first
                # cell too, otherwise a cost like "$5,500,000" in a later cell
                # would spuriously match a query "5".
                #
                # Command rows (first cell ``/town``) are excluded: their key is
                # a single generic word ("town") that a topic query fully covers,
                # which would let a bare ``/town`` screen command outrank the
                # actual how-to sentence. Commands match via has_command instead.
                first_cell_words = _table_first_cell_words(sentence)
                first_cell_digits = {w for w in first_cell_words if w.isdigit()}
                words = {w for w in words if not w.isdigit()} | first_cell_digits
                if first_cell_words:
                    matched_key = query_terms & first_cell_words
                    if matched_key:
                        key_cover = len(matched_key) / len(first_cell_words)
            overlap = len(query_terms & words)
            definition = _definition_match(query_terms, cleaned) if definition_query else 0
            intent = _intent_score(cleaned, howto_query=howto_query)
            intent += _query_sentence_alignment(query, cleaned)
            warning_like = _warning_like(cleaned)
            has_command = _has_command(cleaned)
            has_requirement = _has_requirement(cleaned)
            has_cost_value = _has_cost_value(cleaned)
            # If user did not ask about costs/warnings, slightly penalize those snippets.
            if warning_like and not wants_cost:
                intent -= 0.18
            # For cost-like queries, prioritize numeric/currency snippets over
            # warning-only text so we return actionable values first.
            if wants_cost:
                if has_cost_value:
                    intent += 0.30
                elif warning_like:
                    intent -= 0.12
            # A currency-only value row (e.g. price-guide "Detonate - 1,000,000$")
            # should lose to a description row of the same key when the user asks
            # what something does, not what it costs.
            elif _is_table_row(sentence) and _table_value_is_pricelike(sentence):
                intent -= 0.30
            # On a cost query, a row that names the key but carries no cost value
            # (e.g. the cross-page enchant *description* "Slingshot - Right-click
            # ...") must not claim key_cover over the price row that actually
            # answers the question. Same key name, different page: the price row
            # names it "Rush / Slingshot" (key_cover 0.5) and would otherwise lose
            # to the description's exact-name key_cover 1.0.
            if wants_cost and key_cover > 0.0 and not has_cost_value:
                key_cover = 0.0
            candidates.append(
                (
                    key_cover,
                    definition,
                    overlap,
                    intent,
                    float(result.score),
                    len(cleaned),
                    cleaned,
                    has_command,
                    has_requirement,
                    warning_like,
                    key_cover >= 1.0,
                )
            )
    # A row whose full key the query names wins outright; then (for definition
    # queries only) a definitional sentence, then higher overlap, intent, and
    # retrieval score. For ties, prefer shorter snippets so "how-to" lines beat
    # long warning prose.
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], -x[5]), reverse=True)

    ranked: list[ExtractiveCandidate] = []
    seen: set[str] = set()
    for (
        _key_cover,
        _definition,
        overlap,
        intent,
        score,
        _length,
        text,
        has_command,
        has_requirement,
        warning_like,
        key_match,
    ) in candidates:
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            ExtractiveCandidate(
                overlap=overlap,
                intent=intent,
                score=score,
                text=text,
                has_command=has_command,
                has_requirement=has_requirement,
                warning_like=warning_like,
                key_match=key_match,
            )
        )
    return ranked


@lru_cache(maxsize=2048)
def _parse_source_date(source_path: str) -> tuple[int, int, int] | None:
    """Parse a ``DD-MM-YY`` dated filename into a sortable ``(yy, mm, dd)`` tuple.

    Dated content (changelog/announcement/roadmap) is named ``10-06-26.md``.
    Returns ``(yy, mm, dd)`` — ordered so a plain tuple compare ranks newer
    files higher — or ``None`` for any non-dated page (the entire ``vista-src``
    wiki), which is how recency stays inert for normal wiki content. Cached
    because the same handful of source paths are parsed on every query.
    """
    stem = source_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2})", stem)
    if not m:
        return None
    dd, mm, yy = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if not (1 <= dd <= 31 and 1 <= mm <= 12):
        return None
    return (yy, mm, dd)


def _looks_recency_query(query: str) -> bool:
    """True when the user is asking about what's new/changed/latest.

    These queries should prefer the most recent dated entry over an older one
    that happens to be a closer topical match (the April changelog out-cosines
    the June one for "whats new"). Kept to explicit recency words so a topical
    question ("how do capsules work") never gets hijacked by a changelog.
    """
    q = compact_for_match(query)
    return bool(
        re.search(
            r"\b(latest|newest|recent|recently|new|changed|change|changes|"
            r"update|updates|updated|patch|changelog|announcement|announcements|"
            r"roadmap|upcoming|coming soon|whats new)\b",
            q,
        )
    )


def _looks_howto_query(query: str) -> bool:
    q = compact_for_match(query)
    # "how much"/"how many" are quantity questions, not procedural how-tos.
    # Treating them as how-to wrongly trips the how-to confidence gate, which
    # rejects a plain fact sentence with no command/action word — exactly the
    # shape of the answer ("New players start with a balance of $5,000"). This
    # check takes precedence even when a how-to trigger word ("start") also
    # appears later in the query.
    if re.search(r"\bhow (much|many)\b", q):
        return False
    return bool(
        re.search(
            r"\b(how|create|make|join|claim|start|open|setup|set up|do i|how to)\b",
            q,
        )
    )


def _looks_definition_query(query: str) -> bool:
    """True for explicit "what is/are X" definitional questions.

    Deliberately narrow: only ``what is``/``what are``/``what's`` and ``define``/
    ``meaning of``. It does NOT match behavioural phrasings like "what do X do",
    so those keep their existing ranking (the detonate description test relies on
    that path). The definition tier is inert for every non-definition query.
    """
    q = compact_for_match(query)
    return bool(
        re.search(r"\b(what is|what are|what s|whats|define|meaning of|what does .* mean)\b", q)
    )


def _definition_match(query_terms: set[str], sentence: str) -> int:
    """Grade how much a sentence reads like a definition of the query subject.

    2 — the sentence opens on the subject *and* has a defining verb
        ("Custom enchantments **allow** you to upgrade...").
    1 — a defining verb is present with the subject somewhere in the sentence.
    0 — neither (e.g. a how-to "To enchant an item, open your inventory...").

    Used only for definition queries, as a sort tier below ``key_cover`` (so a
    named table row still wins) and above raw ``overlap`` (so a definition beats
    a how-to that merely repeats the subject noun one extra time).
    """
    s = compact_for_match(sentence)
    if not s or not query_terms:
        return 0
    words = s.split()
    if not words:
        return 0
    has_def_verb = bool(
        re.search(
            r"\b(is|are|allows?|lets?|refers?|means?|provides?|enables?|describes?|"
            r"represents?|denotes?)\b",
            s,
        )
    )
    if not has_def_verb:
        return 0
    starts_with_subject = bool(query_terms & set(words[:4]))
    if starts_with_subject:
        return 2
    if query_terms & set(words):
        return 1
    return 0


def _intent_score(sentence: str, *, howto_query: bool) -> float:
    s = compact_for_match(sentence)
    if not s:
        return 0.0
    score = 0.0
    if howto_query:
        if re.search(r"\b(use|type|run|create|join|claim|start|open|first|then|step)\b", s):
            score += 0.35
        if "/" in sentence:
            score += 0.25
        if re.search(r"\b(failure|bankrupt|lose|penalty|upkeep|cost)\b", s):
            score -= 0.15
    return score


def _query_sentence_alignment(query: str, sentence: str) -> float:
    q = compact_for_match(query)
    s = compact_for_match(sentence)
    score = 0.0
    command_query = _looks_command_query(query)
    has_cmd = _has_command(sentence)

    if command_query:
        if has_cmd:
            score += 0.45
        else:
            score -= 0.10

    # "How do I go/tp..." should not be answered by "set ..." commands.
    nav_query = _looks_navigation_query(query)
    if nav_query:
        if re.search(r"\b(set|setting|configure|create)\b", s):
            score -= 0.30
        if re.search(r"\b(spawn|warp|teleport|tp|go)\b", s):
            score += 0.20
    return score


def _query_overlap_terms(query: str) -> set[str]:
    terms = set()
    for token in compact_for_match(query).split():
        if token in QUERY_STOPWORDS:
            continue
        # Keep single digits ("star rank 5") — they are the key discriminator for
        # numbered table rows, even though they are shorter than the min length.
        if len(token) < 2 and not token.isdigit():
            continue
        terms.add(token)
    return terms


def _is_table_row(text: str) -> bool:
    return "|" in text


def _table_cells(row: str) -> list[str]:
    return [c.strip() for c in row.split("|") if c.strip()]


def _is_data_table_row(text: str) -> bool:
    """True only for genuine key+value data rows (two or more cells).

    A real data row has a key column and at least one value column
    ("Metropolis | 24 | 5 | ..."). A one-cell ``| Shows /town commands |`` line
    is prose with a stray pipe — it has no key/value structure, so treating its
    whole sentence as a "key" produces a bogus fractional ``key_cover`` that can
    outrank the actual answer. The data-key fast-path skips such lines.
    """
    return _is_table_row(text) and len(_table_cells(text)) >= 2


def _table_first_cell_words(row: str) -> set[str]:
    """Match-normalized tokens from a table row's first cell (its key column).

    The first cell is the row's key — "5🌟", "Metropolis", "Detonate". Scoping
    matching to this cell keeps a query like "rank 5" from matching a cost like
    "$5,500,000" in a later cell, and lets a name query measure how much of the
    key it actually names (full key "metropolis" beats partial "town").
    """
    cells = _table_cells(row)
    if not cells:
        return set()
    return {t for t in compact_for_match(cells[0]).split() if t}


def _table_first_cell_is_command(row: str) -> bool:
    """True when a table row's key column is a slash command (``/town ...``).

    Command-reference rows are keyed by the command itself, whose words are
    generic topic words ("town", "plot"). Letting those count as a data key
    lets a bare ``/town`` row outrank a how-to sentence on a "create a town"
    query, so the data-key fast-path skips them and relies on command matching.
    """
    cells = _table_cells(row)
    if not cells:
        return False
    return cells[0].lstrip().startswith("/")


def _table_value_is_pricelike(row: str) -> bool:
    """True when a row's value columns are essentially just a currency amount.

    Distinguishes a price-guide row (``Detonate | 1,000,000$``) from a
    description row of the same key (``Detonate | Chance to excavate...``), so a
    "what does X do" query prefers the description over the price.
    """
    cells = _table_cells(row)
    if len(cells) < 2:
        return False
    tail = " ".join(cells[1:])
    if "$" not in tail and not re.search(r"\d", tail):
        return False
    # Value side is price-like if, after removing currency/number noise, almost
    # no descriptive words remain.
    leftover = re.sub(r"[\d,.\s$%]+", "", tail)
    leftover = re.sub(r"\b(exp|xp)\b", "", leftover, flags=re.IGNORECASE)
    return len(leftover) <= 3


def _low_confidence_match(query: str, top: ExtractiveCandidate) -> bool:
    query_terms = _query_overlap_terms(query)
    if not query_terms:
        return False
    # A row whose key column the query fully names ("rank 7" -> 7🌟 row,
    # "metropolis" -> Metropolis row) is a confident, specific hit even if the
    # rest of the row shares few query words.
    if top.key_match:
        return False
    sentence_terms = set(compact_for_match(top.text).split())
    matched = query_terms & sentence_terms
    match_count = len(matched)
    match_ratio = match_count / max(1, len(query_terms))
    howto_query = _looks_howto_query(query)
    command_query = _looks_command_query(query)
    navigation_query = _looks_navigation_query(query)
    has_action = _has_action_words(top.text)

    if command_query and top.has_command and match_count >= 1:
        return False
    if navigation_query and not has_action:
        return True
    if howto_query and not top.has_command and not has_action and match_count <= 1:
        return True
    if len(query_terms) >= 2 and match_count == 0:
        return True
    if len(query_terms) >= 3 and match_ratio < 0.34 and top.score < 0.35:
        return True
    return False


def _looks_cost_query(query: str) -> bool:
    q = compact_for_match(query)
    # "how much is X" / "how much does X cost" is a price question even though it
    # names no cost word, so the price row of X beats its description row.
    if re.search(r"\bhow much\b", q):
        return True
    return bool(
        re.search(
            r"\b(cost|price|upkeep|bankrupt|money|fee|tax|maintenance)\b",
            q,
        )
    )


def _looks_command_query(query: str) -> bool:
    q = compact_for_match(query)
    return bool(re.search(r"\b(command|cmd|syntax|type|use|what is the command)\b", q))


def _warning_like(sentence: str) -> bool:
    s = compact_for_match(sentence)
    return bool(re.search(r"\b(failure|bankrupt|lose|penalty|upkeep|cost|warning|danger)\b", s))


def _has_command(sentence: str) -> bool:
    return bool(re.search(r"(?:^|\s)/[a-z0-9_]+", sentence.lower()))


def _has_requirement(sentence: str) -> bool:
    s = compact_for_match(sentence)
    return bool(re.search(r"\b(required|requirement|must|need|at least)\b", s))


def _looks_navigation_query(query: str) -> bool:
    q = compact_for_match(query)
    return bool(re.search(r"\b(go|teleport|tp|warp|visit|get to|reach)\b", q))


def _has_action_words(sentence: str) -> bool:
    normalized = normalize_text(sentence)
    if re.search(r"(?:^|\s)/[a-z0-9_]+", normalized):
        return True

    s = compact_for_match(sentence)
    return bool(
        re.search(
            r"\b(use|type|run|go|teleport(?:s|ed|ing)?|tp|warp|portal|enter|visit|claim|create|join|buy|set)\b",
            s,
        )
    )


def _has_cost_value(sentence: str) -> bool:
    s = normalize_text(sentence)
    # Currency sign before the amount ("$5,000") or after it ("250,000$") — the
    # price guide uses the trailing form, so both must count as a cost value.
    if re.search(r"\$\s*\d", s) or re.search(r"\d[\d,.\s]*\$", s):
        return True
    if re.search(r"\b\d[\d\s,._]*\b", s) and re.search(r"\b(cost|price|money|upkeep|fee|tax)\b", s):
        return True
    if re.search(r"\b\d[\d\s,._]*\b", s) and re.search(r"\b(in-game money|dollars?)\b", s):
        return True
    return False


def _compose_answer_from_candidates(query: str, ranked: list[ExtractiveCandidate]) -> str:
    howto_query = _looks_howto_query(query)
    command_query = _looks_command_query(query)
    primary = ranked[0]
    if not howto_query:
        if command_query:
            for c in ranked:
                if c.has_command and c.overlap > 0:
                    return c.text
        return primary.text

    # For how-to, prefer actionable/requirement snippets with overlap.
    for c in ranked:
        if c.overlap <= 0:
            continue
        if command_query and c.has_command:
            primary = c
            break
        if c.has_command or c.has_requirement or not c.warning_like:
            primary = c
            break

    # A how-to question is best answered by an actionable command, but the
    # overlap-first ranking can place a high-overlap prose line above the command
    # that actually answers it: "how do i claim a land" ranks the marketing line
    # "Use the Towny plugin to claim and secure your land" (overlap 2) above
    # "...type /t claim" (overlap 1). Promote a command over a non-command
    # primary, but only when the embedder scored it at least as relevant (dense
    # score >= primary's). That gate is what separates the right command (the
    # dense winner here) from a same-topic but misleading command the embedder
    # ranked lower — e.g. town "/t rank add" for a "star rank" query, which sits
    # below the prose intro and so must NOT hijack the answer.
    if not primary.has_command:
        best_cmd: ExtractiveCandidate | None = None
        for c in ranked:
            if c.overlap > 0 and c.has_command and c.score >= primary.score:
                if best_cmd is None or c.score > best_cmd.score:
                    best_cmd = c
        if best_cmd is not None:
            primary = best_cmd

    secondary: ExtractiveCandidate | None = None
    for c in ranked:
        if c.text == primary.text:
            continue
        if c.overlap <= 0:
            continue
        if c.warning_like and not _looks_cost_query(query):
            continue
        if primary.has_command and c.has_requirement:
            secondary = c
            break
        if primary.has_requirement and c.has_command:
            secondary = c
            break

    if secondary is None:
        return primary.text

    merged = f"{primary.text} {secondary.text}"
    return _dedupe_phrases(merged)


def _dedupe_phrases(text: str) -> str:
    parts = [reformat_extractive(p) for p in split_sentences(text) if p.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = normalize_text(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out).strip()


def _candidate_units(text: str) -> list[str]:
    """Split a chunk into extractive candidate units.

    Table rows (lines starting with `|`) are kept one-per-unit so a single row
    like the Stars rank 5 row survives intact instead of being flattened into
    the surrounding table. Non-table lines are joined and split into sentences
    as before. This preserves the row structure the chunker deliberately keeps.
    """
    table_rows: list[str] = []
    prose_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            # Skip markdown separator rows like |---|---|.
            if re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
                continue
            # A description cell ends in ". ", which also marks where one logical
            # row ends if several are packed on one line. Split there so distinct
            # commands stay separate candidates; rows with no period (e.g. the
            # Stars "Perks" rows) stay whole. The negative lookbehind keeps a
            # repeatable-arg placeholder ("{nation} .. {nation}") intact: its
            # trailing ".." would otherwise sever the command from its
            # description, leaving a fragment keyed on "{nation}" that a "nation"
            # query spuriously matches.
            for part in re.split(r"(?<![.][.!?])(?<=[.!?])\s+", stripped):
                part = part.strip()
                if part:
                    table_rows.append(part)
        else:
            prose_lines.append(stripped)
    units = list(table_rows)
    if prose_lines:
        units.extend(split_sentences(" ".join(prose_lines)))
    return [u.strip() for u in units if u.strip()]


def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+|\s+-\s+|\n+", text)


def reformat_extractive(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text)
    if "|" in text:
        cells = [c.strip() for c in text.split("|") if c.strip()]
        if cells:
            # Keep every column, not just the first two. Table answers (e.g. the
            # "Perks" column of the Stars table) live past column 2, so dropping
            # the tail is what made rank/perk queries return generic intro text.
            text = " - ".join(cells)
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip(" -•")
    return text
