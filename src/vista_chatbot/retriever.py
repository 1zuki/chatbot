from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

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
    "what",
    "wat",
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
        scores = self.embeddings @ q
        if scores.size == 0:
            return []
        n = min(max(top_k * 3, top_k), scores.size)
        candidate_idx = np.argpartition(scores, -n)[-n:]
        ranked = sorted(((int(i), float(scores[i])) for i in candidate_idx), key=lambda x: x[1], reverse=True)
        out: list[SearchResult] = []
        seen_sources: set[tuple[str, tuple[str, ...]]] = set()
        for idx, score in ranked:
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            key = (chunk.source_path, tuple(chunk.heading_path))
            # Keep result diversity, but still allow another chunk if few results.
            if key in seen_sources and len(out) >= math.ceil(top_k / 2):
                continue
            seen_sources.add(key)
            out.append(SearchResult(chunk=chunk, score=score))
            if len(out) >= top_k:
                break
        return out

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
    return safe_truncate(f"Wiki says: {answer}", max_chars)


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
    wants_cost = _looks_cost_query(query)
    candidates: list[tuple[int, float, float, int, str, bool, bool, bool]] = []
    for result in results:
        text = result.chunk.text.replace("\n", " ")
        sentences = [s.strip() for s in split_sentences(text) if len(s.strip()) >= 20]
        for sentence in sentences[:8]:
            cleaned = reformat_extractive(sentence)
            if len(cleaned) < 20:
                continue
            words = set(compact_for_match(cleaned).split())
            overlap = len(query_terms & words)
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
            candidates.append(
                (
                    overlap,
                    intent,
                    float(result.score),
                    len(cleaned),
                    cleaned,
                    has_command,
                    has_requirement,
                    warning_like,
                )
            )
    # Prefer higher lexical overlap, then intent score, then retrieval score.
    # For ties, prefer shorter snippets so "how-to" lines beat long warning prose.
    candidates.sort(key=lambda x: (x[0], x[1], x[2], -x[3]), reverse=True)

    ranked: list[ExtractiveCandidate] = []
    seen: set[str] = set()
    for overlap, intent, score, _length, text, has_command, has_requirement, warning_like in candidates:
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
            )
        )
    return ranked


def _looks_howto_query(query: str) -> bool:
    q = compact_for_match(query)
    return bool(
        re.search(
            r"\b(how|create|make|join|claim|start|open|setup|set up|do i|how to)\b",
            q,
        )
    )


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
        if len(token) < 2:
            continue
        if token in QUERY_STOPWORDS:
            continue
        terms.add(token)
    return terms


def _low_confidence_match(query: str, top: ExtractiveCandidate) -> bool:
    query_terms = _query_overlap_terms(query)
    if not query_terms:
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
    if re.search(r"\$\s*\d", s):
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


def split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[.!?])\s+|\s+-\s+|\n+", text)


def reformat_extractive(text: str) -> str:
    import re

    text = re.sub(r"^#+\s*", "", text)
    if "|" in text:
        cells = [c.strip() for c in text.split("|") if c.strip()]
        if cells:
            if len(cells) >= 2:
                text = f"{cells[0]} - {cells[1]}"
            else:
                text = cells[0]
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text).strip(" -•")
    return text
