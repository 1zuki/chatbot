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


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


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
        return "I don't know from the wiki yet."
    ranked = _rank_extractive_candidates(query, results)
    if ranked and ranked[0][0] > 0:
        answer = ranked[0][2]
    else:
        top = results[0].chunk
        answer = top.text.replace("\n", " ").strip()
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
    out = [text for _, _, text in ranked[:max_candidates]]
    if out:
        return out
    # If sentence splitting fails on a noisy chunk, still offer one fallback candidate.
    fallback = reformat_extractive(results[0].chunk.text.replace("\n", " ").strip())
    return [fallback] if fallback else []


def _rank_extractive_candidates(query: str, results: list[SearchResult]) -> list[tuple[int, float, str]]:
    query_terms = {t for t in compact_for_match(query).split() if len(t) >= 3}
    howto_query = _looks_howto_query(query)
    candidates: list[tuple[int, float, float, int, str]] = []
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
            candidates.append((overlap, intent, float(result.score), len(cleaned), cleaned))
    # Prefer higher lexical overlap, then intent score, then retrieval score.
    # For ties, prefer shorter snippets so "how-to" lines beat long warning prose.
    candidates.sort(key=lambda x: (x[0], x[1], x[2], -x[3]), reverse=True)

    ranked: list[tuple[int, float, str]] = []
    seen: set[str] = set()
    for overlap, _intent, score, _length, text in candidates:
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((overlap, score, text))
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


def split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[.!?])\s+|\s+-\s+|\n+", text)


def reformat_extractive(text: str) -> str:
    import re

    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -•")
    return text
