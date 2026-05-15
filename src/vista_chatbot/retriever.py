from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .chunking import iter_jsonl
from .config import RetrievalConfig
from .text_cleaning import normalize_spaces


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float
    source: str = ""
    kind: str = ""
    title: str = ""
    path: str = ""
    section: str = ""

    def compact(self, max_chars: int = 600) -> str:
        prefix_bits = []
        if self.title:
            prefix_bits.append(self.title)
        if self.section:
            prefix_bits.append(self.section)
        prefix = " / ".join(prefix_bits)
        body = self.text[:max_chars].strip()
        if len(self.text) > max_chars:
            body += "..."
        if prefix:
            return f"[{prefix}] {body}"
        return body


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "sentence-transformers is required for retrieval. Install requirements.txt first."
        ) from exc
    return SentenceTransformer


def _require_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("faiss-cpu or faiss-gpu is required for retrieval indexing/search.") from exc
    return faiss


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class FaissRetriever:
    def __init__(self, index_dir: str | Path, embedding_model: Optional[str] = None):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.config_path = self.index_dir / "config.json"
        self.metadata_path = self.index_dir / "metadata.jsonl"
        self.index_path = self.index_dir / "faiss.index"
        if not self.index_path.exists() or not self.metadata_path.exists() or not self.config_path.exists():
            raise FileNotFoundError(
                f"Retriever index missing in {self.index_dir}. Run scripts/build_retriever.py first."
            )

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        model_name = embedding_model or config["embedding_model"]
        SentenceTransformer = _require_sentence_transformers()
        faiss = _require_faiss()

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(self.index_path))
        self.docs = list(iter_jsonl(self.metadata_path))

    @staticmethod
    def build(
        corpus_path: str | Path,
        index_dir: str | Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
    ) -> dict:
        corpus = list(iter_jsonl(corpus_path))
        if not corpus:
            raise ValueError(f"No corpus rows found in {corpus_path}")

        SentenceTransformer = _require_sentence_transformers()
        faiss = _require_faiss()
        model = SentenceTransformer(embedding_model)

        texts = [normalize_spaces(str(row.get("text", ""))) for row in corpus]
        vectors: List[np.ndarray] = []
        for batch in _batched(texts, batch_size):
            emb = model.encode(
                list(batch),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            vectors.append(emb.astype("float32"))
        matrix = np.vstack(vectors)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        out = Path(index_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out / "faiss.index"))
        with (out / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in corpus:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_model": embedding_model,
                    "size": len(corpus),
                    "dim": int(matrix.shape[1]),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
        return {"index_dir": str(out), "size": len(corpus), "dim": int(matrix.shape[1])}

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> List[RetrievedDoc]:
        query = normalize_spaces(query)
        if not query:
            return []
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(vector, max(1, top_k))
        out: List[RetrievedDoc] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.docs) or score < min_score:
                continue
            row = self.docs[idx]
            out.append(
                RetrievedDoc(
                    id=str(row.get("id", idx)),
                    text=str(row.get("text", "")),
                    score=float(score),
                    source=str(row.get("source", "")),
                    kind=str(row.get("kind", "")),
                    title=str(row.get("title", "")),
                    path=str(row.get("path", "")),
                    section=str(row.get("section", "")),
                )
            )
        return out


class NullRetriever:
    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> List[RetrievedDoc]:
        return []


def load_retriever(project_root: Path, cfg: RetrievalConfig):
    if not cfg.enabled:
        return NullRetriever()
    index_dir = Path(cfg.index_dir).expanduser()
    if not index_dir.is_absolute():
        index_dir = project_root / index_dir
    if not index_dir.exists():
        return NullRetriever()
    return FaissRetriever(index_dir, embedding_model=cfg.embedding_model)
