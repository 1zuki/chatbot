from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

from .chunking import chunk_wiki_dir, iter_jsonl, write_jsonl
from .qa_data import SamplingMode, build_qa_records, qa_to_retriever_doc, qa_to_sft_row


def _resolve_qa_limit(
    wiki_count: int,
    max_qa_records: Optional[int],
    qa_target_records: Optional[int],
    qa_per_wiki_chunk: float,
    min_qa_records: int,
) -> Optional[int]:
    if qa_target_records is not None:
        target = qa_target_records
    else:
        target = int(math.ceil(max(0, wiki_count) * max(0.0, qa_per_wiki_chunk)))
        if wiki_count > 0 and min_qa_records > 0:
            target = max(target, min_qa_records)

    if max_qa_records is not None:
        target = min(target, max_qa_records)

    return max(0, target)


def build_all_datasets(
    wiki_dir: str | Path,
    qa_paths: Iterable[str | Path],
    out_dir: str | Path,
    max_qa_records: Optional[int] = None,
    wiki_max_tokens: int = 380,
    wiki_overlap_tokens: int = 70,
    qa_per_wiki_chunk: float = 1.0,
    qa_target_records: Optional[int] = None,
    min_qa_records: int = 0,
    qa_sampling: SamplingMode = "head",
    qa_seed: int = 42,
) -> dict:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    wiki_chunks_path = out / "wiki_chunks.jsonl"
    qa_retriever_path = out / "qa_retriever.jsonl"
    retriever_corpus_path = out / "retriever_corpus.jsonl"
    sft_train_path = out / "sft_train.jsonl"

    wiki_count = chunk_wiki_dir(
        wiki_dir=wiki_dir,
        out_jsonl=wiki_chunks_path,
        max_tokens=wiki_max_tokens,
        overlap_tokens=wiki_overlap_tokens,
    )

    qa_limit = _resolve_qa_limit(
        wiki_count=wiki_count,
        max_qa_records=max_qa_records,
        qa_target_records=qa_target_records,
        qa_per_wiki_chunk=qa_per_wiki_chunk,
        min_qa_records=min_qa_records,
    )

    qa_records = build_qa_records(
        qa_paths,
        max_records=qa_limit,
        sampling=qa_sampling,
        seed=qa_seed,
    )

    def qa_rows_once():
        for idx, item in enumerate(qa_records):
            yield idx, item

    qa_cache: list[dict] = []
    qa_retriever_count = 0
    with qa_retriever_path.open("w", encoding="utf-8") as retriever_f:
        for idx, item in qa_rows_once():
            qa_cache.append(item)
            retriever_f.write(json.dumps(qa_to_retriever_doc(item, idx), ensure_ascii=False) + "\n")
            qa_retriever_count += 1

    sft_count = write_jsonl(sft_train_path, (qa_to_sft_row(item) for item in qa_cache))

    def corpus_rows():
        yield from iter_jsonl(wiki_chunks_path)
        yield from iter_jsonl(qa_retriever_path)

    corpus_count = write_jsonl(retriever_corpus_path, corpus_rows())

    stats = {
        "wiki_chunks": wiki_count,
        "qa_records": len(qa_cache),
        "qa_retriever_docs": qa_retriever_count,
        "sft_rows": sft_count,
        "retriever_corpus_docs": corpus_count,
        "qa_limit_requested": qa_limit,
        "qa_per_wiki_chunk": qa_per_wiki_chunk,
        "qa_sampling": qa_sampling,
        "wiki_chunks_path": str(wiki_chunks_path),
        "qa_retriever_path": str(qa_retriever_path),
        "retriever_corpus_path": str(retriever_corpus_path),
        "sft_train_path": str(sft_train_path),
    }

    with (out / "corpus_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return stats
