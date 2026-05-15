from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from vista_chatbot.chunking import iter_jsonl
from vista_chatbot.retriever import FaissRetriever


def _count_jsonl(path: Path) -> int:
    n = 0
    for _ in iter_jsonl(path):
        n += 1
    return n


def _qa_limit(wiki_count: int, qa_per_wiki_doc: float, qa_target_docs: Optional[int], max_qa_docs: Optional[int]) -> int:
    if qa_target_docs is not None:
        target = qa_target_docs
    else:
        target = int(math.ceil(max(0, wiki_count) * max(0.0, qa_per_wiki_doc)))
    if max_qa_docs is not None:
        target = min(target, max_qa_docs)
    return max(0, target)


def _write_from_split_files(wiki_path: Path, qa_path: Path, out_path: Path, qa_limit: int) -> dict:
    wiki_count = 0
    qa_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(wiki_path):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            wiki_count += 1
        for row in iter_jsonl(qa_path):
            if qa_count >= qa_limit:
                break
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            qa_count += 1
    return {
        "balanced_corpus_path": str(out_path),
        "wiki_docs": wiki_count,
        "qa_docs": qa_count,
        "total_docs": wiki_count + qa_count,
        "source_mode": "split_files",
    }


def _write_from_combined_corpus(corpus_path: Path, out_path: Path, qa_limit: int) -> dict:
    wiki_count = 0
    qa_count = 0
    other_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(corpus_path):
            kind = str(row.get("kind", ""))
            if kind == "qa":
                if qa_count >= qa_limit:
                    continue
                qa_count += 1
            elif kind == "wiki":
                wiki_count += 1
            else:
                other_count += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "balanced_corpus_path": str(out_path),
        "wiki_docs": wiki_count,
        "qa_docs": qa_count,
        "other_docs": other_count,
        "total_docs": wiki_count + qa_count + other_count,
        "source_mode": "combined_corpus",
    }


def prepare_balanced_corpus(
    corpus_path: str | Path,
    out_dir: str | Path,
    qa_per_wiki_doc: float = 1.0,
    qa_target_docs: Optional[int] = None,
    max_qa_docs: Optional[int] = None,
    wiki_corpus: Optional[str | Path] = None,
    qa_corpus: Optional[str | Path] = None,
) -> dict:
    corpus = Path(corpus_path).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    balanced_path = out / "balanced_retriever_corpus.jsonl"

    guessed_wiki = Path(wiki_corpus).expanduser().resolve() if wiki_corpus else corpus.parent / "wiki_chunks.jsonl"
    guessed_qa = Path(qa_corpus).expanduser().resolve() if qa_corpus else corpus.parent / "qa_retriever.jsonl"

    if guessed_wiki.exists() and guessed_qa.exists():
        wiki_count = _count_jsonl(guessed_wiki)
        limit = _qa_limit(
            wiki_count=wiki_count,
            qa_per_wiki_doc=qa_per_wiki_doc,
            qa_target_docs=qa_target_docs,
            max_qa_docs=max_qa_docs,
        )
        stats = _write_from_split_files(guessed_wiki, guessed_qa, balanced_path, limit)
        stats.update({"qa_limit_requested": limit, "wiki_corpus": str(guessed_wiki), "qa_corpus": str(guessed_qa)})
        return stats

    # Fallback: only a combined corpus is available. This may need to scan the combined file.
    wiki_count = 0
    for row in iter_jsonl(corpus):
        if str(row.get("kind", "")) == "wiki":
            wiki_count += 1
    limit = _qa_limit(
        wiki_count=wiki_count,
        qa_per_wiki_doc=qa_per_wiki_doc,
        qa_target_docs=qa_target_docs,
        max_qa_docs=max_qa_docs,
    )
    stats = _write_from_combined_corpus(corpus, balanced_path, limit)
    stats.update({"qa_limit_requested": limit, "corpus": str(corpus)})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS retriever index from a balanced retriever corpus.")
    parser.add_argument("--corpus", default=str(ROOT / "artifacts" / "corpus" / "retriever_corpus.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "retriever"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--balance-to-wiki",
        dest="balance_to_wiki",
        action="store_true",
        default=True,
        help="Index all wiki docs and only about the same amount of QA docs. Enabled by default.",
    )
    parser.add_argument(
        "--no-balance-to-wiki",
        dest="balance_to_wiki",
        action="store_false",
        help="Index the corpus exactly as provided.",
    )
    parser.add_argument(
        "--qa-per-wiki-doc",
        type=float,
        default=1.0,
        help="QA docs to index per wiki doc when balancing. Default 1.0.",
    )
    parser.add_argument("--qa-target-docs", type=int, default=None, help="Exact QA doc target. Overrides ratio.")
    parser.add_argument("--max-qa-docs", type=int, default=None, help="Hard QA cap for indexing.")
    parser.add_argument("--wiki-corpus", default=None, help="Optional wiki_chunks.jsonl path.")
    parser.add_argument("--qa-corpus", default=None, help="Optional qa_retriever.jsonl path.")
    args = parser.parse_args()

    effective_corpus = args.corpus
    balance_stats = None
    if args.balance_to_wiki:
        balance_stats = prepare_balanced_corpus(
            corpus_path=args.corpus,
            out_dir=args.out,
            qa_per_wiki_doc=args.qa_per_wiki_doc,
            qa_target_docs=args.qa_target_docs,
            max_qa_docs=args.max_qa_docs,
            wiki_corpus=args.wiki_corpus,
            qa_corpus=args.qa_corpus,
        )
        effective_corpus = balance_stats["balanced_corpus_path"]

    stats = FaissRetriever.build(
        corpus_path=effective_corpus,
        index_dir=args.out,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )
    if balance_stats is not None:
        stats["balance"] = balance_stats
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
