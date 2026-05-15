from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from vista_chatbot.dataset_build import build_all_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced wiki chunks, QA SFT rows, and retriever corpus.")
    parser.add_argument("--wiki", default=str(ROOT / "wiki"), help="Path to wiki md/mdx directory.")
    parser.add_argument("--qa", nargs="+", required=True, help="QA .json or .jsonl files with question/answer/source.")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "corpus"), help="Output corpus directory.")
    parser.add_argument(
        "--qa-per-wiki-chunk",
        type=float,
        default=1.0,
        help="How many QA examples to keep per wiki chunk. Default 1.0 means roughly equal wiki and QA counts.",
    )
    parser.add_argument(
        "--qa-target-records",
        type=int,
        default=None,
        help="Exact QA record target. Overrides --qa-per-wiki-chunk when set.",
    )
    parser.add_argument(
        "--max-qa-records",
        type=int,
        default=None,
        help="Hard cap after balance calculation. Useful as an extra safety limit.",
    )
    parser.add_argument(
        "--min-qa-records",
        type=int,
        default=0,
        help="Minimum QA rows when the wiki is tiny. Still respects --max-qa-records.",
    )
    parser.add_argument(
        "--qa-sampling",
        choices=("head", "reservoir"),
        default="head",
        help="head is fast and stops early; reservoir is random but scans the full QA dataset.",
    )
    parser.add_argument("--qa-seed", type=int, default=42, help="Random seed for reservoir sampling.")
    parser.add_argument("--wiki-max-tokens", type=int, default=380)
    parser.add_argument("--wiki-overlap-tokens", type=int, default=70)
    args = parser.parse_args()

    stats = build_all_datasets(
        wiki_dir=args.wiki,
        qa_paths=args.qa,
        out_dir=args.out,
        max_qa_records=args.max_qa_records,
        wiki_max_tokens=args.wiki_max_tokens,
        wiki_overlap_tokens=args.wiki_overlap_tokens,
        qa_per_wiki_chunk=args.qa_per_wiki_chunk,
        qa_target_records=args.qa_target_records,
        min_qa_records=args.min_qa_records,
        qa_sampling=args.qa_sampling,
        qa_seed=args.qa_seed,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
