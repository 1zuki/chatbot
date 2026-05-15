from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from vista_chatbot.evaluate import evaluate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model answers for an eval JSONL.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "bot.json"))
    parser.add_argument("--qa", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "eval" / "predictions.jsonl"))
    args = parser.parse_args()
    results = evaluate_jsonl(args.config, args.qa, args.limit, args.out)
    for r in results[:5]:
        print("Q:", r.question)
        print("A:", r.actual)
        print()
    print(f"wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
