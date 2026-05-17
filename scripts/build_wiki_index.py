from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vista_chatbot.config import BotConfig
from vista_chatbot.retriever import EmbeddingIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local wiki retrieval index.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "bot.json"))
    args = parser.parse_args()

    cfg = BotConfig.load(args.config)
    index = EmbeddingIndex.build(wiki_dir=cfg.wiki_dir, index_dir=cfg.index_dir, config=cfg.retrieval)
    print(f"Built retriever: {len(index.chunks)} chunks -> {cfg.index_dir}")


if __name__ == "__main__":
    main()
