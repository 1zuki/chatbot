from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vista_chatbot.config import BotConfig
from vista_chatbot.retriever import EmbeddingIndex, extractive_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the RAG retriever without Minecraft.")
    parser.add_argument("query", nargs="+", help="Question to ask")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "bot.json"))
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    cfg = BotConfig.load(args.config)
    query = " ".join(args.query)
    index = EmbeddingIndex.load(cfg.index_dir, cfg.retrieval.embedding_model)
    results = index.search(query, top_k=cfg.retrieval.top_k, min_score=cfg.retrieval.min_score)
    print(extractive_answer(query, results, max_chars=cfg.chat.max_chat_chars))
    if args.show_context:
        for r in results:
            print("\n---")
            print(f"score={r.score:.3f} source={r.chunk.source_path} heading={' > '.join(r.chunk.heading_path)}")
            print(r.chunk.text[:700])


if __name__ == "__main__":
    main()
