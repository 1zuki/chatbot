from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from vista_chatbot.config import load_config
from vista_chatbot.runtime import BotEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local non-Minescript smoke test.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "bot.json"))
    parser.add_argument("--message", default="izu how do I claim land?")
    parser.add_argument("--skip-model", action="store_true", help="Only test parsing/trigger logic; do not load the model.")
    args = parser.parse_args()

    engine = BotEngine(load_config(args.config))
    if args.skip_model:
        parsed = engine.parse_chat(args.message)
        print(parsed)
        print("trigger:", engine.should_trigger(parsed.normalized_content))
        print("query:", engine.remove_trigger(parsed.content))
        return
    print(engine.handle_text(args.message))


if __name__ == "__main__":
    main()
