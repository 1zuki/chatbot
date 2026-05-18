from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vista_chatbot.web_api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vista chatbot web API.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "bot.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--public-wiki-base-url",
        default="",
        help="Used to convert source paths into clickable wiki links, e.g. https://wiki.vistavalley.xyz",
    )
    parser.add_argument(
        "--cors-origins",
        default="*",
        help="Comma-separated list. Use '*' for open local testing.",
    )
    parser.add_argument("--max-sessions", type=int, default=500)
    parser.add_argument("--max-session-turns", type=int, default=20)
    args = parser.parse_args()

    origins = [x.strip() for x in args.cors_origins.split(",") if x.strip()]
    app = create_app(
        config_path=args.config,
        host_for_sources=args.public_wiki_base_url,
        cors_origins=origins or ["*"],
        max_sessions=args.max_sessions,
        max_session_turns=args.max_session_turns,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
