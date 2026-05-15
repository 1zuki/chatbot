from __future__ import annotations

"""
Minescript entrypoint.

Copy this file into your Minecraft minescript folder, plus a generated
vista_chatbot_config.json beside it. Then run in-game as:

    \autoreply

No API server and no exported environment variables are required.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import minescript as m

CONFIG_FILE_NAME = "vista_chatbot_config.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_config_paths(argv: list[str]) -> list[Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    ns, _ = parser.parse_known_args(argv[1:])

    script_dir = Path(__file__).resolve().parent
    candidates = []
    if ns.config:
        candidates.append(Path(ns.config).expanduser())
    candidates.append(script_dir / CONFIG_FILE_NAME)
    candidates.append(script_dir / "configs" / "bot.json")
    candidates.append(script_dir.parent / "configs" / "bot.json")
    return candidates


def _find_config(argv: list[str]) -> Path:
    for path in _candidate_config_paths(argv):
        path = path.resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find vista_chatbot_config.json. Run scripts/install_minescript_entry.py "
        "or pass --config /path/to/bot.json."
    )


def _bootstrap_project(config_path: Path) -> None:
    raw = _load_json(config_path)
    root = Path(raw.get("project_root", ".")).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"Project src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    try:
        config_path = _find_config(sys.argv)
        _bootstrap_project(config_path)
        from vista_chatbot.runtime import BotEngine

        engine = BotEngine.from_config_path(config_path)
        m.echo(f"Vista chatbot loading from {config_path}")
        engine.warmup()
        m.echo("Vista chatbot loaded")
        for msg in engine.config.chat.startup_messages:
            m.chat(msg)

        with m.EventQueue() as eq:
            eq.register_chat_listener()
            while not engine.stop_requested:
                event = eq.get()
                if event.type != m.EventType.CHAT:
                    continue
                reply = engine.handle_text(event.message)
                if reply is None:
                    continue
                parts = engine.split_reply(reply)
                engine.remember_sent(parts)
                for part in parts:
                    time.sleep(engine.config.chat.send_delay_seconds)
                    m.chat(part)
    except Exception as exc:
        # Use echo, not chat, so errors do not get broadcast to the server.
        m.echo(f"Vista chatbot failed: {exc}")
        raise
    finally:
        m.echo("Vista chatbot stopped")


if __name__ == "__main__":
    main()
