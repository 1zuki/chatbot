from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy mc_integration.py and config next to Minescript scripts.")
    parser.add_argument("--minescript-dir", required=True, help="Folder where Minescript loads Python scripts from")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "bot.json"))
    args = parser.parse_args()

    dest = Path(args.minescript_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(PROJECT_ROOT / "mc_integration.py", dest / "mc_integration.py")

    config_path = Path(args.config).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["project_root"] = str(PROJECT_ROOT.resolve())
    (dest / "vista_chatbot_config.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Installed mc_integration.py and vista_chatbot_config.json to {dest}")


if __name__ == "__main__":
    main()
