from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from vista_chatbot.config import load_config, save_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy autoreply.py and runtime config into the Minecraft minescript folder.")
    parser.add_argument("--minescript-dir", required=True, help="Your .minecraft/minescript directory.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "bot.json"))
    args = parser.parse_args()

    minescript_dir = Path(args.minescript_dir).expanduser().resolve()
    minescript_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    # Ensure the installed config points to the actual repo root, not the
    # minescript folder.
    config = load_config(args.config)
    runtime_config = minescript_dir / "vista_chatbot_config.json"
    save_runtime_config(config, runtime_config)

    target_autoreply = minescript_dir / "autoreply.py"
    shutil.copy2(ROOT / "autoreply.py", target_autoreply)

    print(f"Installed {target_autoreply}")
    print(f"Installed {runtime_config}")
    print("Run it in Minecraft with: \\autoreply")


if __name__ == "__main__":
    main()
