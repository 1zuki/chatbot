from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def project_root_from_config(config_path: Path, raw_root: str | None) -> Path:
    root = Path(raw_root or ".").expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root
