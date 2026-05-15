from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigError(RuntimeError):
    pass


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass(frozen=True)
class ChatConfig:
    bot_name: str = "izu"
    triggers: List[str] = field(default_factory=lambda: ["izu"])
    hi_patterns: List[str] = field(default_factory=list)
    startup_messages: List[str] = field(default_factory=list)
    cooldown_seconds: float = 4.0
    user_cooldown_seconds: float = 8.0
    max_replies_per_minute: int = 10
    max_chat_chars: int = 248
    send_delay_seconds: float = 1.2
    ignore_after_send_seconds: float = 0.8
    max_input_chars: int = 800
    admin_names: List[str] = field(default_factory=list)
    admin_only_commands: bool = False
    blocked_substrings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path: str = "artifacts/lora_full"
    load_in_4bit: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 96
    temperature: float = 0.35
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.12
    do_sample: bool = True
    max_context_messages: int = 6


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = True
    index_dir: str = "artifacts/retriever"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    min_score: float = 0.18
    max_context_chars: int = 1800


@dataclass(frozen=True)
class LoggingConfig:
    log_file: str = "artifacts/logs/autoreply.log"
    log_level: str = "INFO"


@dataclass(frozen=True)
class BotConfig:
    project_root: Path
    chat: ChatConfig
    model: ModelConfig
    retrieval: RetrievalConfig
    logging: LoggingConfig
    config_path: Optional[Path] = None

    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"

    @property
    def adapter_dir(self) -> Path:
        return _as_path(self.project_root, self.model.adapter_path)

    @property
    def retriever_dir(self) -> Path:
        return _as_path(self.project_root, self.retrieval.index_dir)

    @property
    def log_path(self) -> Path:
        return _as_path(self.project_root, self.logging.log_file)


def _dataclass_from_dict(cls: type, data: Dict[str, Any]) -> Any:
    field_names = {name for name in cls.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
    filtered = {key: value for key, value in data.items() if key in field_names}
    return cls(**filtered)


def load_config(config_path: str | Path) -> BotConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_root = raw.get("project_root", ".")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        # A relative root is relative to the config file location. This lets the
        # repo config use project_root='.' while the Minescript copy can use an
        # absolute path written by scripts/install_minescript_entry.py.
        root = (path.parent / root)
    root = root.resolve()

    return BotConfig(
        project_root=root,
        chat=_dataclass_from_dict(ChatConfig, raw.get("chat", {})),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        retrieval=_dataclass_from_dict(RetrievalConfig, raw.get("retrieval", {})),
        logging=_dataclass_from_dict(LoggingConfig, raw.get("logging", {})),
        config_path=path,
    )


def save_runtime_config(config: BotConfig, target_path: str | Path) -> None:
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "project_root": str(config.project_root),
        "chat": config.chat.__dict__,
        "model": config.model.__dict__,
        "retrieval": config.retrieval.__dict__,
        "logging": config.logging.__dict__,
    }
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
