from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import project_root_from_config, resolve_path


@dataclass(frozen=True)
class ChatConfig:
    bot_name: str = "izu"
    triggers: list[str] = field(default_factory=lambda: ["!vista"])
    hi_patterns: list[str] = field(default_factory=list)
    startup_messages: list[str] = field(default_factory=list)
    cooldown_seconds: float = 0.5
    user_cooldown_seconds: float = 0.5
    max_replies_per_minute: int = 10
    max_chat_chars: int = 248
    send_delay_seconds: float = 0.0
    ignore_after_send_seconds: float = 0.8
    max_input_chars: int = 800
    history_enabled: bool = True
    blacklisted_users: list[str] = field(default_factory=list)
    admin_names: list[str] = field(default_factory=list)
    admin_ranks: list[str] = field(default_factory=list)
    admin_only_commands: bool = False
    admin_command_names: list[str] = field(
        default_factory=lambda: [
            "status",
            "ping",
            "whoami",
            "admins",
            "clear_context",
            "history",
            "context",
            "reload_retriever",
            "stop",
            "quit",
        ]
    )
    critical_admin_commands: list[str] = field(default_factory=lambda: ["stop", "quit"])
    require_rank_for_critical_admin_commands: bool = True
    log_command_events: bool = True
    public_wiki_base_url: str = ""
    blocked_substrings: list[str] = field(default_factory=list)
    reply_prefix: str = ""


@dataclass(frozen=True)
class ModelConfig:
    enabled: bool = True
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
    fallback_to_extractive: bool = True
    llm_select_extractive: bool = False
    llm_select_max_candidates: int = 6
    llm_select_max_new_tokens: int = 8


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = True
    index_dir: str = "artifacts/retriever"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    min_score: float = 0.18
    max_context_chars: int = 1800
    chunk_chars: int = 500
    chunk_overlap: int = 100
    wiki_globs: list[str] = field(default_factory=lambda: ["**/*.md", "**/*.mdx"])


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    patterns: list[str]
    reply: str | None = None
    stop: bool = True
    case_sensitive: bool = False


@dataclass(frozen=True)
class RulesConfig:
    command_prefix: str = "!vista"
    special_cases: list[Rule] = field(default_factory=list)


@dataclass(frozen=True)
class PromptConfig:
    system: str = "You are a Minecraft wiki assistant. Answer only from context."
    answer_style: str = "Use at most 2 short sentences."


@dataclass(frozen=True)
class LoggingConfig:
    log_file: str = "artifacts/logs/autoreply.log"
    query_log_file: str = "artifacts/logs/user_queries.csv"
    query_log_enabled: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class BotConfig:
    path: Path
    project_root: Path
    chat: ChatConfig
    model: ModelConfig
    retrieval: RetrievalConfig
    rules: RulesConfig
    prompt: PromptConfig
    logging: LoggingConfig

    @property
    def wiki_dir(self) -> Path:
        return self.project_root / "wiki"

    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"

    @property
    def index_dir(self) -> Path:
        return resolve_path(self.retrieval.index_dir, base=self.project_root)

    @property
    def log_file(self) -> Path:
        return resolve_path(self.logging.log_file, base=self.project_root)

    @property
    def query_log_file(self) -> Path:
        return resolve_path(self.logging.query_log_file, base=self.project_root)

    @property
    def adapter_path(self) -> Path:
        return resolve_path(self.model.adapter_path, base=self.project_root)

    @classmethod
    def load(cls, path: str | Path) -> "BotConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        project_root = project_root_from_config(config_path, raw.get("project_root"))
        return cls(
            path=config_path,
            project_root=project_root,
            chat=_dataclass_from_dict(ChatConfig, raw.get("chat", {})),
            model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
            retrieval=_dataclass_from_dict(RetrievalConfig, raw.get("retrieval", {})),
            rules=_rules_from_dict(raw.get("rules", {})),
            prompt=_dataclass_from_dict(PromptConfig, raw.get("prompt", {})),
            logging=_dataclass_from_dict(LoggingConfig, raw.get("logging", {})),
        )


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _rules_from_dict(data: dict[str, Any]) -> RulesConfig:
    rules = []
    for item in data.get("special_cases", []):
        rules.append(_dataclass_from_dict(Rule, item))
    return RulesConfig(
        command_prefix=data.get("command_prefix", "!vista"),
        special_cases=rules,
    )
