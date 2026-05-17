from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from collections import defaultdict, deque
from pathlib import Path

from .config import BotConfig
from .conversation import ConversationMemory, SeenMessage
from .llm import LocalGenerator, PromptBuilder
from .logging_utils import configure_logging
from .retriever import EmbeddingIndex, SearchResult
from .rules import RuleEngine
from .text import (
    is_hi_pattern,
    normalize_text,
    parse_minecraft_chat,
    safe_truncate,
    sanitize_chat_output,
    split_for_chat,
)


class MinuteRateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.timestamps: deque[float] = deque()

    def allow(self, now: float) -> bool:
        if self.max_per_minute <= 0:
            return True
        while self.timestamps and now - self.timestamps[0] > 60.0:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_per_minute:
            return False
        self.timestamps.append(now)
        return True


class BotEngine:
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = configure_logging(config.log_file, config.logging.log_level)
        self.rules = RuleEngine(config.rules)
        self.prompt_builder = PromptBuilder(config)
        self.generator = LocalGenerator(config, logger=self.logger)
        self.retriever: EmbeddingIndex | None = None
        self.memory = ConversationMemory(max_turns=max(8, config.model.max_context_messages * 2))
        self.stop_requested = False
        self._global_last_reply = 0.0
        self._user_last_reply: dict[str, float] = defaultdict(float)
        self._minute_limiter = MinuteRateLimiter(config.chat.max_replies_per_minute)
        self._sent_messages: deque[SeenMessage] = deque(maxlen=24)

    @classmethod
    def from_config_path(cls, path: str | Path) -> "BotEngine":
        return cls(BotConfig.load(path))

    def warmup(self) -> None:
        self.logger.info("Starting Vista chatbot from %s", self.config.path)
        self.logger.info("Project root: %s", self.config.project_root)
        if self.config.retrieval.enabled:
            self.retriever = EmbeddingIndex.load(self.config.index_dir, self.config.retrieval.embedding_model)
            self.logger.info("Loaded retriever with %d chunks", len(self.retriever.chunks))
            # Load embedding model early to avoid first-message lag.
            self.retriever._load_model()
        if self.config.model.enabled or self.config.model.llm_select_extractive:
            self.generator.warmup()
        self.logger.info("Warmup complete")

    def handle_text(self, raw_text: str) -> str | None:
        now = time.monotonic()
        parsed = parse_minecraft_chat(raw_text)
        content = parsed.content[: self.config.chat.max_input_chars].strip()
        speaker = parsed.speaker or "unknown"
        rank = parsed.rank

        if not content:
            return None
        if self._is_recent_self_echo(content, now):
            return None
        if self._blocked(content):
            return None

        prefixed = self._strip_query_prefix(content)
        if prefixed is None:
            return None

        rule_result = self.rules.match(prefixed)
        if rule_result.matched:
            self.logger.info("Rule matched: %s speaker=%s rank=%s text=%r", rule_result.rule_name, speaker, rank, prefixed)
            return self._finalize_reply(rule_result.reply) if rule_result.reply else None

        command_reply = self._handle_command(prefixed, speaker=speaker, rank=rank)
        if command_reply is not None:
            return command_reply

        if not self._cooldown_allows(speaker, now):
            return None

        query = prefixed
        if not query or is_hi_pattern(query, self.config.chat.hi_patterns):
            return self._finalize_reply(f"Meow. Ask me a wiki question with {self.config.rules.command_prefix}.")

        self.memory.add_user(query)
        results = self._retrieve(query)
        if not results:
            reply = "I don't know from the wiki yet."
        else:
            prompt = self.prompt_builder.build(
                query=query,
                results=results,
                history=self.memory.recent(self.config.model.max_context_messages),
            )
            reply = self.generator.generate_or_fallback(
                prompt=prompt,
                query=query,
                results=results,
                max_chat_chars=self.config.chat.max_chat_chars,
            )
        self.memory.add_assistant(reply)
        self.logger.info("Reply speaker=%s rank=%s query=%r reply=%r", speaker, rank, query, reply)
        return self._finalize_reply(reply)

    def split_reply(self, reply: str) -> list[str]:
        return split_for_chat(reply, self.config.chat.max_chat_chars)

    def remember_sent(self, parts: list[str]) -> None:
        now = time.monotonic()
        for part in parts:
            normalized = normalize_text(part)
            if normalized:
                self._sent_messages.append(SeenMessage(normalized, now))

    def _retrieve(self, query: str) -> list[SearchResult]:
        if not self.config.retrieval.enabled:
            return []
        if self.retriever is None:
            self.logger.warning("Retriever is enabled but not loaded")
            return []
        return self.retriever.search(
            query,
            top_k=self.config.retrieval.top_k,
            min_score=self.config.retrieval.min_score,
        )

    def _blocked(self, text: str) -> bool:
        t = normalize_text(text)
        return any(normalize_text(s) in t for s in self.config.chat.blocked_substrings)

    def _is_recent_self_echo(self, text: str, now: float) -> bool:
        normalized = normalize_text(text)
        ttl = self.config.chat.ignore_after_send_seconds
        while self._sent_messages and now - self._sent_messages[0].timestamp > ttl:
            self._sent_messages.popleft()
        for item in self._sent_messages:
            if item.text == normalized:
                return True
            # Some servers censor or slightly reformat outgoing chat before it is
            # echoed back. A short-lived fuzzy match prevents reply loops.
            if item.text and (item.text in normalized or normalized in item.text):
                return True
            if len(item.text) >= 32 and len(normalized) >= 32:
                if SequenceMatcher(None, item.text, normalized).ratio() >= 0.88:
                    return True
        return False

    def _cooldown_allows(self, speaker: str, now: float) -> bool:
        if now - self._global_last_reply < self.config.chat.cooldown_seconds:
            return False
        key = normalize_text(speaker or "unknown")
        if now - self._user_last_reply[key] < self.config.chat.user_cooldown_seconds:
            return False
        if not self._minute_limiter.allow(now):
            return False
        self._global_last_reply = now
        self._user_last_reply[key] = now
        return True

    def _strip_query_prefix(self, text: str) -> str | None:
        prefixes = self._query_prefixes()
        if not prefixes:
            return text.strip()

        # Normal case after parse_minecraft_chat(): content starts with !vista.
        for prefix in prefixes:
            direct = self._strip_one_prefix_at_start(text, prefix)
            if direct is not None:
                return direct

        # Safety net for heavily decorated server chat where Minescript gives the
        # full rendered line, e.g. `🏕 ➟ TOPAZ ➡ !vista what is fluff`.
        # The boundary check avoids matching `!vista2` while still allowing rank
        # prefixes before the command.
        for prefix in prefixes:
            pat = re.compile(rf"(?i)(^|[\s:>»›➡➜-]){re.escape(prefix)}(?=$|\s)")
            match = pat.search(text)
            if match:
                return text[match.end() :].strip()
        return None

    def _query_prefixes(self) -> list[str]:
        seen: set[str] = set()
        prefixes: list[str] = []
        for value in [self.config.rules.command_prefix, *self.config.chat.triggers]:
            value = value.strip()
            if not value or value.lower() in seen:
                continue
            seen.add(value.lower())
            prefixes.append(value)
        return prefixes

    @staticmethod
    def _strip_one_prefix_at_start(text: str, prefix: str) -> str | None:
        if not text.lower().startswith(prefix.lower()):
            return None
        remaining = text[len(prefix) :]
        if remaining and not remaining[0].isspace():
            return None
        return remaining.strip()

    def _handle_command(self, prefixed_text: str, *, speaker: str, rank: str | None) -> str | None:
        raw = prefixed_text.strip()
        if not raw:
            return self._command_help()

        parts = raw.split(None, 1)
        cmd = parts[0].lower()

        aliases = {
            "reload": "reload_retriever",
            "reloadretriever": "reload_retriever",
            "clear": "clear_context",
            "clearcontext": "clear_context",
        }
        cmd = aliases.get(cmd, cmd)

        if cmd not in {
            "stop",
            "quit",
            "status",
            "ping",
            "help",
            "whoami",
            "admins",
            "clear_context",
            "reload_retriever",
        }:
            return None

        admin_required = self._command_requires_admin(cmd)
        allowed = (not admin_required) or self._is_admin(speaker=speaker, rank=rank, command=cmd)
        self._log_command_event(
            cmd=cmd,
            speaker=speaker,
            rank=rank,
            allowed=allowed,
            reason="admin_required" if admin_required else "open_command",
        )

        if not allowed:
            return self._finalize_reply("No permission.")

        if cmd in {"stop", "quit"}:
            self.stop_requested = True
            return self._finalize_reply("Stopping Vista chatbot.")
        if cmd in {"status", "ping"}:
            chunks = len(self.retriever.chunks) if self.retriever else 0
            llm_chat = "on" if self.config.model.enabled and self.generator.loaded else "off"
            selector = "on" if self.config.model.llm_select_extractive and self.generator.loaded else "off"
            admin = "yes" if self._is_admin(speaker=speaker, rank=rank, command=cmd) else "no"
            return self._finalize_reply(
                f"Vista online. speaker={speaker} rank={rank or '-'} admin={admin} wiki_chunks={chunks} "
                f"llm_chat={llm_chat} llm_selector={selector}"
            )
        if cmd == "whoami":
            admin = "yes" if self._is_admin(speaker=speaker, rank=rank, command=cmd) else "no"
            return self._finalize_reply(f"Parsed speaker={speaker}, rank={rank or '-'}. admin={admin}.")
        if cmd == "admins":
            names = ", ".join(self.config.chat.admin_names) if self.config.chat.admin_names else "-"
            ranks = ", ".join(self.config.chat.admin_ranks) if self.config.chat.admin_ranks else "-"
            return self._finalize_reply(f"Admin names: {names}. Admin ranks: {ranks}.")
        if cmd == "clear_context":
            self.memory.clear()
            return self._finalize_reply("Conversation context cleared.")
        if cmd == "reload_retriever":
            return self._finalize_reply(self._reload_retriever_message())
        if cmd == "help":
            return self._command_help()

        return self._command_help()

    def _command_requires_admin(self, cmd: str) -> bool:
        if self.config.chat.admin_only_commands:
            return True
        return cmd in {normalize_text(x) for x in self.config.chat.admin_command_names}

    def _is_admin(self, *, speaker: str, rank: str | None, command: str) -> bool:
        admin_names = {normalize_text(x) for x in self.config.chat.admin_names}
        admin_ranks = {normalize_text(x) for x in self.config.chat.admin_ranks}
        speaker_key = normalize_text(speaker)
        rank_key = normalize_text(rank or "")
        by_name = bool(speaker_key and speaker_key in admin_names)
        by_rank = bool(rank_key and rank_key in admin_ranks)

        critical = normalize_text(command) in {normalize_text(x) for x in self.config.chat.critical_admin_commands}
        if (
            critical
            and self.config.chat.require_rank_for_critical_admin_commands
            and admin_ranks
        ):
            # Hardening against name spoof / nickname changes for sensitive commands.
            return by_rank
        return by_name or by_rank

    def _log_command_event(
        self,
        *,
        cmd: str,
        speaker: str,
        rank: str | None,
        allowed: bool,
        reason: str,
    ) -> None:
        if not self.config.chat.log_command_events:
            return
        self.logger.info(
            "Command %s: cmd=%s speaker=%s rank=%s reason=%s",
            "ALLOW" if allowed else "DENY",
            cmd,
            speaker,
            rank or "-",
            reason,
        )

    def _reload_retriever_message(self) -> str:
        if not self.config.retrieval.enabled:
            return "Retriever disabled in config."
        try:
            self.retriever = EmbeddingIndex.load(self.config.index_dir, self.config.retrieval.embedding_model)
            chunks = len(self.retriever.chunks)
            return f"Retriever reloaded. wiki_chunks={chunks}."
        except Exception as exc:
            self.logger.exception("Failed to reload retriever: %s", exc)
            return f"Retriever reload failed: {str(exc)[:120]}"

    def _command_help(self) -> str:
        p = self.config.rules.command_prefix
        return (
            f"Use: {p} <question>. Commands: {p} help, {p} status, {p} whoami, "
            f"{p} admins, {p} clear_context, {p} reload_retriever, {p} stop"
        )

    def _finalize_reply(self, reply: str | None) -> str | None:
        if reply is None:
            return None
        reply = sanitize_chat_output(reply)
        if self.config.chat.reply_prefix:
            reply = f"{self.config.chat.reply_prefix}{reply}"
        return safe_truncate(reply, self.config.chat.max_chat_chars * 3)
