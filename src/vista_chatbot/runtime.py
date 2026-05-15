from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .config import BotConfig, load_config
from .logging_utils import setup_logging
from .model import LocalChatModel, clear_history
from .text_cleaning import normalize_for_match, sanitize_chat_output


@dataclass
class ParsedChat:
    raw: str
    normalized: str
    content: str
    normalized_content: str
    sender: str = ""


class RateLimiter:
    def __init__(self, per_minute: int, global_cooldown: float, user_cooldown: float):
        self.per_minute = per_minute
        self.global_cooldown = global_cooldown
        self.user_cooldown = user_cooldown
        self.recent: Deque[float] = deque()
        self.last_global = 0.0
        self.last_by_user: Dict[str, float] = defaultdict(float)

    def allow(self, user_key: str) -> bool:
        now = time.time()
        while self.recent and now - self.recent[0] > 60:
            self.recent.popleft()
        if len(self.recent) >= self.per_minute:
            return False
        if now - self.last_global < self.global_cooldown:
            return False
        if user_key and now - self.last_by_user[user_key] < self.user_cooldown:
            return False
        self.recent.append(now)
        self.last_global = now
        if user_key:
            self.last_by_user[user_key] = now
        return True


class BotEngine:
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = setup_logging(config.log_path, config.logging.log_level)
        self.model = LocalChatModel(config)
        self.stop_requested = False
        self.ignore_until = 0.0
        self.last_sent_parts: List[str] = []
        self.rate_limiter = RateLimiter(
            per_minute=config.chat.max_replies_per_minute,
            global_cooldown=config.chat.cooldown_seconds,
            user_cooldown=config.chat.user_cooldown_seconds,
        )
        self.trigger_re = re.compile(
            r"^(?:" + "|".join(re.escape(t) for t in config.chat.triggers) + r")(?:\b|\s|[:,.!?\-])",
            re.IGNORECASE,
        )
        self.remove_trigger_re = re.compile(
            r"^(?:" + "|".join(re.escape(t) for t in config.chat.triggers) + r")(?:\b|\s|[:,.!?\-])*",
            re.IGNORECASE,
        )
        self.hi_patterns = {normalize_for_match(x) for x in config.chat.hi_patterns}
        self.admin_names = {normalize_for_match(x) for x in config.chat.admin_names}

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "BotEngine":
        return cls(load_config(config_path))

    def warmup(self) -> None:
        self.model.load()

    def parse_chat(self, raw_msg: str) -> ParsedChat:
        raw = str(raw_msg).strip()
        # Common server formats. Sender extraction is intentionally best-effort;
        # command safety also works without it if admin_only_commands=false.
        msg = raw
        sender = ""

        # Remove Minecraft formatting then split arrows used in your baseline.
        normalized_raw = normalize_for_match(raw)
        if "➡" in msg:
            before, after = msg.split("➡", 1)
            sender = _guess_sender(before)
            msg = after
        elif ">" in msg and "<" in msg:
            m = re.search(r"<([^>]{1,32})>\s*(.*)", msg)
            if m:
                sender = m.group(1)
                msg = m.group(2)
        else:
            m = re.match(r"^\s*([A-Za-z0-9_]{2,32})\s*[:»>]\s*(.*)$", msg)
            if m:
                sender = m.group(1)
                msg = m.group(2)

        content = msg.strip()
        return ParsedChat(
            raw=raw,
            normalized=normalized_raw,
            content=content,
            normalized_content=normalize_for_match(content),
            sender=normalize_for_match(sender),
        )

    def is_own_message(self, normalized_content: str) -> bool:
        return normalized_content in self.last_sent_parts

    def split_reply(self, text: str) -> List[str]:
        max_len = self.config.chat.max_chat_chars
        parts: List[str] = []
        text = sanitize_chat_output(text)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            while len(line) > max_len:
                cut = line.rfind(" ", 0, max_len)
                if cut <= 0:
                    cut = max_len
                parts.append(line[:cut].strip())
                line = line[cut:].strip()
            if line:
                parts.append(line)
        return parts or ["eh?"]

    def remember_sent(self, parts: Iterable[str]) -> None:
        self.last_sent_parts = [normalize_for_match(p) for p in parts]
        self.ignore_until = time.time() + self.config.chat.ignore_after_send_seconds

    def command_reply(self, chat: ParsedChat) -> Optional[str]:
        msg = chat.normalized_content
        if not msg:
            return None

        needs_admin = any(
            token in msg
            for token in [
                "clear context",
                "shutdown",
                "shutd0wn",
                "reload retriever",
            ]
        )
        if needs_admin and self.config.chat.admin_only_commands:
            if not chat.sender or chat.sender not in self.admin_names:
                return "no permission"

        if msg in {"izu bot status", "izu print backend"}:
            return "backend local, model loaded" if self.model.model is not None else "backend local, model not loaded yet"
        if msg == "glaz3 d0ot":
            return "w doot"
        if msg in self.hi_patterns:
            return "nihao"
        if "izu clear context" in msg:
            clear_history()
            return "context cleared"
        if "izu bot reload retriever" in msg:
            return self.model.reload_retriever()
        if "izu bot help" in msg:
            return "ask: izu <question>. commands: izu bot status, izu clear context, izu bot reload retriever, izu shutdown"
        if "izu shutdown" in msg or "izu shutd0wn" in msg:
            self.stop_requested = True
            return "shutting down"
        return None

    def should_ignore(self, chat: ParsedChat) -> bool:
        if not chat.normalized_content:
            return True
        if time.time() < self.ignore_until:
            return True
        if self.is_own_message(chat.normalized_content):
            return True
        for blocked in self.config.chat.blocked_substrings:
            if normalize_for_match(blocked) in chat.normalized_content:
                return True
        return False

    def should_trigger(self, normalized_content: str) -> bool:
        return self.trigger_re.search(normalized_content) is not None

    def remove_trigger(self, content: str) -> str:
        return self.remove_trigger_re.sub("", content, count=1).strip()

    def handle_text(self, raw_msg: str) -> Optional[str]:
        chat = self.parse_chat(raw_msg)
        if self.should_ignore(chat):
            return None

        command = self.command_reply(chat)
        if command is not None:
            return command

        if not self.should_trigger(chat.normalized_content):
            return None
        if not self.rate_limiter.allow(chat.sender or "global"):
            return None

        query = self.remove_trigger(chat.content)
        query = query[: self.config.chat.max_input_chars].strip()
        if not query:
            return "hai?"

        try:
            return self.model.generate(query)
        except Exception as exc:
            self.logger.exception("generation failed")
            return f"model error: {str(exc)[:120]}"


def _guess_sender(prefix: str) -> str:
    # Try last username-like token in the server prefix.
    tokens = re.findall(r"[A-Za-z0-9_]{2,32}", prefix)
    return tokens[-1] if tokens else ""
