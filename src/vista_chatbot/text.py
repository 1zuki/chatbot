from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


_FORMAT_CODE_RE = re.compile(r"(?:§.|&[0-9A-FK-ORa-fk-or])")


def strip_minecraft_formatting(text: str) -> str:
    """Remove common Minecraft color/style codes from rendered chat."""
    return _FORMAT_CODE_RE.sub("", text)


def normalize_text(text: str) -> str:
    text = strip_minecraft_formatting(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compact_for_match(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9_\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars - 1)].rstrip()
    last = max(cut.rfind(" "), cut.rfind("."), cut.rfind(","), cut.rfind(";"))
    if last >= max_chars * 0.55:
        cut = cut[:last].rstrip()
    return cut + "…"


def sanitize_chat_output(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Avoid accidental pings or generated multi-line junk.
    text = text.replace("@everyone", "everyone").replace("@here", "here")
    return text


def split_for_chat(text: str, max_chars: int) -> list[str]:
    text = sanitize_chat_output(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining.strip())
            break
        cut = remaining[:max_chars]
        split_at = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "), cut.rfind(" "))
        if split_at < max_chars * 0.45:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return [p for p in parts if p]


def word_pattern(word: str) -> re.Pattern[str]:
    key = normalize_text(word)
    if key not in _WORD_RE_CACHE:
        _WORD_RE_CACHE[key] = re.compile(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])", re.IGNORECASE)
    return _WORD_RE_CACHE[key]


@dataclass(frozen=True)
class ParsedChat:
    raw: str
    speaker: str | None
    rank: str | None
    content: str


def _extract_identity_from_decorated_left(left: str) -> tuple[str | None, str | None]:
    # Remove trailing flair tags like `[❄ '24]` repeatedly.
    cleaned = left.strip()
    while True:
        nxt = re.sub(r"\s*\[[^\]\n]{1,80}\]\s*$", "", cleaned).strip()
        if nxt == cleaned:
            break
        cleaned = nxt
    tokens = re.findall(r"[A-Za-z0-9_]{2,32}", cleaned)
    if not tokens:
        return None, None
    if len(tokens) == 1:
        return None, tokens[-1]
    # In rank+name formats, the last token is usually player name and the one
    # before it is rank.
    return tokens[-2], tokens[-1]


def _split_decorated_chat(text: str) -> tuple[str | None, str | None, str] | None:
    """Parse lines like `🏕 ➟ RANK Player [tag] ➡ !vista question`."""
    spaced = [" ➡ ", " » ", " › ", " -> ", " : "]
    for sep in spaced:
        if sep not in text:
            continue
        left, right = text.rsplit(sep, 1)
        msg = right.strip()
        if not msg:
            continue
        rank, speaker = _extract_identity_from_decorated_left(left)
        return rank, speaker, msg

    glyphs = ["➡", "»", "›"]
    positions = [(text.rfind(g), g) for g in glyphs]
    pos, glyph = max(positions, key=lambda x: x[0])
    if pos == -1:
        return None
    left = text[:pos]
    msg = text[pos + len(glyph) :].strip()
    if not msg:
        return None
    rank, speaker = _extract_identity_from_decorated_left(left)
    return rank, speaker, msg


def parse_minecraft_chat(raw: str) -> ParsedChat:
    """Best-effort parser for common Minecraft chat formats.

    Server formats differ a lot. This parser first tries normal vanilla-ish
    formats, then falls back to taking the text after the final decorated chat
    separator, which is what rank/world prefixes usually use.
    """
    text = strip_minecraft_formatting(raw).strip()
    patterns = [
        r"^<(?P<name>[^>]{1,32})>\s*(?P<msg>.+)$",
        r"^\[(?:[^\]]+)\]\s*(?P<name>[A-Za-z0-9_]{2,32})\s*[»:>➡]\s*(?P<msg>.+)$",
        r"^(?P<name>[A-Za-z0-9_]{2,32})\s*[:>»➡]\s*(?P<msg>.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, text)
        if m:
            return ParsedChat(raw=raw, speaker=m.group("name"), rank=None, content=m.group("msg").strip())

    decorated = _split_decorated_chat(text)
    if decorated:
        rank, speaker, content = decorated
        return ParsedChat(raw=raw, speaker=speaker, rank=rank, content=content)
    return ParsedChat(raw=raw, speaker=None, rank=None, content=text)


def contains_trigger(text: str, triggers: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(word_pattern(t).search(normalized) for t in triggers)


def strip_trigger(text: str, triggers: Iterable[str]) -> str:
    out = text
    for trigger in sorted(triggers, key=len, reverse=True):
        out = word_pattern(trigger).sub(" ", out, count=1)
    out = re.sub(r"^[,;:!?.\s-]+", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def is_hi_pattern(text: str, patterns: Iterable[str]) -> bool:
    normalized = compact_for_match(text)
    return any(normalized == compact_for_match(p) for p in patterns)
