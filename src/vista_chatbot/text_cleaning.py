from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List

_MC_FORMAT_RE = re.compile(r"§.")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_MDX_IMPORT_EXPORT_RE = re.compile(r"^\s*(import|export)\s+.*$", re.MULTILINE)
_JSX_BLOCK_RE = re.compile(r"<([A-Z][A-Za-z0-9_.:-]*)(?:\s|>|/).*?(?:</\1>|/>)", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize_spaces(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _MC_FORMAT_RE.sub("", text)
    text = _ANSI_RE.sub("", text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 .,!?:;_@#/'\-]+", " ", text)
    return normalize_spaces(text)


def sanitize_chat_output(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _MC_FORMAT_RE.sub("", text)
    text = _ANSI_RE.sub("", text)
    text = _URL_RE.sub("", text)
    # Keep normal punctuation, alphanumerics, and readable symbols. Strip line
    # breaks because many servers treat them weirdly in chat packets.
    text = re.sub(r"[^\w .,!?:;/'\"()\[\]{}+\-=#@%&*<>|~\n]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_markdown_mdx(text: str, keep_code: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FRONTMATTER_RE.sub("", text)
    text = _MDX_IMPORT_EXPORT_RE.sub("", text)
    text = _JSX_BLOCK_RE.sub(" ", text)
    if not keep_code:
        text = _CODE_FENCE_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: m.group(1), text)
    text = _MD_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _INLINE_CODE_RE.sub(lambda m: m.group(1), text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = normalize_for_match(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
