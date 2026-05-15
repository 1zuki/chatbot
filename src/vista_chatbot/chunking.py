from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .text_cleaning import normalize_spaces, strip_markdown_mdx

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    id: str
    text: str
    source: str
    kind: str
    title: str = ""
    path: str = ""
    section: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    # Cheap approximation good enough for chunk boundaries without pulling a tokenizer.
    return max(1, int(len(text) / 4))


def read_markdown_files(wiki_dir: Path) -> Iterator[Path]:
    for suffix in ("*.md", "*.mdx"):
        yield from sorted(wiki_dir.rglob(suffix))


def _split_by_headings(raw_text: str) -> List[Tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(raw_text))
    if not matches:
        return [("", raw_text)]

    sections: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        prelude = raw_text[: matches[0].start()].strip()
        if prelude:
            sections.append(("", prelude))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _window_sentences(text: str, max_tokens: int, overlap_tokens: int) -> Iterator[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip())

    current: List[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunk = "\n\n".join(current).strip()
            if chunk:
                yield chunk

            if overlap_tokens > 0:
                overlap: List[str] = []
                overlap_total = 0
                for old in reversed(current):
                    t = estimate_tokens(old)
                    if overlap_total + t > overlap_tokens:
                        break
                    overlap.append(old)
                    overlap_total += t
                current = list(reversed(overlap))
                current_tokens = overlap_total
            else:
                current = []
                current_tokens = 0

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunk = "\n\n".join(current).strip()
        if chunk:
            yield chunk


def chunk_markdown_file(
    file_path: Path,
    wiki_root: Path,
    max_tokens: int = 380,
    overlap_tokens: int = 70,
    keep_code: bool = False,
) -> Iterator[TextChunk]:
    rel = file_path.relative_to(wiki_root).as_posix()
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    title = file_path.stem.replace("_", " ").replace("-", " ").strip().title()

    for section_idx, (section, body) in enumerate(_split_by_headings(raw)):
        cleaned = strip_markdown_mdx(body, keep_code=keep_code)
        cleaned = cleaned.strip()
        if not cleaned:
            continue
        for chunk_idx, chunk_text in enumerate(_window_sentences(cleaned, max_tokens, overlap_tokens)):
            text = normalize_spaces(chunk_text)
            if len(text) < 40:
                continue
            chunk_id = f"wiki:{rel}:{section_idx}:{chunk_idx}"
            yield TextChunk(
                id=chunk_id,
                text=text,
                source=f"wiki:{rel}",
                kind="wiki",
                title=title,
                path=rel,
                section=section,
            )


def chunk_wiki_dir(
    wiki_dir: str | Path,
    out_jsonl: str | Path,
    max_tokens: int = 380,
    overlap_tokens: int = 70,
    keep_code: bool = False,
) -> int:
    root = Path(wiki_dir).expanduser().resolve()
    out = Path(out_jsonl).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for path in read_markdown_files(root):
            for chunk in chunk_markdown_file(path, root, max_tokens, overlap_tokens, keep_code):
                f.write(chunk.to_json() + "\n")
                count += 1
    return count


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
