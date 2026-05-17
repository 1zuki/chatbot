from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    title: str
    heading_path: list[str]
    text: str
    start_char: int
    end_char: int


def discover_wiki_files(wiki_dir: Path, globs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for glob in globs:
        files.extend(p for p in wiki_dir.glob(glob) if p.is_file())
    return sorted(set(files))


def load_wiki_chunks(
    wiki_dir: Path,
    *,
    globs: Iterable[str],
    chunk_chars: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in discover_wiki_files(wiki_dir, globs):
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_mdx(raw)
        rel = path.relative_to(wiki_dir).as_posix()
        title = infer_title(cleaned, fallback=path.stem.replace("-", " ").replace("_", " ").title())
        sections = list(split_sections(cleaned)) or [([], cleaned)]
        for heading_path, section_text in sections:
            for start, end, piece in sliding_chunks(section_text, chunk_chars, chunk_overlap):
                text = piece.strip()
                if len(text) < 40:
                    continue
                chunk_id = stable_chunk_id(rel, heading_path, start, text)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_path=rel,
                        title=title,
                        heading_path=heading_path,
                        text=text,
                        start_char=start,
                        end_char=end,
                    )
                )
    return chunks


def clean_mdx(text: str) -> str:
    text = text.replace("\r\n", "\n")
    # YAML frontmatter.
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # MDX imports/exports and JSX blocks that usually do not help retrieval.
    text = re.sub(r"^\s*import\s+.+?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*export\s+.+?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<Tabs>[\s\S]*?</Tabs>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<TabItem[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</TabItem>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown links/images -> keep readable label.
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Code fences: keep contents, remove fence markers.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n", "", text)
    text = text.replace("```", "")
    # Inline code markers and emphasis.
    text = text.replace("`", "")
    text = re.sub(r"[*_]{1,3}", "", text)
    # Docusaurus admonitions.
    text = re.sub(r"^:::\w+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^:::$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
    return fallback


def split_sections(text: str) -> Iterator[tuple[list[str], str]]:
    lines = text.splitlines()
    current_heading: list[str] = []
    current_lines: list[str] = []

    def flush() -> Iterator[tuple[list[str], str]]:
        section = "\n".join(current_lines).strip()
        if section:
            yield current_heading.copy(), section

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if m:
            yield from flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading[:] = current_heading[: level - 1] + [title]
            current_lines[:] = [title]
        else:
            current_lines.append(line)
    yield from flush()


def sliding_chunks(text: str, chunk_chars: int, overlap: int) -> Iterator[tuple[int, int, str]]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(normalized) <= chunk_chars:
        yield 0, len(normalized), normalized
        return
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_chars)
        if end < len(normalized):
            window = normalized[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"), window.rfind(" "))
            if split_at > chunk_chars * 0.55:
                end = start + split_at
        piece = normalized[start:end].strip()
        if piece:
            yield start, end, piece
        if end >= len(normalized):
            break
        start = max(0, end - overlap)


def stable_chunk_id(source_path: str, heading_path: list[str], start: int, text: str) -> str:
    blob = json.dumps([source_path, heading_path, start, text[:160]], ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            chunks.append(Chunk(**item))
    return chunks
