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
            table_context = heading_path[-1] if heading_path else title
            for start, end, piece in chunk_section(
                section_text, chunk_chars, chunk_overlap, table_context=table_context
            ):
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
    # MDX imports/exports that never help retrieval.
    text = re.sub(r"^\s*import\s+.+?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*export\s+.+?$", "", text, flags=re.MULTILINE)

    # Protect code spans BEFORE stripping JSX/emphasis. Wiki commands like
    # `/warp <warp-name>` live in backticks; the generic <...> strip below would
    # otherwise delete the `<warp-name>` argument and leave a bare `/warp`.
    protected: list[str] = []

    def _protect(inner: str) -> str:
        token = f"\x00C{len(protected)}\x00"
        protected.append(inner)
        return token

    # Fenced code: keep contents, drop the fence markers.
    text = re.sub(
        r"```[a-zA-Z0-9_-]*\n(.*?)```",
        lambda m: _protect(m.group(1).rstrip("\n")),
        text,
        flags=re.DOTALL,
    )
    text = text.replace("```", "")
    # Inline code: keep contents verbatim.
    text = re.sub(r"`([^`\n]+)`", lambda m: _protect(m.group(1)), text)

    # JSX blocks that usually do not help retrieval.
    text = re.sub(r"<Tabs>[\s\S]*?</Tabs>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<TabItem[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</TabItem>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown links/images -> keep readable label.
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Emphasis markers.
    text = re.sub(r"[*_]{1,3}", "", text)
    # Starlight/Docusaurus admonitions: keep the optional title, drop the marker.
    text = re.sub(
        r"^:::\w+(?:\[([^\]]*)\])?\s*$",
        lambda m: (m.group(1) or "").strip(),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^:::$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Restore protected code spans now that destructive passes are done.
    for i, inner in enumerate(protected):
        text = text.replace(f"\x00C{i}\x00", inner)
    return text.strip()


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
    return fallback


def split_sections(text: str) -> Iterator[tuple[list[str], str]]:
    lines = text.splitlines()
    # Stack of (level, title) so sibling headings replace rather than nest.
    # Without tracking levels, three sibling `### /town` style headings would
    # accumulate into a bogus `/towny > /plot > /town` path and pollute the
    # embedding text for command tables.
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def heading_path() -> list[str]:
        return [title for _, title in stack]

    def flush() -> Iterator[tuple[list[str], str]]:
        section = "\n".join(current_lines).strip()
        if section:
            yield heading_path(), section

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if m:
            yield from flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_lines[:] = [title]
        else:
            current_lines.append(line)
    yield from flush()


def chunk_section(
    text: str, chunk_chars: int, overlap: int, *, table_context: str = ""
) -> Iterator[tuple[int, int, str]]:
    """Split one heading section into retrieval chunks, respecting structure.

    Command-reference tables (the bulk of this wiki) are kept row-intact and
    batched so a single row like `/town set taxes {$}` is never cut in half;
    each table chunk is prefixed with ``table_context`` (the section heading or
    page title) so every row carries its topic word — e.g. a Stars perk row gets
    "Stars" prepended so a query about "star rank" overlaps it. Prose is packed
    paragraph-by-paragraph instead of by raw character windows.
    """
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return
    context = table_context.strip()
    for kind, block_text, base in _segment_blocks(normalized):
        if kind == "table":
            yield from _chunk_table(block_text, base, context, chunk_chars)
        else:
            yield from _chunk_prose(block_text, base, chunk_chars, overlap)


def _is_table_line(stripped: str) -> bool:
    return stripped.startswith("|")


def _is_separator_row(stripped: str) -> bool:
    return "-" in stripped and re.fullmatch(r"\|?[\s:|-]+\|?", stripped) is not None


def _segment_blocks(normalized: str) -> Iterator[tuple[str, str, int]]:
    """Yield (kind, block_text, start_char) where kind is 'table' or 'prose'.

    Consecutive table rows form one table block; everything else forms prose
    blocks. Blank lines stay with the surrounding block rather than splitting it.
    """
    cur_kind: str | None = None
    cur_lines: list[str] = []
    cur_start = 0
    pos = 0
    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            kind = cur_kind or "prose"
        else:
            kind = "table" if _is_table_line(stripped) else "prose"
        if cur_kind is None:
            cur_kind, cur_start, cur_lines = kind, pos, [line]
        elif kind == cur_kind:
            cur_lines.append(line)
        else:
            yield cur_kind, "\n".join(cur_lines).strip("\n"), cur_start
            cur_kind, cur_start, cur_lines = kind, pos, [line]
        pos += len(line) + 1
    if cur_lines and cur_kind is not None:
        yield cur_kind, "\n".join(cur_lines).strip("\n"), cur_start


def _chunk_table(
    block_text: str, base: int, context: str, chunk_chars: int
) -> Iterator[tuple[int, int, str]]:
    # A markdown header row is the one immediately followed by a |---| separator.
    # Headers ("Rank | Requirements | Perks", "Command | Description") are column
    # labels, never answers, so dropping them keeps them out of extraction.
    block_lines = block_text.split("\n")
    header_idx = -1
    for i, line in enumerate(block_lines):
        if _is_separator_row(line.strip()) and i > 0:
            header_idx = i - 1
            break

    rows: list[tuple[int, str]] = []
    pos = 0
    for i, line in enumerate(block_lines):
        stripped = line.strip()
        if (
            stripped
            and _is_table_line(stripped)
            and not _is_separator_row(stripped)
            and i != header_idx
        ):
            rows.append((base + pos, line.rstrip()))
        pos += len(line) + 1
    if not rows:
        return
    ctx = context.strip()
    prefix = f"{ctx}\n" if ctx and not _is_table_line(ctx) else ""
    batch: list[str] = []
    batch_start = rows[0][0]
    cur = len(prefix)
    for off, row in rows:
        row_len = len(row) + 1
        if batch and cur + row_len > chunk_chars:
            piece = prefix + "\n".join(batch)
            yield batch_start, batch_start + len(piece), piece
            batch, cur = [], len(prefix)
        if not batch:
            batch_start = off
        batch.append(row)
        cur += row_len
    if batch:
        piece = prefix + "\n".join(batch)
        yield batch_start, batch_start + len(piece), piece


def _split_paragraphs(block_text: str, base: int) -> list[tuple[int, str]]:
    paras: list[tuple[int, str]] = []
    idx = 0
    for raw in block_text.split("\n\n"):
        p = raw.strip()
        if p:
            paras.append((base + idx, p))
        idx += len(raw) + 2
    return paras


def _chunk_prose(
    block_text: str, base: int, chunk_chars: int, overlap: int
) -> Iterator[tuple[int, int, str]]:
    paras = _split_paragraphs(block_text, base)
    if not paras:
        return
    buf: list[tuple[int, str]] = []
    cur = 0
    for off, p in paras:
        if len(p) > chunk_chars:
            if buf:
                yield _emit_paras(buf)
                buf, cur = [], 0
            for s, e, sub in sliding_chunks(p, chunk_chars, overlap):
                yield off + s, off + e, sub
            continue
        if buf and cur + len(p) + 2 > chunk_chars:
            yield _emit_paras(buf)
            if overlap > 0 and len(buf[-1][1]) <= max(overlap, 1) * 2:
                buf = [buf[-1]]
                cur = len(buf[-1][1]) + 2
            else:
                buf, cur = [], 0
        buf.append((off, p))
        cur += len(p) + 2
    if buf:
        yield _emit_paras(buf)


def _emit_paras(buf: list[tuple[int, str]]) -> tuple[int, int, str]:
    start = buf[0][0]
    piece = "\n\n".join(p for _, p in buf)
    return start, start + len(piece), piece


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
