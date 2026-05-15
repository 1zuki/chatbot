from __future__ import annotations

import json
import random
from json import JSONDecodeError
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

from .text_cleaning import normalize_spaces

SamplingMode = Literal["head", "reservoir"]


def _first_non_ws_char(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                return ""
            stripped = chunk.lstrip()
            if stripped:
                return stripped[0]


def _iter_top_level_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Stream records from a large top-level JSON array without loading it all.

    This matters for 700k-record QA dumps. It lets ``--qa-sampling head`` stop
    as soon as enough valid examples are found.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False
    started = False

    def read_more(f) -> None:
        nonlocal buffer, pos, eof
        chunk = f.read(chunk_size)
        if not chunk:
            eof = True
            return
        if pos > 0:
            buffer = buffer[pos:] + chunk
            pos = 0
        else:
            buffer += chunk

    with path.open("r", encoding="utf-8") as f:
        read_more(f)
        while True:
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1

                if not started:
                    if pos >= len(buffer):
                        if eof:
                            return
                        read_more(f)
                        continue
                    if buffer[pos] != "[":
                        raise ValueError(f"Expected top-level JSON array in {path}")
                    started = True
                    pos += 1
                    continue

                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos >= len(buffer):
                    if eof:
                        return
                    read_more(f)
                    continue
                if buffer[pos] == ",":
                    pos += 1
                    continue
                if buffer[pos] == "]":
                    return
                break

            try:
                row, end = decoder.raw_decode(buffer, pos)
            except JSONDecodeError:
                if eof:
                    raise
                read_more(f)
                continue

            pos = end
            if isinstance(row, dict):
                yield row


def iter_raw_json_records(path: str | Path) -> Iterator[dict]:
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return

    # Fast streaming path for huge dumps like [{question, answer, source}, ...].
    if p.suffix.lower() == ".json" and _first_non_ws_char(p) == "[":
        yield from _iter_top_level_json_array(p)
        return

    # Dict wrappers still use json.load so formats like {"data": [...]} keep working.
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
    elif isinstance(data, dict):
        # Accept either {"data": [...]} or {"items": [...]} style dumps.
        for key in ("data", "items", "examples", "records"):
            if isinstance(data.get(key), list):
                for row in data[key]:
                    if isinstance(row, dict):
                        yield row
                return
        # Single record fallback.
        yield data


def normalize_qa(row: dict) -> Optional[dict]:
    q = normalize_spaces(str(row.get("question", "")))
    a = normalize_spaces(str(row.get("answer", "")))
    source = normalize_spaces(str(row.get("source", "minecraft_qa")))
    if len(q) < 3 or len(a) < 2:
        return None
    return {"question": q, "answer": a, "source": source}


def _iter_normalized_qa(paths: Iterable[str | Path]) -> Iterator[dict]:
    for path in paths:
        for row in iter_raw_json_records(path):
            item = normalize_qa(row)
            if item is not None:
                yield item


def build_qa_records(
    paths: Iterable[str | Path],
    max_records: Optional[int] = None,
    sampling: SamplingMode = "head",
    seed: int = 42,
) -> Iterator[dict]:
    """Yield normalized QA records.

    ``head`` is intentionally the default because it is fastest on a laptop: it stops
    reading once ``max_records`` valid examples are found. ``reservoir`` gives a more
    representative random sample, but it must scan the full QA input.
    """
    if max_records is not None and max_records <= 0:
        return

    if sampling == "head" or max_records is None:
        count = 0
        for item in _iter_normalized_qa(paths):
            yield item
            count += 1
            if max_records is not None and count >= max_records:
                return
        return

    if sampling != "reservoir":
        raise ValueError(f"Unsupported QA sampling mode: {sampling}")

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    for item in _iter_normalized_qa(paths):
        seen += 1
        if len(reservoir) < max_records:
            reservoir.append(item)
            continue
        j = rng.randrange(seen)
        if j < max_records:
            reservoir[j] = item

    rng.shuffle(reservoir)
    yield from reservoir


def qa_to_retriever_doc(item: dict, idx: int) -> dict:
    question = item["question"]
    answer = item["answer"]
    source = item.get("source", "minecraft_qa")
    return {
        "id": f"qa:{idx}",
        "kind": "qa",
        "title": question[:120],
        "source": source,
        "path": source,
        "section": "",
        "text": f"Question: {question}\nAnswer: {answer}",
    }


def qa_to_sft_row(item: dict) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Vista, a concise Minecraft server assistant. Answer accurately using server wiki and Minecraft knowledge. If unsure, say you do not know.",
            },
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ],
        "source": item.get("source", "minecraft_qa"),
    }
