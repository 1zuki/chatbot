from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .config import load_config
from .runtime import BotEngine


@dataclass
class EvalResult:
    question: str
    expected: str
    actual: str
    source: str


def evaluate_jsonl(config_path: str | Path, qa_path: str | Path, limit: int = 50, out_path: Optional[str | Path] = None) -> List[EvalResult]:
    engine = BotEngine(load_config(config_path))
    engine.warmup()
    results: List[EvalResult] = []
    with Path(qa_path).open("r", encoding="utf-8") as f:
        for line in f:
            if len(results) >= limit:
                break
            row = json.loads(line)
            question = row.get("question") or (row.get("messages", [{}, {"content": ""}])[1].get("content", ""))
            expected = row.get("answer") or (row.get("messages", [{}, {}, {"content": ""}])[-1].get("content", ""))
            actual = engine.model.generate(str(question))
            results.append(EvalResult(str(question), str(expected), actual, str(row.get("source", ""))))

    if out_path:
        out = Path(out_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    return results
