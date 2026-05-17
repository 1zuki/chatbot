from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Rule, RulesConfig
from .text import compact_for_match, normalize_text


@dataclass(frozen=True)
class RuleResult:
    matched: bool
    reply: str | None
    stop: bool
    rule_name: str | None = None


class RuleEngine:
    def __init__(self, config: RulesConfig):
        self.config = config
        self._compiled: dict[str, list[re.Pattern[str]]] = {}
        for rule in config.special_cases:
            if rule.kind == "regex":
                flags = 0 if rule.case_sensitive else re.IGNORECASE
                self._compiled[rule.name] = [re.compile(p, flags) for p in rule.patterns]

    def match(self, text: str) -> RuleResult:
        for rule in self.config.special_cases:
            if self._matches(rule, text):
                return RuleResult(True, rule.reply, rule.stop, rule.name)
        return RuleResult(False, None, False, None)

    def _matches(self, rule: Rule, text: str) -> bool:
        if rule.kind == "contains":
            hay = text if rule.case_sensitive else normalize_text(text)
            for pattern in rule.patterns:
                needle = pattern if rule.case_sensitive else normalize_text(pattern)
                if needle in hay:
                    return True
            return False
        if rule.kind == "exact_normalized":
            hay = compact_for_match(text)
            return any(hay == compact_for_match(p) for p in rule.patterns)
        if rule.kind == "regex":
            return any(p.search(text) for p in self._compiled.get(rule.name, []))
        raise ValueError(f"Unknown rule kind {rule.kind!r} for rule {rule.name!r}")
