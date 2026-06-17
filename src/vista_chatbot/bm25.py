from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from .text import compact_for_match

_SPLIT_RE = re.compile(r"[\s-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text the same way for documents and queries.

    Reuses ``compact_for_match`` (lowercase, strip punctuation) so a command
    like ``/town set taxes`` indexes as ``town set taxes`` and matches a query
    phrased as "town set taxes". Hyphens split too, so ``warp-name`` matches a
    query that says "warp name".
    """
    compact = compact_for_match(text)
    if not compact:
        return []
    return [t for t in _SPLIT_RE.split(compact) if t]


class BM25Index:
    """Pure-stdlib Okapi BM25 over a fixed document corpus.

    Built in-memory from the same chunk list the dense index holds, so it adds
    no new files on disk and no heavy dependencies. For a server wiki (a few
    hundred chunks) scoring every document per query is well under a millisecond.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.n = 0

    @classmethod
    def build(cls, tokenized_docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        self = cls(k1=k1, b=b)
        self.n = len(tokenized_docs)
        df: Counter[str] = Counter()
        total_len = 0
        for toks in tokenized_docs:
            counts = Counter(toks)
            self.doc_freqs.append(counts)
            self.doc_len.append(len(toks))
            total_len += len(toks)
            for term in counts:
                df[term] += 1
        self.avgdl = (total_len / self.n) if self.n else 0.0
        self.df = dict(df)
        for term, freq in df.items():
            # log(1 + ...) form keeps IDF non-negative, so a term appearing in
            # more than half the corpus can't push a score below zero.
            self.idf[term] = math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
        return self

    def scores(self, query_terms: Iterable[str]) -> list[float]:
        terms = [t for t in query_terms if t in self.idf]
        out = [0.0] * self.n
        if not terms or self.avgdl <= 0:
            return out
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i]
            if not dl:
                continue
            norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            s = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                s += self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)
            out[i] = s
        return out
