"""N-gram decontamination: drop training text that overlaps an eval set.

A benchmark question that sits in the training data turns the eval score
into a memory test. The check follows GPT-3's 13-gram rule, adapted to mixed
Chinese/English text: the text is split into units, an English word or number
being one unit and every CJK character another (Chinese has no spaces, and a
13-character overlap is already a copied sentence). Units are lower-cased and
punctuation is dropped, so formatting changes do not hide a copy.

    idx = NgramIndex(n=13)
    for q in eval_questions: idx.add(q)
    idx.hits(train_doc)   # number of the doc's n-grams that occur in any eval text
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_UNIT_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*|[㐀-䶿一-鿿豈-﫿]")


def units(text: str) -> list[str]:
    return _UNIT_RE.findall(text.lower())


def _key(gram: Iterable[str]) -> bytes:
    return hashlib.blake2b("\x1f".join(gram).encode(), digest_size=8).digest()


def ngram_keys(text: str, n: int) -> set[bytes]:
    u = units(text)
    return {_key(u[i:i + n]) for i in range(len(u) - n + 1)}


class NgramIndex:
    def __init__(self, n: int = 13) -> None:
        self.n = n
        self.keys: set[bytes] = set()

    def add(self, text: str) -> None:
        self.keys |= ngram_keys(text, self.n)

    def hits(self, text: str) -> int:
        return sum(1 for k in ngram_keys(text, self.n) if k in self.keys)

    def contaminated(self, text: str, min_hits: int = 1) -> bool:
        return self.hits(text) >= min_hits

    def __len__(self) -> int:
        return len(self.keys)
