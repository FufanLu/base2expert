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

import numpy as np

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


def ngram_hashes(text: str, n: int) -> np.ndarray:
    """The text's n-gram keys as a sorted unique uint64 array."""
    keys = ngram_keys(text, n)
    return np.unique(np.fromiter((int.from_bytes(k, "little") for k in keys), dtype=np.uint64, count=len(keys)))


class FrozenNgramIndex:
    """Read-only index over many eval texts: a sorted uint64 array on disk.

    A Python set costs ~100 bytes per n-gram; eval sets with hundreds of
    thousands of court rulings give ~10^8 n-grams, too much per worker. As a
    sorted array it is 8 bytes each, loaded with mmap so all workers on a
    node share one copy in the page cache, and membership is a binary search.
    """

    def __init__(self, path: str, n: int = 13) -> None:
        self.n = n
        self.keys = np.load(path, mmap_mode="r")

    @staticmethod
    def build(texts: Iterable[str], path: str, n: int = 13) -> int:
        parts = [ngram_hashes(t, n) for t in texts]
        keys = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)
        np.save(path, keys)
        return len(keys)

    def hits(self, text: str) -> int:
        q = ngram_hashes(text, self.n)
        if len(q) == 0 or len(self.keys) == 0:
            return 0
        pos = np.searchsorted(self.keys, q)
        pos[pos == len(self.keys)] = len(self.keys) - 1
        return int((np.asarray(self.keys[pos]) == q).sum())

    def contaminated(self, text: str, min_hits: int = 1) -> bool:
        return self.hits(text) >= min_hits

    def __len__(self) -> int:
        return len(self.keys)
