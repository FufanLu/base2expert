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
    def build(texts: Iterable[str], path: str, n: int = 13, max_df: int = 0) -> int:
        """max_df > 0 drops n-grams found in max_df or more distinct eval items.

        Those are shared phrasing, not test content: in Chinese a 13-character
        window is a set phrase, and slogans or legal boilerplate recur across
        C-Eval / CMMLU politics questions and CAIL rulings. A training document
        that contains them has copied no test item (docs/runs.md, 11347805)."""
        parts = [ngram_hashes(t, n) for t in texts]  # unique within each item
        if not parts:
            keys = np.zeros(0, dtype=np.uint64)
        elif max_df:
            allk, counts = np.unique(np.concatenate(parts), return_counts=True)
            keys = allk[counts < max_df]
        else:
            keys = np.unique(np.concatenate(parts))
        np.save(path, keys)
        return len(keys)

    def match_runs(self, text: str) -> list[tuple[int, float]]:
        """Maximal runs of consecutive matching n-grams, as (length in units,
        fraction of the run's units that are CJK characters)."""
        u = units(text)
        if len(u) < self.n or len(self.keys) == 0:
            return []
        h = np.fromiter((int.from_bytes(_key(u[i:i + self.n]), "little") for i in range(len(u) - self.n + 1)),
                        dtype=np.uint64)
        pos = np.searchsorted(self.keys, h)
        pos[pos == len(self.keys)] = len(self.keys) - 1
        hit = np.asarray(self.keys[pos]) == h
        out, start = [], None
        for i, flag in enumerate(list(hit) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                span = u[start:i - 1 + self.n]
                cjk = sum(1 for x in span if len(x) == 1 and "\u3400" <= x <= "\ufaff") / len(span)
                out.append((len(span), cjk))
                start = None
        return out

    def contaminated_by_runs(self, text: str, min_units_en: int = 13, min_units_cjk: int = 20) -> bool:
        """A run counts as copied test content when it is long enough for its script:
        13 English words (GPT-3's rule) or 20 consecutive CJK characters."""
        return any(length >= (min_units_cjk if cjk > 0.5 else min_units_en) for length, cjk in self.match_runs(text))

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


class ItemNgramIndex:
    """Which eval item does a training document copy, and how much of it?

    Following the Qwen2 report's decontamination rule (an n-gram match plus a
    longest-common-subsequence of at least 60% of the shorter text), a
    training document is contaminated only when it covers at least
    `min_coverage` of some eval item's n-grams. Quoting a statute or a slogan
    that a test question also quotes covers a small part of that item; a
    copied question (with its options) covers most of it.

    Stored as three arrays: sorted n-gram hashes, the eval item id of each
    (an n-gram in k items appears k times), and each item's n-gram count.
    """

    def __init__(self, prefix: str, n: int = 13, max_items_per_ngram: int = 50) -> None:
        self.n = n
        self.keys = np.load(prefix + ".keys.npy", mmap_mode="r")
        self.items = np.load(prefix + ".items.npy", mmap_mode="r")
        self.item_ngrams = np.load(prefix + ".item_ngrams.npy", mmap_mode="r")
        self.max_items = max_items_per_ngram

    @staticmethod
    def build(texts: Iterable[str], prefix: str, n: int = 13) -> tuple[int, int]:
        keys, items, counts = [], [], []
        for i, t in enumerate(texts):
            h = ngram_hashes(t, n)
            keys.append(h)
            items.append(np.full(len(h), i, dtype=np.uint32))
            counts.append(len(h))
        k = np.concatenate(keys) if keys else np.zeros(0, dtype=np.uint64)
        it = np.concatenate(items) if items else np.zeros(0, dtype=np.uint32)
        order = np.argsort(k, kind="stable")
        np.save(prefix + ".keys.npy", k[order])
        np.save(prefix + ".items.npy", it[order])
        np.save(prefix + ".item_ngrams.npy", np.asarray(counts, dtype=np.uint32))
        return len(k), len(counts)

    def coverage(self, text: str) -> tuple[float, int]:
        """(highest fraction of one eval item's n-grams found in text, that item's id)."""
        h = ngram_hashes(text, self.n)
        if len(h) == 0 or len(self.keys) == 0:
            return 0.0, -1
        lo = np.searchsorted(self.keys, h, side="left")
        hi = np.searchsorted(self.keys, h, side="right")
        width = hi - lo
        # an n-gram shared by very many items is phrasing, not content
        sel = (width > 0) & (width <= self.max_items)
        if not sel.any():
            return 0.0, -1
        ids = np.concatenate([np.asarray(self.items[a:b]) for a, b in zip(lo[sel], hi[sel], strict=True)])
        uniq, cnt = np.unique(ids, return_counts=True)
        cov = cnt / np.asarray(self.item_ngrams[uniq]).astype(np.float64)
        j = int(cov.argmax())
        return float(cov[j]), int(uniq[j])

    def contaminated(self, text: str, min_coverage: float = 0.6) -> bool:
        return self.coverage(text)[0] >= min_coverage
