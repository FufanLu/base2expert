"""Why was a document dropped as contaminated? Print its matching n-gram spans.

For each removed document, slides the 13-unit window over its units and
prints the spans found in the frozen eval index, merged into maximal runs,
so one can tell a copied test passage (a long run) from shared boilerplate
(many short, common phrases). Also reports the run-length distribution.

    python -P tools/decontam_inspect.py --index eval_13gram.npy --n 13 --show 15 removed/*.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from b2e.data.decontam import _key, units


def runs(u: list[str], keys: np.ndarray, n: int) -> list[tuple[int, int]]:
    hashes = np.array([int.from_bytes(_key(u[i:i + n]), "little") for i in range(len(u) - n + 1)], dtype=np.uint64)
    if len(hashes) == 0:
        return []
    pos = np.searchsorted(keys, hashes)
    pos[pos == len(keys)] = len(keys) - 1
    hit = np.asarray(keys[pos]) == hashes
    out, start = [], None
    for i, h in enumerate(hit):
        if h and start is None:
            start = i
        if not h and start is not None:
            out.append((start, i - 1 + n))
            start = None
    if start is not None:
        out.append((start, len(hit) - 1 + n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    keys = np.load(a.index, mmap_mode="r")
    lengths: Counter = Counter()
    longest = []
    shown = 0
    for f in a.files:
        for text in pq.read_table(f, columns=["text"]).column("text").to_pylist():
            u = units(text)
            r = runs(u, keys, a.n)
            for s, e in r:
                lengths[min(e - s, 60)] += 1
            longest.append(max((e - s for s, e in r), default=0))
            if shown < a.show:
                spans = ["".join(u[s:e]) if all(len(x) == 1 for x in u[s:e]) else " ".join(u[s:e]) for s, e in r[:4]]
                print(f"--- doc {shown}: {len(r)} matching runs, longest {max((e - s for s, e in r), default=0)} units")
                for sp in spans:
                    print("   ", sp[:160])
                shown += 1
    print("\nrun length (units) -> count:", sorted(lengths.items()))
    ls = np.array(longest)
    for t in (13, 20, 30, 50):
        print(f"docs whose longest matching run is >= {t} units: {(ls >= t).sum()} of {len(ls)}")


if __name__ == "__main__":
    main()
