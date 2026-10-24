"""Were the documents MinHash removed really near-duplicates of a kept one?

datatrove keeps one document per duplicate cluster and drops the rest
(removed/minhash). For a sample of dropped documents this finds the most
similar kept document (datasketch LSH over the kept set, same 5-unit
shingles as the cleaning run), reports the exact Jaccard similarity of their
shingle sets, and prints the start of a few pairs side by side. A dropped
document without any kept partner above the threshold would be a false
positive of the bucketing.

    python -P tools/dedup_inspect.py --kept data/clean/fin_en --removed data/clean_work/fin_en/removed/minhash \\
        --sample 300 --show 5
"""

from __future__ import annotations

import argparse
import glob
import random
from collections import Counter

import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

from b2e.data.decontam import units


def shingles(text: str, n: int = 5) -> set[bytes]:
    u = units(text)
    return {"\x1f".join(u[i:i + n]).encode() for i in range(len(u) - n + 1)}


def texts(path: str) -> list[str]:
    out: list[str] = []
    for f in sorted(glob.glob(f"{path}/**/*.parquet", recursive=True)):
        out.extend(pq.read_table(f, columns=["text"]).column("text").to_pylist())
    return out


def signature(sh: set[bytes], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    m.update_batch(list(sh))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", required=True)
    ap.add_argument("--removed", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5, help="LSH candidate threshold (below datatrove's ~0.72)")
    ap.add_argument("--num-perm", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    removed = texts(args.removed)
    random.seed(args.seed)
    sample = random.sample(removed, min(args.sample, len(removed)))
    kept = texts(args.kept)
    print(f"kept {len(kept)}, removed {len(removed)}, sampled {len(sample)}", flush=True)

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    for i, t in enumerate(kept):  # shingle sets are not kept: ~80 M of them would not fit in memory
        sh = shingles(t)
        if sh:
            lsh.insert(i, signature(sh, args.num_perm))
    best = []
    for t in sample:
        sh = shingles(t)
        cands = lsh.query(signature(sh, args.num_perm)) if sh else []
        scored = []
        for c in cands:
            other = shingles(kept[c])
            scored.append((len(sh & other) / len(sh | other), c))
        best.append((max(scored) if scored else (0.0, -1), t))

    bins = Counter(min(int(j * 10), 9) / 10 for (j, _), _ in best)
    print("best exact Jaccard to a kept document (bin: count):", dict(sorted(bins.items())))
    print("no kept partner >= 0.5:", sum(j < 0.5 for (j, _), _ in best), "of", len(best))
    for (j, c), t in sorted(best, key=lambda x: x[0][0])[: args.show]:
        print(f"\n--- lowest: Jaccard {j:.3f}\nREMOVED: {t[:300]!r}\nKEPT:    {kept[c][:300]!r}" if c >= 0 else
              f"\n--- no partner\nREMOVED: {t[:300]!r}")
    for (j, c), t in random.sample(best, min(args.show, len(best))):
        if c >= 0:
            print(f"\n--- random: Jaccard {j:.3f}\nREMOVED: {t[:300]!r}\nKEPT:    {kept[c][:300]!r}")


if __name__ == "__main__":
    main()
