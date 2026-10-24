"""Show the context of every PII match in the first documents of a parquet file,
to check the scrubber for false positives (e.g. section numbers as IPs).

    python -P tools/pii_inspect.py --docs 2500 --kinds ip,card file.parquet
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq

from b2e.data import pii

PATTERNS = {"ip": pii.IPV4, "card": pii.CARD, "mobile": pii.CN_MOBILE, "id": pii.CN_ID, "email": pii.EMAIL}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2500)
    ap.add_argument("--kinds", default="ip,card,mobile,id")
    ap.add_argument("file")
    a = ap.parse_args()
    texts = pq.read_table(a.file, columns=["text"]).column("text").to_pylist()[: a.docs]
    for kind in a.kinds.split(","):
        pat = PATTERNS[kind]
        n = 0
        for t in texts:
            for m in pat.finditer(t):
                n += 1
                if n <= 25:
                    ctx = t[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
                    print(f"[{kind}] {m.group(0)!r} :: {ctx}")
        print(f"== {kind}: {n} raw matches (before checksum checks) in {len(texts)} docs\n")


if __name__ == "__main__":
    main()
