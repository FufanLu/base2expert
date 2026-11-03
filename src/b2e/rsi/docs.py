"""Real report excerpts for grounded proposals (P7.5, the 11-03 redesign).

In the pilots the proposer wrote its own excerpt. Of the problems a round
would have kept, 9 of 20 without thinking were flawed, 3 of them because the
text and the table disagreed (docs/runs.md, 11-03). A grounded proposal starts
from a real excerpt instead: one record per distinct FinQA or TAT-QA train
excerpt, with the questions the dataset already asks about it and their
numeric answers. b2e.rsi.propose shows the proposer those questions and drops
a proposal that asks one of them again (same excerpt, same answer).

Excerpts that cover an eval item (b2e.data.decontam, the same rule as the
training data) are left out, so no proposal is written over an eval excerpt.

    python -P -m b2e.rsi.docs --out $B2E_ROOT/data/rsi/docs.parquet \\
        --decontam-index $B2E_ROOT/data/processed/eval_items \\
        $B2E_ROOT/data/processed/finance/finqa_train.parquet $B2E_ROOT/data/processed/finance/tatqa_train.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from b2e.data.finance import INSTRUCTION

SEP = "\n\nQuestion: "


def split_prompt(content: str) -> tuple[str, str]:
    """(excerpt, question) of a b2e.data.finance prompt, f"{excerpt}\\n\\nQuestion: {q}\\n\\n{INSTRUCTION}"."""
    if not content.endswith(INSTRUCTION) or SEP not in content:
        raise ValueError(f"not a b2e.data.finance prompt: {content[-120:]!r}")
    body = content[: -len(INSTRUCTION)].rstrip("\n")
    excerpt, question = body.rsplit(SEP, 1)  # the excerpt may itself contain "Question:"
    return excerpt, question


def doc_id(excerpt: str) -> str:
    return hashlib.sha1(excerpt.encode()).hexdigest()[:16]


def build(rows: Iterable[dict]) -> list[dict]:
    """One record per distinct excerpt: doc_id, source, excerpt, the dataset's questions
    on it, and the numeric answers among them (TAT-QA values in the table's units)."""
    docs: dict[str, dict] = {}
    for r in rows:
        excerpt, question = split_prompt(r["prompt"][0]["content"])
        gt = json.loads(r["reward_model"]["ground_truth"])
        d = docs.setdefault(doc_id(excerpt), {"doc_id": doc_id(excerpt), "source": r["data_source"],
                                               "excerpt": excerpt, "questions": [], "answers": [], "row_ids": []})
        d["questions"].append(question)
        d["row_ids"].append(str(r["extra_info"]["id"]))
        if gt["kind"] == "number":
            d["answers"].append(float(gt["value"]))
    return list(docs.values())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decontam-index", default="", help="b2e.data.decontam ItemNgramIndex prefix")
    ap.add_argument("inputs", nargs="+", help="b2e.data.finance train parquet files")
    args = ap.parse_args(argv)
    import pandas as pd

    rows = [r for p in args.inputs for r in pd.read_parquet(p).to_dict("records")]
    docs = build(rows)
    dropped = 0
    if args.decontam_index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.decontam_index)
        kept = [d for d in docs if not index.contaminated(d["excerpt"])]
        dropped = len(docs) - len(kept)
        docs = kept
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(docs).to_parquet(args.out, index=False)
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    print(json.dumps({"rows": len(rows), "docs": len(docs), "eval_overlap_dropped": dropped, "by_source": by_source}))


if __name__ == "__main__":
    main()
