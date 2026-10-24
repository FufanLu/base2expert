"""General-ability eval sets in verl's RL format, for measuring forgetting.

Mid-training and every later stage are checked on the same general sets, so
"did the domain training cost general ability" has a number at each step:
  mmlu_pro  English, 10 options, stratified: --per-category items from each
            of the 14 categories of the test split
  ceval     Chinese, 4 options, the whole val split (the test split has no
            answers)
  cmmlu     Chinese, 4 options, stratified: --per-subject items from each
            subject of the test split
Prompts are zero-shot and ask for the option letter in \\boxed{}, the same
format as the medical sets in b2e.data.law_med, so b2e.eval.run_eval scores
them with the choice verifier.

    python -P -m b2e.data.general_eval --hf $B2E_ROOT/data/hf --out $B2E_ROOT/data/processed/general
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from b2e.data.law_med import mcq_row

LETTERS = "ABCDEFGHIJ"


def mmlu_pro(hf: Path, per_category: int, seed: int) -> list[dict]:
    df = pd.read_parquet(next((hf / "TIGER-Lab/MMLU-Pro/data").glob("test-*.parquet")))
    df = df.groupby("category", group_keys=False).apply(lambda g: g.sample(n=min(per_category, len(g)), random_state=seed))
    rows = []
    for r in df.itertuples(index=False):
        opts = {LETTERS[i]: o for i, o in enumerate(r.options)}
        row = mcq_row("mmlu_pro", r.question, opts, r.answer, "test", r.question_id, zh=False)
        row["ability"] = "general"
        row["extra_info"]["category"] = r.category
        rows.append(row)
    return rows


def ceval(hf: Path) -> list[dict]:
    rows = []
    for f in sorted((hf / "ceval/ceval-exam").glob("*/val-*.parquet")):
        subject = f.parent.name
        for r in pd.read_parquet(f).itertuples(index=False):
            opts = {k: getattr(r, k) for k in "ABCD"}
            row = mcq_row("ceval", r.question, opts, r.answer, "val", f"{subject}-{r.id}", zh=True)
            row["ability"] = "general"
            row["extra_info"]["category"] = subject
            rows.append(row)
    return rows


def cmmlu(hf: Path, per_subject: int, seed: int) -> list[dict]:
    rows = []
    with zipfile.ZipFile(hf / "haonan-li/cmmlu/cmmlu_v1_0_1.zip") as zf:
        for name in sorted(zf.namelist()):
            if not (name.startswith("test/") and name.endswith(".csv")):
                continue
            subject = Path(name).stem
            df = pd.DataFrame(list(csv.DictReader(io.StringIO(zf.read(name).decode("utf-8")))))
            df = df.sample(n=min(per_subject, len(df)), random_state=seed)
            for i, r in df.iterrows():
                opts = {k: r[k] for k in "ABCD"}
                row = mcq_row("cmmlu", r["Question"], opts, r["Answer"].strip(), "test", f"{subject}-{i}", zh=True)
                row["ability"] = "general"
                row["extra_info"]["category"] = subject
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=100, help="MMLU-Pro items per category (14 categories)")
    ap.add_argument("--per-subject", type=int, default=30, help="CMMLU items per subject (67 subjects)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sets = {
        "mmlu_pro_test": mmlu_pro(args.hf, args.per_category, args.seed),
        "ceval_val": ceval(args.hf),
        "cmmlu_test": cmmlu(args.hf, args.per_subject, args.seed),
    }
    for name, rows in sets.items():
        pd.DataFrame(rows).to_parquet(args.out / f"{name}.parquet", index=False)
        cats = pd.Series([r["extra_info"]["category"] for r in rows]).nunique()
        print(name, len(rows), "items,", cats, "categories")


if __name__ == "__main__":
    main()
