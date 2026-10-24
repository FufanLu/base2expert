"""General-ability eval sets in verl's RL format, for measuring forgetting.

Mid-training and every later stage are checked on the same general sets, so
"did the domain training cost general ability" has a number at each step:
  mmlu_pro  English, 10 options, stratified: --per-category items from each
            of the 14 categories of the test split
  ceval     Chinese, 4 options, the whole val split (the test split has no
            answers)
  cmmlu     Chinese, 4 options, stratified: --per-subject items from each
            subject of the test split
  gsm8k     English grade-school math, the whole test split, numeric answer
Prompts are zero-shot and ask for the option letter in \\boxed{}, the same
format as the medical sets in b2e.data.law_med, so b2e.eval.run_eval scores
them with the choice verifier.

    python -P -m b2e.data.general_eval --hf $B2E_ROOT/data/hf --out $B2E_ROOT/data/processed/general
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from b2e.data.law_med import mcq_row

LETTERS = "ABCDEFGHIJ"


def mmlu_pro(hf: Path, per_category: int, seed: int) -> list[dict]:
    df = pd.read_parquet(next((hf / "TIGER-Lab/MMLU-Pro/data").glob("test-*.parquet")))
    # pick indices per group: since pandas 3, groupby.apply no longer passes the grouping column on
    keep = [i for _, g in df.groupby("category") for i in g.sample(n=min(per_category, len(g)), random_state=seed).index]
    df = df.loc[keep]
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


GSM8K_HEAD = "Solve the problem. Reason step by step, then give only the final number in \\boxed{}."


def gsm8k(processed: Path) -> list[dict]:
    """GSM8K test in our format. The copy in processed/gsm8k was made for the P0
    smoke by verl's example preprocessing: its prompt asks for the answer after
    "####" and its ground truth is a bare number string, which b2e's scorer does
    not take (docs/failures.md F#29). The question text is kept, the
    instruction replaced, and the gold wrapped as a number."""
    df = pd.read_parquet(processed / "gsm8k" / "test.parquet")
    rows = []
    for r in df.itertuples(index=False):
        q = list(r.prompt)[0]["content"]
        q = q.split("Let's think step by step")[0].strip()
        gold = float(str(r.reward_model["ground_truth"]).replace(",", ""))
        rows.append({
            "data_source": "gsm8k",
            "prompt": [{"role": "user", "content": f"{GSM8K_HEAD}\n\n{q}"}],
            "ability": "general",
            "reward_model": {"style": "rule", "ground_truth": json.dumps({"kind": "number", "value": gold})},
            "extra_info": {"split": "test", "id": str(r.extra_info["index"]), "category": "math"},
        })
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
        "gsm8k_test": gsm8k(args.out.parent),
    }
    for name, rows in sets.items():
        pd.DataFrame(rows).to_parquet(args.out / f"{name}.parquet", index=False)
        cats = pd.Series([r["extra_info"]["category"] for r in rows]).nunique()
        print(name, len(rows), "items,", cats, "categories")


if __name__ == "__main__":
    main()
