"""FinQA and TAT-QA in verl's RL format, with numeric ground truth.

Each example becomes one user message: the report excerpt (text before the
table, the table as Markdown, text after), the question, and the answer
format. Ground truth is a dict the reward function understands:

  {"kind": "number", "value": 0.14464, "scale": ""}      FinQA (fractions for %)
  {"kind": "number", "value": 1.9, "scale": "million"}   TAT-QA arithmetic/count
  {"kind": "text", "value": "yes"}                       FinQA yes/no questions

TAT-QA span and multi-span questions are left out of the RL pool: their
answers are copied phrases, checked by string overlap rather than a
verifier, so they go to SFT only.

TAT-QA's test split is read from tatqa_dataset_test_gold.json, the version
with answers. Its dev split is the TAT-QA half of the RL val set, so the
TAT-QA eval is test (F#66: dev was used for both until 10-31).

    python -P -m b2e.data.finance --finqa DIR --tatqa DIR --out DIR
    python -P -m b2e.data.finance --finqa DIR --tatqa DIR --out DIR --only tatqa_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INSTRUCTION = ("Answer the question using the financial report excerpt. Show the calculation step by step, "
               "then give the final answer as a single number in \\boxed{} (write percentages with %).")


def md_table(rows: list[list]) -> str:
    if not rows:
        return ""
    cells = [[str(c).strip() for c in r] for r in rows]
    width = max(len(r) for r in cells)
    cells = [r + [""] * (width - len(r)) for r in cells]
    head = "| " + " | ".join(cells[0]) + " |"
    sep = "|" + "---|" * width
    body = ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join([head, sep, *body])


def finqa_rows(path: Path, split: str) -> list[dict]:
    out = []
    for ex in json.loads(path.read_text()):
        qa = ex["qa"]
        ans = qa["exe_ans"]
        gt = ({"kind": "number", "value": float(ans), "scale": ""} if isinstance(ans, int | float)
              else {"kind": "text", "value": str(ans).strip().lower()})
        context = "\n\n".join(x for x in [" ".join(ex.get("pre_text", [])), md_table(ex.get("table", [])),
                                          " ".join(ex.get("post_text", []))] if x)
        out.append({
            "data_source": "finqa",
            "prompt": [{"role": "user", "content": f"{context}\n\nQuestion: {qa['question']}\n\n{INSTRUCTION}"}],
            "ability": "finance",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"split": split, "id": ex["id"], "program": qa.get("program", "")},
        })
    return out


def tatqa_rows(path: Path, split: str) -> list[dict]:
    out = []
    for ex in json.loads(path.read_text()):
        table = md_table(ex.get("table", {}).get("table", []))
        paras = "\n".join(p["text"] for p in sorted(ex.get("paragraphs", []), key=lambda p: p.get("order", 0)))
        for q in ex.get("questions", []):
            if q.get("answer_type") not in ("arithmetic", "count"):
                continue
            try:
                value = float(q["answer"])
            except (TypeError, ValueError):
                continue
            gt = {"kind": "number", "value": value, "scale": q.get("scale", "")}
            out.append({
                "data_source": "tatqa",
                "prompt": [{"role": "user", "content": f"{table}\n\n{paras}\n\nQuestion: {q['question']}\n\n{INSTRUCTION}"}],
                "ability": "finance",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                "extra_info": {"split": split, "id": q["uid"], "answer_type": q["answer_type"]},
            })
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finqa", type=Path, required=True)
    ap.add_argument("--tatqa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="+", help="write only these sets; the other files in --out stay as they are")
    args = ap.parse_args(argv)
    import pandas as pd

    sets = {
        "finqa_train": lambda: finqa_rows(args.finqa / "train.json", "train"),
        "finqa_dev": lambda: finqa_rows(args.finqa / "dev.json", "dev"),
        "finqa_test": lambda: finqa_rows(args.finqa / "test.json", "test"),
        "tatqa_train": lambda: tatqa_rows(args.tatqa / "tatqa_dataset_train.json", "train"),
        "tatqa_dev": lambda: tatqa_rows(args.tatqa / "tatqa_dataset_dev.json", "dev"),
        "tatqa_test": lambda: tatqa_rows(args.tatqa / "tatqa_dataset_test_gold.json", "test"),
    }
    unknown = sorted(set(args.only or []) - set(sets))
    if unknown:
        raise SystemExit(f"--only: no set named {unknown}; the sets are {sorted(sets)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, make in sets.items():
        if args.only and name not in args.only:
            continue
        rows = make()
        pd.DataFrame(rows).to_parquet(args.out / f"{name}.parquet", index=False)
        print(name, len(rows))


if __name__ == "__main__":
    main()
