"""Law (CAIL2018) and medicine (MedQA, MedMCQA, CMExam) in verl's RL format.

Law: each case yields one of three tasks, chosen by a seeded draw so the pool
covers all three: charges (罪名), law articles (法条), prison term (刑期).
Medicine: multiple choice with the option letters as ground truth.

    python -P -m b2e.data.law_med --hf $B2E_ROOT/data/hf --out $B2E_ROOT/data/processed/law_med \\
        --law-train 30000 --law-eval 1000
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import pandas as pd

LAW_TASKS = {
    "charges": ("根据以下案件事实，判断被告人所犯的罪名。只写罪名本身（如「盗窃」），多个罪名用顿号分隔。"
                "先简要分析，最后把罪名写在 \\boxed{} 中。"),
    "articles": ("根据以下案件事实，判断应适用的《中华人民共和国刑法》条文。先简要分析，最后把条文编号写在 "
                 "\\boxed{} 中（只写数字，多个用顿号分隔，如 \\boxed{264、67}）。"),
    "term": ("根据以下案件事实，预测法院判处的有期徒刑刑期。先简要分析，最后把刑期写在 \\boxed{} 中"
             "（如 \\boxed{1年6个月}；判处死刑或无期徒刑则写「死刑」或「无期徒刑」；不判有期徒刑写 0个月）。"),
}
MCQ_EN = "Answer the multiple-choice question. Reason briefly, then give only the option letter in \\boxed{}."
MCQ_ZH = "回答下面的选择题。先简要分析，最后只把选项字母写在 \\boxed{} 中。"


def _lst(v):
    return ast.literal_eval(v) if isinstance(v, str) else list(v)


def law_rows(df: pd.DataFrame, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        task = rng.choice(list(LAW_TASKS))
        gt: dict[str, object]
        if task == "charges":
            gt = {"kind": "charges", "value": sorted(_lst(r.accusation))}
        elif task == "articles":
            gt = {"kind": "articles", "value": sorted(int(x) for x in _lst(r.relevant_articles))}
        else:
            gt = {"kind": "term", "months": float(r.imprisonment), "death": str(r.death_penalty) == "True",
                  "life": str(r.life_imprisonment) == "True"}
        rows.append({
            "data_source": f"cail2018_{task}",
            "prompt": [{"role": "user", "content": f"{LAW_TASKS[task]}\n\n案件事实：{r.fact}"}],
            "ability": "law",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
            "extra_info": {"split": split, "id": f"{split}-{i}", "task": task},
        })
    return rows


def mcq_row(source: str, question: str, options: dict[str, str], answer: str, split: str, idx, zh: bool) -> dict:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    head = MCQ_ZH if zh else MCQ_EN
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": f"{head}\n\n{question}\n{opts}"}],
        "ability": "medicine",
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"kind": "choice", "value": answer})},
        "extra_info": {"split": split, "id": str(idx)},
    }


def medqa(hf: Path, split: str) -> list[dict]:
    f = hf / "GBaker/MedQA-USMLE-4-options" / f"phrases_no_exclude_{split}.jsonl"
    rows = []
    for i, line in enumerate(f.read_text().splitlines()):
        ex = json.loads(line)
        rows.append(mcq_row("medqa", ex["question"], ex["options"], ex["answer_idx"], split, i, zh=False))
    return rows


def medmcqa(hf: Path, split: str) -> list[dict]:
    df = pd.read_parquet(next((hf / "openlifescienceai/medmcqa/data").glob(f"{split}-*.parquet")))
    rows = []
    for r in df.itertuples(index=False):
        if r.choice_type != "single":
            continue
        opts = {"A": r.opa, "B": r.opb, "C": r.opc, "D": r.opd}
        rows.append(mcq_row("medmcqa", r.question, opts, "ABCD"[int(r.cop)], split, r.id, zh=False))
    return rows


def cmexam(hf: Path, split: str) -> list[dict]:
    rows = []
    for i, line in enumerate((hf / "fzkuji/CMExam" / f"{split}.json").read_text().splitlines()):
        if not line.strip():
            continue
        ex = json.loads(line)
        opts = {o["key"]: o["value"] for o in _lst(ex["Options"])}
        ans = str(ex["Answer"]).strip().upper()
        if not ans or not set(ans) <= set(opts):
            continue
        rows.append(mcq_row("cmexam", ex["Question"], opts, ans, split, i, zh=True))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--law-train", type=int, default=30000)
    ap.add_argument("--law-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    cail = args.hf / "china-ai-law-challenge/cail2018/data"
    tr = pd.read_parquet(next(cail.glob("exercise_contest_train-*.parquet"))).sample(n=args.law_train, random_state=args.seed)
    te = pd.read_parquet(next(cail.glob("exercise_contest_test-*.parquet"))).sample(n=args.law_eval, random_state=args.seed)
    sets = {
        "cail_train": law_rows(tr, "train", rng),
        "cail_test": law_rows(te, "test", rng),
        "medqa_train": medqa(args.hf, "train"),
        "medqa_test": medqa(args.hf, "test"),
        "medmcqa_train": medmcqa(args.hf, "train"),
        "medmcqa_dev": medmcqa(args.hf, "validation"),
        "cmexam_train": cmexam(args.hf, "train"),
        "cmexam_test": cmexam(args.hf, "test"),
    }
    for name, rows in sets.items():
        pd.DataFrame(rows).to_parquet(args.out / f"{name}.parquet", index=False)
        tasks = pd.Series([r["data_source"] for r in rows]).value_counts().to_dict()
        print(name, len(rows), tasks)


if __name__ == "__main__":
    main()
