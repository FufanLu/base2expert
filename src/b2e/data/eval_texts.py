"""Collect the text of every evaluation split, for decontamination.

Everything a test question shows the model counts: the question, the
options, and the context passage. The context matters most for finance:
FinQA and TAT-QA passages are taken from company annual reports, and the
finance mid-training corpus may contain the same reports. Training splits
are not collected (they are training data).

    python -P -m b2e.data.eval_texts --hf $B2E_ROOT/data/hf --raw $B2E_ROOT/data/raw \\
        --out $B2E_ROOT/data/processed/eval_texts.jsonl

One JSON line per item: {"set": ..., "split": ..., "text": ...}. Sets whose
files are missing are skipped and listed at the end.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pandas as pd


def _join(*parts) -> str:
    out = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, list | tuple):
            out.append(" ".join(str(x) for x in p if x))
        elif isinstance(p, dict):
            out.append(" ".join(f"{k}. {v}" for k, v in p.items()))
        else:
            out.append(str(p))
    return "\n".join(x for x in out if x)


def finqa(raw: Path) -> Iterator[tuple[str, str]]:
    for split in ("dev", "test"):
        f = raw / "finqa" / f"{split}.json"
        if not f.exists():
            continue
        for ex in json.loads(f.read_text()):
            table = "\n".join(" | ".join(map(str, row)) for row in ex.get("table", []))
            yield split, _join(ex.get("pre_text"), table, ex.get("post_text"), ex.get("qa", {}).get("question"))


def convfinqa(raw: Path) -> Iterator[tuple[str, str]]:
    z = raw / "convfinqa" / "data.zip"
    if not z.exists():
        return
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if not name.endswith(".json") or "train" in name:
                continue
            for ex in json.loads(zf.read(name)):
                table = "\n".join(" | ".join(map(str, row)) for row in ex.get("table", []))
                qs = ex.get("annotation", {}).get("dialogue_break") or []
                yield Path(name).stem, _join(ex.get("pre_text"), table, ex.get("post_text"), qs)


def tatqa(hf: Path) -> Iterator[tuple[str, str]]:
    for split, fn in (("dev", "tatqa_dataset_dev.json"), ("test", "tatqa_dataset_test.json")):
        f = hf / "next-tat/TAT-QA" / fn
        if not f.exists():
            continue
        for ex in json.loads(f.read_text()):
            table = "\n".join(" | ".join(map(str, r)) for r in ex.get("table", {}).get("table", []))
            paras = [p.get("text") for p in ex.get("paragraphs", [])]
            qs = [q.get("question") for q in ex.get("questions", [])]
            yield split, _join(table, paras, qs)


def medqa(hf: Path) -> Iterator[tuple[str, str]]:
    f = hf / "GBaker/MedQA-USMLE-4-options" / "phrases_no_exclude_test.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            ex = json.loads(line)
            yield "test", _join(ex.get("question"), ex.get("options"))


def parquet_rows(files: list[Path], cols: list[str], split: str) -> Iterator[tuple[str, str]]:
    for f in files:
        df = pd.read_parquet(f)
        use = [c for c in cols if c in df.columns]
        for row in df[use].itertuples(index=False):
            yield split, _join(*row)


def medmcqa(hf: Path) -> Iterator[tuple[str, str]]:
    d = hf / "openlifescienceai/medmcqa/data"
    for split in ("validation", "test"):
        yield from parquet_rows(sorted(d.glob(f"{split}-*.parquet")), ["question", "opa", "opb", "opc", "opd"], split)


def cmexam(hf: Path) -> Iterator[tuple[str, str]]:
    for split in ("valid", "test"):
        f = hf / "fzkuji/CMExam" / f"{split}.json"
        if not f.exists():
            continue
        text = f.read_text()
        items = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x]
        for ex in items:
            yield split, _join(ex.get("Question"), ex.get("Options"))


def cail2018(hf: Path) -> Iterator[tuple[str, str]]:
    d = hf / "china-ai-law-challenge/cail2018/data"
    for split in ("exercise_contest_test", "exercise_contest_valid", "first_stage_test", "final_test"):
        yield from parquet_rows(sorted(d.glob(f"{split}-*.parquet")), ["fact"], split)


def mmlu_pro(hf: Path) -> Iterator[tuple[str, str]]:
    d = hf / "TIGER-Lab/MMLU-Pro/data"
    for split in ("validation", "test"):
        yield from parquet_rows(sorted(d.glob(f"{split}-*.parquet")), ["question", "options"], split)


def ceval(hf: Path) -> Iterator[tuple[str, str]]:
    d = hf / "ceval/ceval-exam"
    for split in ("dev", "val", "test"):
        yield from parquet_rows(sorted(d.glob(f"*/{split}-*.parquet")), ["question", "A", "B", "C", "D"], split)


def cmmlu(hf: Path) -> Iterator[tuple[str, str]]:
    z = hf / "haonan-li/cmmlu" / "cmmlu_v1_0_1.zip"
    if not z.exists():
        return
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            split = "test" if "test/" in name else "dev"  # zip paths look like "test/agronomy.csv"
            reader = csv.DictReader(io.StringIO(zf.read(name).decode("utf-8")))
            for row in reader:
                yield split, _join(row.get("Question"), row.get("A"), row.get("B"), row.get("C"), row.get("D"))


def ifeval(hf: Path) -> Iterator[tuple[str, str]]:
    for f in sorted((hf / "google/IFEval").rglob("*.jsonl")) + sorted((hf / "google/IFEval").rglob("*.parquet")):
        if f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                yield "train", json.loads(line).get("prompt", "")  # IFEval's only split is its test set
        else:
            yield from parquet_rows([f], ["prompt"], "train")


def gsm8k(processed: Path) -> Iterator[tuple[str, str]]:
    f = processed / "gsm8k" / "test.parquet"
    if f.exists():
        for p in pd.read_parquet(f)["prompt"]:
            yield "test", str(list(p)[0]["content"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--processed", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    processed = args.processed or args.out.parent

    loaders = {
        "finqa": lambda: finqa(args.raw), "convfinqa": lambda: convfinqa(args.raw),
        "tatqa": lambda: tatqa(args.hf), "medqa": lambda: medqa(args.hf), "medmcqa": lambda: medmcqa(args.hf),
        "cmexam": lambda: cmexam(args.hf), "cail2018": lambda: cail2018(args.hf), "mmlu_pro": lambda: mmlu_pro(args.hf),
        "ceval": lambda: ceval(args.hf), "cmmlu": lambda: cmmlu(args.hf), "ifeval": lambda: ifeval(args.hf),
        "gsm8k": lambda: gsm8k(processed),
    }
    counts: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for name, load in loaders.items():
            for split, text in load():
                if text and text.strip():
                    f.write(json.dumps({"set": name, "split": split, "text": text}, ensure_ascii=False) + "\n")
                    counts[(name, split)] += 1
    for (name, split), n in sorted(counts.items()):
        print(f"{name:<10} {split:<24} {n}")
    missing = [n for n in loaders if not any(k[0] == n for k in counts)]
    print("missing:", missing)


if __name__ == "__main__":
    main()
