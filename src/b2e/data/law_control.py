"""P9's control for the law-articles prompt (F#92): the same questions with another example.

The articles prompt (`LAW_TASKS["articles"]` in b2e.data.law_med) shows `\\boxed{264、67}`, and 67
(voluntary surrender, 自首) is a general-part article (1-101) that no CAIL2018 label holds; S-law
added general-part articles to a third of its articles answers before RL. To measure what the
example cost, the articles rows of an eval set are written again with the example replaced, and
the models are evaluated on both. The new examples come from the labels, not from memory: the
most common one-article label and the most common two-article label of the pool, every article in
the specific part (>= 102) and present in the labels.

    python -P -m b2e.data.law_control --pool $P/law_med/cail_train.parquet \\
        --data $P/law_med/cail_test.parquet --out $P/law_med/control
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from b2e.data.law_med import LAW_TASKS

OLD = "\\boxed{264、67}"
SOURCE = "cail2018_articles"
SPECIFIC_PART = 102  # articles 1-101 are the general part of the Criminal Law


def labels(df: pd.DataFrame) -> list[tuple[int, ...]]:
    """Each articles row's gold article set, sorted."""
    rows = df[df["data_source"] == SOURCE]
    return [tuple(sorted(json.loads(r["ground_truth"])["value"])) for r in rows["reward_model"]]


def pick_examples(golds: list[tuple[int, ...]]) -> dict[int, tuple[int, ...]]:
    """The most common label of one and of two articles, each article in the specific part."""
    out = {}
    for size in (1, 2):
        counts = Counter(g for g in golds if len(g) == size and min(g) >= SPECIFIC_PART)
        if not counts:
            raise ValueError(f"no {size}-article label in the specific part")
        out[size] = counts.most_common(1)[0][0]
    return out


def check_example(example: tuple[int, ...], golds: list[tuple[int, ...]]) -> None:
    seen = {a for g in golds for a in g}
    for a in example:
        if a < SPECIFIC_PART:
            raise ValueError(f"article {a} is in the general part, which no label holds")
        if a not in seen:
            raise ValueError(f"article {a} is in no label")


def box(example: tuple[int, ...]) -> str:
    return "\\boxed{" + "、".join(map(str, example)) + "}"


def swap(df: pd.DataFrame, example: tuple[int, ...]) -> pd.DataFrame:
    """The articles rows, their prompt's example replaced; any other change to the prompt is an error."""
    rows = df[df["data_source"] == SOURCE].copy()
    new = box(example)
    prompts = []
    for p in rows["prompt"]:
        msgs = [dict(m) for m in p]
        text = msgs[0]["content"]
        if text.count(OLD) != 1 or not text.startswith(LAW_TASKS["articles"]):
            raise ValueError("an articles prompt without the expected instruction and example")
        msgs[0]["content"] = text.replace(OLD, new)
        prompts.append(msgs)
    rows["prompt"] = prompts
    return rows.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="the labels the examples are drawn from")
    ap.add_argument("--data", type=Path, required=True, help="the eval set whose articles rows are rewritten")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    pool, data = labels(pd.read_parquet(args.pool)), pd.read_parquet(args.data)
    sizes = Counter(len(g) for g in pool)
    report: dict = {"pool": str(args.pool), "pool_rows": len(pool),
                    "label_sizes": {k: sizes[k] for k in sorted(sizes)},
                    "general_part_in_labels": sum(min(g) < SPECIFIC_PART for g in pool),
                    "top_labels": [[list(g), n] for g, n in Counter(pool).most_common(10)]}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.data.stem
    df = data[data["data_source"] == SOURCE]
    report["data"], report["data_rows"] = str(args.data), len(df)
    df.reset_index(drop=True).to_parquet(args.out / f"{stem}_articles_orig.parquet")  # the same rows as they were
    for size, ex in pick_examples(pool).items():
        check_example(ex, pool)
        out = swap(data, ex)
        out.to_parquet(args.out / f"{stem}_articles_ex{size}.parquet")
        report[f"ex{size}"] = {"example": box(ex), "pool_rows_with_this_label": Counter(pool)[ex]}
    (args.out / f"{stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
