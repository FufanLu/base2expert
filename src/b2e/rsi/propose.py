"""Self-generated finance problems for the self-improvement loop (P7.5).

One round: the current model writes new problems, each as a report excerpt,
a question and a Python program that computes the answer from the excerpt's
numbers (FinQA's gold answers come from programs too). A problem is kept only
if every check passes:

  parse       the answer has <context>, <question> and <program> blocks
  program     b2e.rsi.sandbox runs it and it prints one finite number
  grounded    every number literal in the program (apart from unit constants
              such as 100 or 1000) appears in the excerpt, so the question is
              answerable from what the solver will see
  novel       not a near-duplicate of a problem kept earlier (5-gram Jaccard
              of the question + excerpt < 0.8)
  clean       not covering >= 60% of an eval item (b2e.data.decontam)

Kept problems are written in the FinQA row format (same instruction, number
ground truth), so b2e.data.synth / rl_pool / the RL trainer take them as they
are. Which of them are worth training on is decided afterwards by the
solver's own pass rate (rl_pool, [0.1, 0.9]).

    python -P -m b2e.rsi.propose --model /tmp/b2e-models/r_fin --n 4000 --round 1 \\
        --out $B2E_ROOT/data/rsi/round1 --decontam-index $B2E_ROOT/data/processed/eval_items
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

from b2e.data.finance import INSTRUCTION
from b2e.rsi.sandbox import run

TOPICS = [
    "year-over-year revenue growth", "gross margin", "operating margin", "net income change",
    "earnings per share", "share repurchases", "dividend payout ratio", "debt-to-equity ratio",
    "interest coverage", "free cash flow", "capital expenditure as a share of revenue",
    "working capital", "inventory turnover", "days sales outstanding", "effective tax rate",
    "segment revenue mix", "foreign exchange impact on sales", "pension obligations",
    "lease liabilities", "goodwill impairment", "stock-based compensation", "average cost of debt",
    "compound annual growth rate", "return on equity", "backlog conversion",
]
SECTORS = ["a regional bank", "a software company", "a retailer", "an airline", "a utility", "a chemicals maker",
           "a hospital operator", "an insurer", "a semiconductor company", "a consumer-goods company",
           "a REIT", "an oil and gas producer", "a logistics company", "a telecom operator"]

EXAMPLE = """<context>
The company's net revenues were $4,981 million in 2018 and $5,260 million in 2019. Cost of revenues rose from $2,940 million to $3,031 million over the same period.

| | 2019 | 2018 |
|---|---|---|
| net revenues | 5,260 | 4,981 |
| cost of revenues | 3,031 | 2,940 |
</context>
<question>
By how many percentage points did the gross margin change from 2018 to 2019?
</question>
<program>
margin_2019 = (5260 - 3031) / 5260 * 100
margin_2018 = (4981 - 2940) / 4981 * 100
print(round(margin_2019 - margin_2018, 2))
</program>"""

PROPOSER = """Write one new practice problem for financial-report question answering.

Topic: {topic}. Company: {sector}, fiscal years {y0} and {y1}.

1. In <context>, write a short excerpt of an annual report: two to four sentences and a small Markdown table with plausible, made-up numbers.
2. In <question>, ask one question that needs two to four arithmetic steps over numbers from the excerpt.
3. In <program>, write Python that computes the answer from those numbers, written exactly as they appear in the excerpt, and prints only the final number. Use only arithmetic and the math module.

Follow this format exactly:

{example}"""

UNIT_CONSTANTS = {0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 100.0, 365.0, 1000.0, 1e6, 1e9}
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def proposer_messages(rng: random.Random) -> list[dict]:
    y1 = rng.randint(2015, 2024)
    return [{"role": "user", "content": PROPOSER.format(topic=rng.choice(TOPICS), sector=rng.choice(SECTORS),
                                                        y0=y1 - 1, y1=y1, example=EXAMPLE)}]


def parse(text: str) -> dict | None:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    out = {}
    for tag in ("context", "question", "program"):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
        if not m or not m.group(1).strip():
            return None
        out[tag] = m.group(1).strip()
    out["program"] = re.sub(r"^```(?:python)?\s*|\s*```$", "", out["program"]).strip()
    return out


def numbers_in(text: str) -> set[float]:
    return {float(m.group(0).replace(",", "")) for m in _NUM.finditer(text)}


def grounded(program: str, context: str) -> bool:
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return False
    lits = {float(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, int | float) and not isinstance(n.value, bool)}
    have = numbers_in(context)
    return all(v in UNIT_CONSTANTS or any(abs(v - h) <= 1e-9 * max(1.0, abs(h)) for h in have) for v in lits)


def shingles(text: str, n: int = 5) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def to_row(prob: dict, value: float, round_idx: int, idx: int) -> dict:
    return {
        "data_source": "rsi_fin",
        "prompt": [{"role": "user", "content": f"{prob['context']}\n\nQuestion: {prob['question']}\n\n{INSTRUCTION}"}],
        "ability": "finance",
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"kind": "number", "value": value, "scale": ""})},
        "extra_info": {"split": "train", "id": f"r{round_idx}-{idx}", "program": prob["program"], "round": round_idx},
    }


def filter_problems(texts: list[str], round_idx: int, index=None, seen: list[set[str]] | None = None,
                    thinking: bool = False, outcomes: list[str] | None = None) -> tuple[list[dict], Counter]:
    """Rows for the problems that pass every check, and the count of each failure.

    thinking: the texts were sampled with the thinking block open, so one
    without </think> was cut off inside it, and any blocks in it are drafts
    (the same rule as the reward's, F#53). outcomes, if given, gets each
    text's result ("kept" or the failed check) in order."""
    stats: Counter = Counter()
    seen = [] if seen is None else seen
    rows: list[dict] = []

    def note(kind: str) -> None:
        stats[kind] += 1
        if outcomes is not None:
            outcomes.append(kind)

    for text in texts:
        if thinking and "</think>" not in text:
            note("unclosed")
            continue
        prob = parse(text)
        if prob is None:
            note("parse")
            continue
        res = run(prob["program"])
        if not res.ok:
            note("program:" + res.error.split(":")[0])
            continue
        if not grounded(prob["program"], prob["context"]):
            note("ungrounded")
            continue
        sh = shingles(prob["question"] + " " + prob["context"])
        if any(len(sh & s) / max(1, len(sh | s)) >= 0.8 for s in seen):
            note("duplicate")
            continue
        if index is not None and index.contaminated(prob["context"] + "\n" + prob["question"]):
            note("eval_overlap")
            continue
        seen.append(sh)
        assert res.value is not None
        rows.append(to_row(prob, res.value, round_idx, len(rows)))
        note("kept")
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4000, help="proposals to sample")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decontam-index", default="")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed + args.round)
    msgs = [proposer_messages(rng) for _ in range(args.n)]
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=8192, enable_prefix_caching=True)
    outs = llm.chat(msgs, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed),
                    chat_template_kwargs={"enable_thinking": True})
    index = None
    if args.decontam_index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.decontam_index)
    texts = [o.outputs[0].text for o in outs]
    outcomes: list[str] = []
    rows, stats = filter_problems(texts, args.round, index, thinking=True, outcomes=outcomes)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out / "problems.parquet", index=False)
    # every proposal with its outcome, to read why problems fail
    tokens = [len(o.outputs[0].token_ids) for o in outs]
    pd.DataFrame({"prompt": [m[0]["content"].split("\n\n")[1] for m in msgs], "text": texts,
                  "outcome": outcomes, "tokens": tokens}).to_parquet(args.out / "proposals.parquet", index=False)
    summary = {"round": args.round, "proposed": args.n, **dict(stats), "mean_tokens": round(sum(tokens) / len(tokens)),
               "cut_at_max_tokens": sum(o.outputs[0].finish_reason == "length" for o in outs)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
