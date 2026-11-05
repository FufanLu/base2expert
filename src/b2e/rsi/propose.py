"""Self-generated finance problems for the self-improvement loop (P7.5).

One round: the current model writes new problems, each as a report excerpt,
a question and a Python program that computes the answer from the excerpt's
numbers (FinQA's gold answers come from programs too). A problem is kept only
if every check passes:

  parse       the answer has <context>, <question> and <program> blocks
  program     b2e.rsi.sandbox runs it and it prints one finite number
  grounded    every number literal in the program (apart from unit constants
              such as 100 or 1000) appears in the excerpt or the question, a
              percentage also as a fraction (25% as 0.25), so the problem is
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

Grounded mode (--docs, the 11-03 redesign; the excerpt-writing mode above stays
for the pilots' record): the proposer is given a real FinQA / TAT-QA train
excerpt (b2e.rsi.docs) and writes only a question and a program, so no figure
is invented and text and table cannot disagree. The checks change with it:

  grounded    against the real excerpt and the question
  asked       the answer equals one of the dataset's own answers on that
              excerpt (or is it in percent): the dataset's question again
  duplicate   the same excerpt and the same answer as a problem kept earlier
              (this round or a --prior round), or a 5-gram Jaccard >= 0.8 of
              the question with one of them. Problems on different excerpts
              are built from different figures, so they are not compared.

and the program prints the unrounded result (FinQA's golds are the program's
own output): rounding to 2 places cost 4 of the 60 problems read by hand
their right answer (docs/runs.md, 11-03).

    python -P -m b2e.rsi.propose --model /tmp/b2e-models/r_fin --n 2000 --round 1 \\
        --docs $B2E_ROOT/data/rsi/docs.parquet --no-thinking \\
        --out $B2E_ROOT/data/rsi/round1 --decontam-index $B2E_ROOT/data/processed/eval_items
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter
from collections.abc import Sequence
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
3. In <program>, write Python that computes the answer from those numbers and prints only the final number. Write each number with the excerpt's digits but without thousands separators (5,260 is 5260). Use only arithmetic, round() and the math module.

Follow this format exactly:

{example}"""

HINTS = ["a growth rate", "a ratio of two line items", "a share of a total", "an average over the years shown",
         "the difference between two segments or years", "a margin", "a compound annual growth rate",
         "a per-share or per-unit figure", "the change in a ratio", "a percentage-point change"]

GROUNDED_PROPOSER = """Write one new practice question for financial-report question answering about the excerpt below, with a Python program that computes its answer.

<excerpt>
{excerpt}
</excerpt>

The dataset already asks these questions about this excerpt; ask something else:
{asked}

Rules:
1. The question needs two to four arithmetic steps over numbers in the excerpt, preferably {hint} if the excerpt supports one.
2. Name the rows, columns and years it uses, so that it has exactly one answer. Do not ask for a number the excerpt states outright.
3. A "percentage change" is (new - old) / old * 100; a "change in percentage points" is the difference of two percentages; a decrease is negative.
4. The program uses only numbers from the excerpt or the question, with the excerpt's digits but no thousands separators (5,260 is 5260), and prints only the final number, unrounded. Use only arithmetic and the math module.

Answer in this format (the example is about another excerpt):

<question>
By how many percentage points did the gross margin change from 2018 to 2019?
</question>
<program>
margin_2019 = (5260 - 3031) / 5260 * 100
margin_2018 = (4981 - 2940) / 4981 * 100
print(margin_2019 - margin_2018)
</program>"""

UNIT_CONSTANTS = {0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 100.0, 365.0, 1000.0, 1e6, 1e9}
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_PCT = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*(?:%|percent\b)")


def proposer_messages(rng: random.Random) -> list[dict]:
    y1 = rng.randint(2015, 2024)
    return [{"role": "user", "content": PROPOSER.format(topic=rng.choice(TOPICS), sector=rng.choice(SECTORS),
                                                        y0=y1 - 1, y1=y1, example=EXAMPLE)}]


def grounded_messages(doc: dict, hint: str) -> list[dict]:
    asked = "\n".join(f"- {q}" for q in doc["questions"]) or "- (none)"
    return [{"role": "user", "content": GROUNDED_PROPOSER.format(excerpt=doc["excerpt"], asked=asked, hint=hint)}]


def parse(text: str, tags: Sequence[str] = ("context", "question", "program")) -> dict | None:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    out = {}
    for tag in tags:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
        if not m or not m.group(1).strip():
            return None
        out[tag] = m.group(1).strip()
    out["program"] = re.sub(r"^```(?:python)?\s*|\s*```$", "", out["program"]).strip()
    return out


def same_value(a: float, b: float) -> bool:
    """Two answers to one question: equal, or one is the other in percent."""
    return any(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-12) for x, y in ((a, b), (100 * a, b), (a, 100 * b)))


def numbers_in(text: str) -> set[float]:
    return {float(m.group(0).replace(",", "")) for m in _NUM.finditer(text)}


def grounded(program: str, context: str) -> bool:
    """Every number literal of the program is a unit constant or one of context's numbers.

    Compared by magnitude: the AST holds -85 as -(85). 0.5 is not a unit
    constant: both 10-31 pilot programs with ** 0.5 took a one-year CAGR as
    a two-year one, and this check was what caught them."""
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return False
    lits = {abs(float(n.value)) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, int | float) and not isinstance(n.value, bool)}
    have = {abs(h) for h in numbers_in(context)}
    have |= {abs(float(m.group(1).replace(",", ""))) / 100 for m in _PCT.finditer(context)}
    return all(v in UNIT_CONSTANTS or any(abs(v - h) <= 1e-9 * max(1.0, h) for h in have) for v in lits)


def shingles(text: str, n: int = 5) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def to_row(prob: dict, value: float, round_idx: int, idx: int, doc: dict | None = None) -> dict:
    extra = {"split": "train", "id": f"r{round_idx}-{idx}", "program": prob["program"], "round": round_idx}
    if doc is not None:
        extra |= {"doc_id": doc["doc_id"], "doc_source": doc["source"]}
    return {
        "data_source": "rsi_fin",
        "prompt": [{"role": "user", "content": f"{prob['context']}\n\nQuestion: {prob['question']}\n\n{INSTRUCTION}"}],
        "ability": "finance",
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"kind": "number", "value": value, "scale": ""})},
        "extra_info": extra,
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
        if not grounded(prob["program"], prob["context"] + "\n" + prob["question"]):
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


Taken = dict[str, list[tuple[set[str], float]]]


def duplicate(sh: set[str], value: float, past: Sequence[tuple[set[str], float]]) -> bool:
    """A problem on an excerpt repeats one already kept there: the same answer (or it in percent),
    or a question whose 5-gram shingles overlap one of theirs by a Jaccard of 0.8 or more."""
    return any(same_value(value, v) or len(sh & s) / max(1, len(sh | s)) >= 0.8 for s, v in past)


def filter_grounded(texts: list[str], docs: Sequence[dict], round_idx: int, prior: Taken | None = None, index=None,
                    thinking: bool = False, outcomes: list[str] | None = None) -> tuple[list[dict], Counter]:
    """filter_problems for grounded proposals: texts[i] was written about docs[i] (b2e.rsi.docs records).

    prior maps a doc_id to the (question 5-gram shingles, answer) of the problems kept on that
    excerpt in earlier rounds (prior_problems); this round's kept problems are added to it."""
    stats: Counter = Counter()
    taken: Taken = {} if prior is None else prior
    rows: list[dict] = []

    def note(kind: str) -> None:
        stats[kind] += 1
        if outcomes is not None:
            outcomes.append(kind)

    for text, doc in zip(texts, docs, strict=True):
        if thinking and "</think>" not in text:
            note("unclosed")
            continue
        prob = parse(text, tags=("question", "program"))
        if prob is None:
            note("parse")
            continue
        res = run(prob["program"])
        if not res.ok:
            note("program:" + res.error.split(":")[0])
            continue
        if not grounded(prob["program"], doc["excerpt"] + "\n" + prob["question"]):
            note("ungrounded")
            continue
        assert res.value is not None
        value = res.value
        if any(same_value(value, float(a)) for a in doc["answers"]):
            note("asked")
            continue
        sh = shingles(prob["question"])
        past = taken.setdefault(doc["doc_id"], [])
        if duplicate(sh, value, past):
            note("duplicate")
            continue
        if index is not None and index.contaminated(doc["excerpt"] + "\n" + prob["question"]):
            note("eval_overlap")
            continue
        past.append((sh, value))
        rows.append(to_row({**prob, "context": doc["excerpt"]}, value, round_idx, len(rows), doc))
        note("kept")
    return rows, stats


def draw_grounded(pool: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[str]]:
    """n (excerpt, hint) pairs to propose about. Up to the pool's size, a sample without replacement
    and a hint drawn for each (the pilot's draw). Beyond it, every excerpt once before any twice,
    and one excerpt's hints all different for up to len(HINTS) passes: drawn with replacement, a
    pair would come back with the same prompt and, under one seed, the same proposal, which the
    duplicate check then drops (about one in nine of a 10,000-proposal round)."""
    if n <= len(pool):
        docs = rng.sample(pool, n)
        return docs, [rng.choice(HINTS) for _ in docs]
    order = {d["doc_id"]: rng.sample(HINTS, len(HINTS)) for d in pool}
    docs: list[dict] = []
    hints: list[str] = []
    passes = 0
    while len(docs) < n:
        for d in rng.sample(pool, len(pool)):
            docs.append(d)
            hints.append(order[d["doc_id"]][passes % len(HINTS)])
        passes += 1
    return docs[:n], hints[:n]


def prior_problems(rows: Sequence[dict]) -> Taken:
    """doc_id -> (question shingles, answer) of the grounded problems among earlier rounds' rows."""
    from b2e.rsi.docs import split_prompt

    out: Taken = {}
    for r in rows:
        doc = (r.get("extra_info") or {}).get("doc_id")
        if not doc:
            continue
        _, question = split_prompt(r["prompt"][0]["content"])
        value = float(json.loads(r["reward_model"]["ground_truth"])["value"])
        out.setdefault(doc, []).append((shingles(question), value))
    return out


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
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="sample with the thinking block open (--no-thinking: write the problem directly)")
    ap.add_argument("--docs", default="", help="b2e.rsi.docs parquet: propose about real excerpts (grounded mode)")
    ap.add_argument("--prior", nargs="*", default=[],
                    help="grounded mode: earlier rounds' problems.parquet, whose problems are not asked again")
    args = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed + args.round)
    docs = hints = None
    if args.docs:
        docs, hints = draw_grounded(pd.read_parquet(args.docs).to_dict("records"), args.n, rng)
        msgs = [grounded_messages(d, h) for d, h in zip(docs, hints, strict=True)]
    else:
        msgs = [proposer_messages(rng) for _ in range(args.n)]
    # a real excerpt can run to a few thousand tokens before the answer starts
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=16384 if docs else 8192,
              enable_prefix_caching=True)
    outs = llm.chat(msgs, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed),
                    chat_template_kwargs={"enable_thinking": args.thinking})
    index = None
    if args.decontam_index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.decontam_index)
    texts = [o.outputs[0].text for o in outs]
    outcomes: list[str] = []
    if docs is not None:
        prior = prior_problems([r for p in args.prior for r in pd.read_parquet(p).to_dict("records")])
        rows, stats = filter_grounded(texts, docs, args.round, prior, index, thinking=args.thinking,
                                      outcomes=outcomes)
    else:
        rows, stats = filter_problems(texts, args.round, index, thinking=args.thinking, outcomes=outcomes)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out / "problems.parquet", index=False)
    # every proposal with its outcome, to read why problems fail
    tokens = [len(o.outputs[0].token_ids) for o in outs]
    what = ({"doc_id": [d["doc_id"] for d in docs], "hint": hints} if docs is not None
            else {"prompt": [m[0]["content"].split("\n\n")[1] for m in msgs]})
    pd.DataFrame({**what, "text": texts, "outcome": outcomes, "tokens": tokens}).to_parquet(
        args.out / "proposals.parquet", index=False)
    summary = {"round": args.round, "thinking": args.thinking, "grounded": docs is not None, "proposed": args.n,
               **dict(stats),
               "mean_tokens": round(sum(tokens) / len(tokens)),
               "cut_at_max_tokens": sum(o.outputs[0].finish_reason == "length" for o in outs)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
