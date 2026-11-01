"""Preference pairs for offline preference optimization (P5), from a policy's own samples.

b2e.data.synth samples a policy k times per prompt and keeps every text
(<out>_parts/samples_*.parquet) beside the pass-rate stats (stats_*.parquet).
Where some finished answers to a prompt are right and some wrong, a right one
is chosen and a wrong one rejected: pairs from the policy's own distribution,
labelled by the verifier. P5's pairs come from S-fin's k = 8 samples of every
FinQA and TAT-QA train prompt at the RL rollout settings (11400102), the
samples P6's pool was picked from, so preference optimization and the P6 RL
arms start from the same policy and the same prompts.

Each choice guards against a known failure:
  finished answers only   a cut answer scores 0 for running out of tokens, not
                          for a wrong result; as rejected it would teach
                          finishing, not correctness (F#71: half of P6's pool
                          was there because of cuts). The same for an answer
                          that stopped inside its thinking block.
  each sample once        min(right, wrong) pairs a prompt, so a prompt with 4
                          right and 4 wrong answers counts 4 times, not 16.
  matched by length       the matching of the shorter list into the longer one
                          with the least summed token difference: DPO learns any
                          length gap between chosen and rejected, and S-fin's
                          wrong answers run ~100 tokens longer than its right
                          ones to the same prompt.
  current verifier        every sample is scored again, so a verifier change
                          since the sampling cannot leave a pair labelled by the
                          old one; the summary counts the scores that changed.
  held-out prompts        --test-per-source pairable prompts per source are set
                          aside before pairing, for reward-model and implicit-
                          reward accuracy on prompts no train pair comes from.
Graded scores (0 < s < 1, CAIL term) are neither chosen nor rejected.

Rows are TRL's conversational preference format: prompt (the messages), chosen
and rejected (one assistant message each, written "<think>\\n...\\n</think>\\n\\n..."
as synth's SFT rows), plus the sample indices, token counts and ground truth.

    python -P -m b2e.data.pref_pairs --parts $B/data/rl_pool/s_fin_k8_parts \\
        --prompts $B/data/processed/finance/finqa_train.parquet $B/data/processed/finance/tatqa_train.parquet \\
        --out $B/data/processed/pref_fin --test-per-source 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from b2e.data.synth import split_thinking
from b2e.reward.score import compute_score


def match_by_length(short: Sequence[float], long: Sequence[float]) -> list[tuple[int, int]]:
    """Pairs (i, j): every position of short matched to its own position of long
    (len(short) <= len(long)), with the least summed |short[i] - long[j]|.

    On a line, an optimal matching never crosses (the cost is a Monge array), so
    matching the sorted lists in order with a skip-or-take DP over long finds it.
    """
    if len(short) > len(long):
        raise ValueError("short is longer than long")
    si = sorted(range(len(short)), key=lambda i: short[i])
    li = sorted(range(len(long)), key=lambda j: long[j])
    n, m = len(si), len(li)
    inf = float("inf")
    # cost[i][j]: the first i of short (sorted) matched into the first j of long (sorted)
    cost = [[0.0] * (m + 1)] + [[inf] * (m + 1) for _ in range(n)]
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            take = cost[i - 1][j - 1] + abs(short[si[i - 1]] - long[li[j - 1]])
            cost[i][j] = min(cost[i][j - 1], take)
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0:
        if cost[i][j] == cost[i][j - 1] and j > i:
            j -= 1
        else:
            pairs.append((si[i - 1], li[j - 1]))
            i, j = i - 1, j - 1
    return sorted(pairs)


def pair_prompt(scores: Sequence[float], finish: Sequence[str], tokens: Sequence[float],
                texts: Sequence[str]) -> list[tuple[int, int]]:
    """(chosen, rejected) sample indices for one prompt: finished answers with a </think>,
    score 1 against score 0, min(right, wrong) pairs matched by length."""
    done = [f == "stop" and "</think>" in t for f, t in zip(finish, texts, strict=True)]
    right = [i for i, (s, d) in enumerate(zip(scores, done, strict=True)) if d and s >= 1]
    wrong = [i for i, (s, d) in enumerate(zip(scores, done, strict=True)) if d and s <= 0]
    if len(right) <= len(wrong):
        return [(right[a], wrong[b]) for a, b in
                match_by_length([tokens[i] for i in right], [tokens[i] for i in wrong])]
    return [(right[b], wrong[a]) for a, b in
            match_by_length([tokens[i] for i in wrong], [tokens[i] for i in right])]


def random_pairs(scores: Sequence[float], finish: Sequence[str], texts: Sequence[str],
                 rng: random.Random) -> list[tuple[int, int]]:
    """As pair_prompt, but the min(right, wrong) pairs drawn at random: the length gap
    the matching removes is measured against this."""
    done = [f == "stop" and "</think>" in t for f, t in zip(finish, texts, strict=True)]
    right = [i for i, (s, d) in enumerate(zip(scores, done, strict=True)) if d and s >= 1]
    wrong = [i for i, (s, d) in enumerate(zip(scores, done, strict=True)) if d and s <= 0]
    rng.shuffle(right)
    rng.shuffle(wrong)
    return list(zip(right, wrong, strict=False))  # min(right, wrong) pairs


def held_out(keys: pd.DataFrame, per_source: int, seed: int) -> set[tuple[str, str]]:
    """(data_source, id) of the test prompts: per source a seeded sample of per_source
    of the given keys (all of them if fewer)."""
    picked = [g.sample(n=min(per_source, len(g)), random_state=seed) for _, g in keys.groupby("data_source")]
    return {(ds, i) for df in picked for ds, i in zip(df["data_source"], df["id"], strict=True)}


def assistant(text: str) -> list[dict]:
    reasoning, answer = split_thinking(text)
    return [{"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"}]


def load_samples(parts: Path) -> pd.DataFrame:
    """One row per prompt: stats (scores, tokens, finish, ground truth) with the texts."""
    stats = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("stats_*.parquet"))], ignore_index=True)
    texts = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("samples_*.parquet"))], ignore_index=True)
    df = stats.merge(texts, on=["data_source", "id"], how="left", validate="one_to_one")
    if df["texts"].isna().any():
        raise SystemExit(f"{int(df['texts'].isna().sum())} prompts in the stats have no samples in {parts}")
    return df


def build(samples: pd.DataFrame, prompts: pd.DataFrame, per_source: int, seed: int
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(train, test, summary) from load_samples' rows and the prompt rows synth sampled
    (data_source, prompt, extra_info.id)."""
    by_key = {(ds, e["id"]): p for ds, e, p in zip(prompts["data_source"], prompts["extra_info"],
                                                   prompts["prompt"], strict=True)}
    missing = [k for k in zip(samples["data_source"], samples["id"], strict=True) if k not in by_key]
    if missing:
        raise SystemExit(f"{len(missing)} sampled prompts are not in --prompts, e.g. {missing[:3]}")
    rescored, changed = [], 0
    for ds, texts, gt, old in zip(samples["data_source"], samples["texts"], samples["ground_truth"],
                                  samples["scores"], strict=True):
        new = [compute_score(ds, t, gt, thinking=True) for t in texts]
        changed += int(sum(abs(a - b) > 1e-9 for a, b in zip(new, old, strict=True)))  # numpy bools from parquet
        rescored.append(new)
    samples = samples.assign(scores=rescored)
    samples["pairs"] = [pair_prompt(s, f, t, x) for s, f, t, x in
                        zip(samples["scores"], samples["finish"], samples["tokens"], samples["texts"], strict=True)]
    pairable = samples[samples["pairs"].map(len) > 0]
    if pairable.empty:
        raise SystemExit("no prompt has both a finished right and a finished wrong answer")
    test_keys = held_out(pairable[["data_source", "id"]], per_source, seed)

    rows: dict[str, list[dict]] = {"train": [], "test": []}
    for r in pairable.itertuples(index=False):
        split = "test" if (r.data_source, r.id) in test_keys else "train"
        for c, w in r.pairs:
            rows[split].append({
                "data_source": r.data_source, "id": r.id, "prompt": list(by_key[(r.data_source, r.id)]),
                "chosen": assistant(r.texts[c]), "rejected": assistant(r.texts[w]),
                "chosen_index": c, "rejected_index": w,
                "chosen_tokens": int(r.tokens[c]), "rejected_tokens": int(r.tokens[w]),
                "ground_truth": r.ground_truth})
    train, test = pd.DataFrame(rows["train"]), pd.DataFrame(rows["test"])

    rng = random.Random(seed)
    rand = [(t[c], t[w]) for s, f, t, x in zip(pairable["scores"], pairable["finish"], pairable["tokens"],
                                                pairable["texts"], strict=True)
            for c, w in random_pairs(s, f, x, rng)]
    both = pd.concat([train, test], ignore_index=True)
    gap = both["rejected_tokens"] - both["chosen_tokens"]
    summary = {
        "seed": seed, "test_per_source": per_source, "prompts": len(samples), "samples": int(samples["texts"].map(len).sum()),
        "scores_changed_by_rescoring": changed,
        "finished_without_think_end": int(sum(f == "stop" and "</think>" not in t for fs, ts in
                                              zip(samples["finish"], samples["texts"], strict=True)
                                              for f, t in zip(fs, ts, strict=True))),
        "sources": {ds: {"prompts": int((samples["data_source"] == ds).sum()),
                         "pairable": int((pairable["data_source"] == ds).sum()),
                         "train_prompts": int(train.loc[train["data_source"] == ds, "id"].nunique()) if len(train) else 0,
                         "test_prompts": int(test.loc[test["data_source"] == ds, "id"].nunique()) if len(test) else 0,
                         "train_pairs": int((train["data_source"] == ds).sum()) if len(train) else 0,
                         "test_pairs": int((test["data_source"] == ds).sum()) if len(test) else 0}
                    for ds in sorted(samples["data_source"].unique())},
        "tokens": {"chosen_mean": round(float(both["chosen_tokens"].mean()), 1),
                   "rejected_mean": round(float(both["rejected_tokens"].mean()), 1),
                   "matched_gap_mean": round(float(gap.mean()), 1),
                   "matched_abs_gap_mean": round(float(gap.abs().mean()), 1),
                   "random_gap_mean": round(statistics.fmean(w - c for c, w in rand), 1),
                   "random_abs_gap_mean": round(statistics.fmean(abs(w - c) for c, w in rand), 1)},
    }
    return train, test, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, required=True, help="a b2e.data.synth <out>_parts dir")
    ap.add_argument("--prompts", nargs="+", required=True, help="the parquet files synth sampled")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-per-source", type=int, default=50, help="pairable prompts per source held out")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.parts)
    prompts = pd.concat([pd.read_parquet(f) for f in args.prompts], ignore_index=True)
    train, test, summary = build(samples, prompts, args.test_per_source, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out / "train.parquet", index=False)
    test.to_parquet(args.out / "test.parquet", index=False)
    summary.update(parts=str(args.parts), prompt_files=args.prompts)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
