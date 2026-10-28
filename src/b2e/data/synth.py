"""SFT data by rejection sampling from the teacher (Qwen3.6-35B-A3B).

For each training prompt (verl RL format) the teacher samples k answers in
thinking mode. Each answer is scored by the same verifier the RL reward
uses (b2e.reward.score), and only verified answers are kept, the shortest
one per prompt (long-winded reasoning is not what the student should copy).
Every kept example is written twice:

  thinking:     assistant = "<think>\\n{reasoning}\\n</think>\\n\\n{answer}", enable_thinking=True
  non-thinking: assistant = "{answer}", enable_thinking=False

so the student learns both modes from the same problems. The per-prompt pass
rate (correct / k) is saved as well: it measures difficulty for the teacher
and is used later to filter the RL prompt pool. Next to it go each sample's
boxed answer, score and finish reason, so prompts the teacher never solves
can be audited for a wrong gold answer (19% of the FinQA pilot, 11351439).
Every sample's full text goes to <stem>_parts/samples_*.parquet, so a
verifier fix is applied by scoring again on a CPU, not by sampling again
(F#32 had to be sampled again: the IF run kept only the passing texts).
--only-ids restricts the run to listed prompts, e.g. the ones a first pass
never solved.

    python -P -m b2e.data.synth --teacher /tmp/b2e-models/Qwen3.6-35B-A3B --k 4 \\
        --per-source 2000 --out sft_fin.parquet finqa_train.parquet tatqa_train.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from b2e.prompts import fits
from b2e.reward.numeric import extract_boxed
from b2e.reward.score import compute_score

MAX_MODEL_LEN = 16384


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer). The teacher's text starts inside the <think> block
    that the chat template opened, so the reasoning is everything before </think>."""
    if "</think>" not in text:
        return "", text.strip()
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "").strip(), answer.strip()


def score_prompt(row, outputs, min_score: float) -> tuple[dict, list[dict]]:
    """Pass-rate record and SFT rows (thinking + non-thinking, or none) for one prompt.

    outputs: (text, token count, finish reason) per sample. Only answers that
    finished on their own (not cut at the token limit) are kept for SFT.
    """
    gt = row.reward_model["ground_truth"]
    # generated with enable_thinking=True: an answer without </think> was cut off (F#53)
    scored = [(compute_score(row.data_source, text, gt, thinking=True), n, text, fin) for text, n, fin in outputs]
    stat = {"data_source": row.data_source, "id": row.extra_info["id"],
            "pass_rate": sum(x[0] >= min_score for x in scored) / len(scored),
            "mean_score": sum(x[0] for x in scored) / len(scored),
            "ground_truth": gt, "answers": [extract_boxed(x[2]) for x in scored],
            "scores": [x[0] for x in scored], "tokens": [x[1] for x in scored], "finish": [x[3] for x in scored]}
    good = [x for x in scored if x[0] >= min_score and x[3] == "stop"]
    if not good:
        return stat, []
    _, _, text, _ = min(good, key=lambda x: x[1])
    reasoning, answer = split_thinking(text)
    user = list(row.prompt)
    base = {"data_source": row.data_source, "id": row.extra_info["id"], "ground_truth": gt}
    return stat, [
        {**base, "enable_thinking": True,
         "messages": user + [{"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"}]},
        {**base, "enable_thinking": False, "messages": user + [{"role": "assistant", "content": answer}]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--per-source", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="1.0 with --temperature 1.0 reproduces RL rollouts, for pass rates that filter the RL pool")
    ap.add_argument("--min-score", type=float, default=1.0, help="graded rewards (CAIL term) count from here")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="prompts per generation call; each chunk is written as it finishes, and a rerun "
                         "skips chunks already on disk, so a job that hits its time limit loses one chunk at most")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-ids", default="", help="text file, one prompt id per line: run on these prompts only")
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    only = set(Path(args.only_ids).read_text().split("\n")) - {""} if args.only_ids else None
    frames = []
    for path in args.inputs:
        df = pd.read_parquet(path)
        if only is not None:
            df = df[df["extra_info"].apply(lambda e: e["id"] in only)]
        for _, g in df.groupby("data_source"):
            frames.append(g.sample(n=min(args.per_source, len(g)), random_state=args.seed))
    df = pd.concat(frames, ignore_index=True)
    key = pd.DataFrame({"s": df["data_source"], "i": df["extra_info"].apply(lambda e: e["id"])})
    if key.duplicated().any():  # pass rates, SFT pairs and --only-ids all identify a prompt by (source, id)
        raise SystemExit(f"duplicate prompt ids: {key[key.duplicated()].head().values.tolist()}")
    out = Path(args.out)
    parts = out.with_name(out.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)

    llm = sp = None
    for c0 in range(0, len(df), args.chunk):
        tag = f"{c0 // args.chunk:04d}"
        if (parts / f"stats_{tag}.parquet").exists():
            continue
        if llm is None:
            from vllm import LLM, SamplingParams

            llm = LLM(model=args.teacher, gpu_memory_utilization=0.9, max_model_len=MAX_MODEL_LEN,
                      enable_prefix_caching=True)
            sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                seed=args.seed)
        chunk = df.iloc[c0:c0 + args.chunk]
        # drop prompts too long for the context inside the chunk, so chunk boundaries (and the
        # parts already on disk) stay the same when a run resumes
        tok = llm.get_tokenizer()
        ok = [fits(tok, list(p), True, args.max_tokens, MAX_MODEL_LEN) for p in chunk["prompt"]]
        (parts / f"skipped_{tag}.txt").write_text(str(len(ok) - sum(ok)))
        chunk = chunk[ok]
        t0 = time.time()
        outs = llm.chat([list(p) for p in chunk["prompt"]], sp, chat_template_kwargs={"enable_thinking": True})
        dt = time.time() - t0
        sft, stats, samples = [], [], []
        for row, o in zip(chunk.itertuples(index=False), outs, strict=True):
            stat, rows = score_prompt(row, [(c.text, len(c.token_ids), c.finish_reason) for c in o.outputs],
                                      args.min_score)
            stats.append(stat)
            sft.extend(rows)
            samples.append({"data_source": row.data_source, "id": row.extra_info["id"],
                            "texts": [c.text for c in o.outputs]})
        pd.DataFrame(samples).to_parquet(parts / f"samples_{tag}.parquet", index=False)
        pd.DataFrame(sft).to_parquet(parts / f"sft_{tag}.parquet", index=False)
        st = pd.DataFrame(stats)
        st["seconds"] = dt / len(st)  # the chunk's generation time spread over its prompts
        st.to_parquet(parts / f"stats_{tag}.parquet", index=False)  # written last: marks the chunk done
        print(f"chunk {tag}: {len(chunk)} prompts, {len(sft) // 2} kept, {dt:.0f}s", flush=True)

    sft_df = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("sft_*.parquet"))], ignore_index=True)
    st = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("stats_*.parquet"))], ignore_index=True)
    sft_df.to_parquet(out, index=False)
    st.to_parquet(out.with_name(out.stem + "_passrate.parquet"), index=False)
    gen_tokens = int(st["tokens"].apply(sum).sum())
    seconds = float(st["seconds"].sum())
    summary = {
        "prompts": len(st), "k": args.k, "kept_prompts": len(sft_df) // 2, "seconds": round(seconds, 1),
        "generated_tokens": gen_tokens, "tokens_per_s": round(gen_tokens / seconds, 1),
        "truncated_samples": int(st["finish"].apply(lambda f: sum(x == "length" for x in f)).sum()),
        "skipped_too_long": sum(int(f.read_text()) for f in parts.glob("skipped_*.txt")),
        "pass_rate_by_source": st.groupby("data_source")["pass_rate"].mean().round(3).to_dict(),
        "never_solved_by_source": st[st["pass_rate"] == 0].groupby("data_source").size().to_dict(),
        "kept_by_source": sft_df[sft_df["enable_thinking"]].groupby("data_source").size().to_dict(),
    }
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
