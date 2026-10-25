"""Build an SFT mix (S-fin / S-law / S-med / S-mix) from a recipe.

The recipe (configs/data/sft_*.yaml) lists sources of two kinds:
  synth     b2e.data.synth output: two rows per prompt (thinking and
            non-thinking answer to the same problem). Sampled by prompt, so a
            prompt's two rows stay together (and stay out of validation
            together). Every row is scored again with the current verifier
            first and its prompt dropped if either answer fails: a verifier
            fix after the synthesis (F#32) then reaches the mix without
            another GPU run.
  smoltalk  HuggingFaceTB/smoltalk2 SFT files: `messages` plus
            `chat_template_kwargs` (enable_thinking, custom_instructions =
            the system prompt). Only the first user -> assistant exchange is
            kept, with its system prompt: Qwen3.5's chat template drops the
            reasoning of earlier assistant turns, so in a multi-turn thinking
            conversation the earlier turns would be trained in a form the
            model never sees at inference.
Every example is rendered with the student's chat template and dropped if it
exceeds --max-tokens (the SFT config fails on longer rows instead of
truncating them), or if its prompt covers >= 60% of an eval item.

    python -P -m b2e.data.sft_mix --recipe configs/data/sft_fin.yaml --data-root $B2E_ROOT/data \\
        --tokenizer $B2E_ROOT/models/Qwen3.5-4B-Base-chat --decontam-index $B2E_ROOT/data/processed/eval_items \\
        --out $B2E_ROOT/data/sft/mix_fin
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


def first_exchange(messages, kwargs) -> tuple[list[dict], bool] | None:
    """System prompt (if any) + first user and assistant turn, and the thinking flag."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return None
    think = bool((kwargs or {}).get("enable_thinking", False))
    answer = msgs[1]["content"]
    has_think = "<think>" in answer and "</think>" in answer
    if think != has_think:  # a thinking example without reasoning (or the reverse) would teach the wrong mode
        return None
    system = (kwargs or {}).get("custom_instructions") or ""
    out = ([{"role": "system", "content": system}] if system.strip() else []) + msgs[:2]
    return out, think


def load_synth(files: list[str], prompts, rng: random.Random, min_score: float = 1.0) -> tuple[list[dict], int]:
    """Rows of the sampled prompts, and the number of prompts that failed re-verification."""
    from b2e.reward.score import compute_score

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ok = [compute_score(r.data_source, r.messages[-1]["content"], r.ground_truth) >= min_score
          for r in df.itertuples(index=False)]
    failed = set(zip(df["data_source"][~pd.Series(ok)], df["id"][~pd.Series(ok)], strict=True))
    ids = sorted(set(zip(df["data_source"], df["id"], strict=True)) - failed)
    if prompts != "all":
        ids = rng.sample(ids, min(int(prompts), len(ids)))
    keep = set(ids)
    return [{"messages": list(r.messages), "enable_thinking": bool(r.enable_thinking),
             "group": f"{r.data_source}/{r.id}"}
            for r in df.itertuples(index=False) if (r.data_source, r.id) in keep], len(failed)


def load_smoltalk(files: list[str], rows: int, rng: random.Random) -> list[dict]:
    df = pd.concat([pd.read_parquet(f, columns=["messages", "chat_template_kwargs"]) for f in files],
                   ignore_index=True)
    order = list(range(len(df)))
    rng.shuffle(order)
    out = []
    for i in order:
        ex = first_exchange(df.at[i, "messages"], df.at[i, "chat_template_kwargs"])
        if ex is None:
            continue
        out.append({"messages": ex[0], "enable_thinking": ex[1], "group": f"smoltalk/{i}"})
        if len(out) >= rows * 2:  # head room for the length and decontamination filters
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--decontam-index", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    index = None
    if args.decontam_index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.decontam_index)
    rng = random.Random(args.seed)
    rows, manifest = [], {"recipe": recipe, "sources": {}}
    for src in recipe["sources"]:
        files = sorted(f for p in src["files"] for f in glob.glob(str(Path(args.data_root) / p)))
        if not files:
            raise SystemExit(f"{src['name']}: no files match {src['files']}")
        drops: Counter = Counter()
        if src["kind"] == "synth":
            cand, drops["unverified_prompts"] = load_synth(files, src.get("prompts", "all"), rng,
                                                           float(src.get("min_score", 1.0)))
            want = len(cand)
        else:
            cand = load_smoltalk(files, int(src["rows"]), rng)
            want = int(src["rows"])
        kept, lengths = [], []
        bad_groups: set[str] = set()
        for ex in cand:
            text = tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=ex["enable_thinking"])
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            prompt = "\n".join(m["content"] for m in ex["messages"] if m["role"] != "assistant")
            if n > args.max_tokens:
                drops["too_long"] += 1
                bad_groups.add(ex["group"])
            elif index is not None and index.contaminated(prompt):
                drops["eval_overlap"] += 1
                bad_groups.add(ex["group"])
            else:
                kept.append({**ex, "source": src["name"], "tokens": n})
        kept = [ex for ex in kept if ex["group"] not in bad_groups][:want]  # a synth pair goes or stays whole
        lengths = [ex["tokens"] for ex in kept]
        rows.extend(kept)
        manifest["sources"][src["name"]] = {
            "files": files, "rows": len(kept), "wanted": want, "dropped": dict(drops),
            "thinking": sum(ex["enable_thinking"] for ex in kept),
            "tokens_total": sum(lengths), "tokens_p50": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "tokens_max": max(lengths, default=0)}
        print(src["name"], manifest["sources"][src["name"]]["rows"], "rows", dict(drops), flush=True)

    groups = sorted({r["group"] for r in rows})
    rng.shuffle(groups)
    val_groups = set(groups[: int(len(groups) * recipe.get("val_fraction", 0.01))])
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df[~df["group"].isin(val_groups)].to_parquet(out / "train.parquet", index=False)
    df[df["group"].isin(val_groups)].to_parquet(out / "val.parquet", index=False)
    manifest.update({"train_rows": int((~df["group"].isin(val_groups)).sum()), "val_rows": int(df["group"].isin(val_groups).sum()),
                     "tokens_total": int(df["tokens"].sum())})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in manifest.items() if k != "recipe" and k != "sources"}))


if __name__ == "__main__":
    main()
