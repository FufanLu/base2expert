"""Render SFT examples through the dataset class the SFT trainer uses
(b2e.train.sft_dataset.SystemPromptSFTDataset, verl's MultiTurnSFTDataset
underneath) and show the tokens and their loss mask, so we see exactly what the
student is trained on.

Tokenizer and processor are loaded as the trainer loads them (verl's
hf_tokenizer / hf_processor on the model directory): Qwen3.5 ships a processor,
and with it verl renders message contents as lists. The first version of this
check (11348093) used the tokenizer only and had no system prompt, and the
SFT smoke (11378546) then failed on the first system-prompt row.

Checks, for a thinking, a non-thinking and a system-prompt example:
  - <think> / </think> are the single special tokens 248068 / 248069, not text
  - the assistant turn's <|im_end|> is a target (the model must learn to stop)
  - prompt and system tokens are masked out
  - the non-thinking target matches what the chat template produces at
    inference with enable_thinking=False

With --data, every row of an SFT mix goes through the dataset as the trainer
would draw it; failures are counted per source and the exit code is 1 if any
row fails.

    python -P tools/check_sft_render.py --model /tmp/b2e-models/Qwen3.5-4B-Base-chat [--data mix/train.parquet]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter

import pandas as pd

Q = [{"role": "user", "content": "What is 2 + 3? Give the answer in \\boxed{}."}]
SYS = [{"role": "system", "content": "You are a terse math tutor. Answer in one sentence."}]
EXAMPLES = [
    {"messages": Q + [{"role": "assistant", "content": "<think>\n2 plus 3 is 5.\n</think>\n\nThe answer is \\boxed{5}."}],
     "enable_thinking": True},
    {"messages": Q + [{"role": "assistant", "content": "The answer is \\boxed{5}."}], "enable_thinking": False},
    {"messages": SYS + Q + [{"role": "assistant", "content": "The answer is \\boxed{5}."}], "enable_thinking": False},
]
CFG = {"pad_mode": "no_padding", "max_length": 12288, "truncation": "error",
       "messages_key": "messages", "enable_thinking_key": "enable_thinking"}


def show(tok, ds, i: int, label: str) -> None:
    item = ds[i]
    ids, mask = item["input_ids"].tolist(), item["loss_mask"].tolist()
    print(f"=== {label}, {len(ids)} tokens")
    print("full text:", repr(tok.decode(ids)))
    print("target   :", repr(tok.decode([t for t, m in zip(ids, mask, strict=True) if m])))
    print("special ids present:", {t: ids.count(t) for t in (248068, 248069, 248045, 248046)})
    print("<|im_end|> loss mask values:", [m for t, m in zip(ids, mask, strict=True) if t == 248046])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="", help="an SFT mix parquet: run every row through the dataset")
    a = ap.parse_args()
    from omegaconf import OmegaConf
    from verl.utils import hf_processor, hf_tokenizer

    from b2e.train.sft_dataset import SystemPromptSFTDataset

    tok, proc = hf_tokenizer(a.model), hf_processor(a.model)
    print("processor:", type(proc).__name__ if proc is not None else None)
    cfg = OmegaConf.create(CFG)
    if a.data:
        df = pd.read_parquet(a.data)
        ds = SystemPromptSFTDataset(a.data, tok, cfg, processor=proc)
        bad: Counter = Counter()
        first_err: dict[str, str] = {}
        shown: set[str] = set()
        for i in range(len(ds)):
            src = df["source"].iat[i]
            try:
                item = ds[i]
            except Exception as e:  # noqa: BLE001 - count every failure the trainer would hit, by source
                bad[src] += 1
                first_err.setdefault(src, f"row {i}: {type(e).__name__}: {str(e)[:300]}")
                continue
            if src not in shown:  # one example per source: where the target starts
                shown.add(src)
                ids, mask = item["input_ids"].tolist(), item["loss_mask"].tolist()
                start = mask.index(1)
                print(f"--- {src} row {i}: system={df['messages'].iat[i][0]['role'] == 'system'}, "
                      f"thinking={bool(df['enable_thinking'].iat[i])}, {len(ids)} tokens, "
                      f"prompt ends {tok.decode(ids[max(0, start - 12):start])!r}, "
                      f"target starts {tok.decode(ids[start:start + 8])!r}, "
                      f"target ends {tok.decode([t for t, m in zip(ids, mask, strict=True) if m][-4:])!r}")
        print("rows per source:", dict(Counter(df["source"])))
        print("failed per source:", dict(bad))
        for src, err in first_err.items():
            print(f"first failure in {src}: {err}")
        sys.exit(1 if bad else 0)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/sft.parquet"
        pd.DataFrame(EXAMPLES).to_parquet(path)
        ds = SystemPromptSFTDataset(path, tok, cfg, processor=proc)
        for i, ex in enumerate(EXAMPLES):
            show(tok, ds, i, f"example {i} (enable_thinking={ex['enable_thinking']}, "
                             f"system={ex['messages'][0]['role'] == 'system'})")
        gen = tok.apply_chat_template(Q, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        print("=== inference prompt with enable_thinking=False:", repr(gen))
        gen_t = tok.apply_chat_template(Q, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        print("=== inference prompt with enable_thinking=True :", repr(gen_t))


if __name__ == "__main__":
    main()
