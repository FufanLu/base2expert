"""Render SFT examples through verl's own MultiTurnSFTDataset and show the
tokens and their loss mask, so we see exactly what the student is trained on.

Checks, for a thinking and a non-thinking example:
  - <think> / </think> are the single special tokens 248068 / 248069, not text
  - the assistant turn's <|im_end|> is a target (the model must learn to stop)
  - prompt tokens are masked out
  - the non-thinking target matches what the chat template produces at
    inference with enable_thinking=False

    python -P tools/check_sft_render.py --model /tmp/b2e-models/Qwen3.5-4B-Base-chat
"""

from __future__ import annotations

import argparse
import tempfile

import pandas as pd

Q = [{"role": "user", "content": "What is 2 + 3? Give the answer in \\boxed{}."}]
EXAMPLES = [
    {"messages": Q + [{"role": "assistant", "content": "<think>\n2 plus 3 is 5.\n</think>\n\nThe answer is \\boxed{5}."}],
     "enable_thinking": True},
    {"messages": Q + [{"role": "assistant", "content": "The answer is \\boxed{5}."}], "enable_thinking": False},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    tok = AutoTokenizer.from_pretrained(a.model)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/sft.parquet"
        pd.DataFrame(EXAMPLES).to_parquet(path)
        cfg = OmegaConf.create({"pad_mode": "no_padding", "max_length": 512, "truncation": "error",
                                "messages_key": "messages", "enable_thinking_key": "enable_thinking"})
        ds = MultiTurnSFTDataset(path, tok, cfg)
        for i in range(len(ds)):
            item = ds[i]
            ids, mask = item["input_ids"].tolist(), item["loss_mask"].tolist()
            print(f"=== example {i} (enable_thinking={EXAMPLES[i]['enable_thinking']}), {len(ids)} tokens")
            print("full text:", repr(tok.decode(ids)))
            print("target   :", repr(tok.decode([t for t, m in zip(ids, mask, strict=True) if m])))
            print("special ids present:", {t: ids.count(t) for t in (248068, 248069, 248045, 248046)})
            im_end_targets = [m for t, m in zip(ids, mask, strict=True) if t == 248046]
            print("<|im_end|> loss mask values:", im_end_targets)
        gen = tok.apply_chat_template(Q, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        print("=== inference prompt with enable_thinking=False:", repr(gen))
        gen_t = tok.apply_chat_template(Q, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        print("=== inference prompt with enable_thinking=True :", repr(gen_t))


if __name__ == "__main__":
    main()
