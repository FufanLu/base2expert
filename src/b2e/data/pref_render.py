"""Render preference pairs into the strings TRL trains on, through the model's chat template.

TRL 0.27's DPOTrainer renders a conversational row itself and then appends eos to
every completion. With Qwen3.5's template, whose turn end is "<|im_end|>\\n" and
whose eos is <|im_end|>, every answer then ends "<|im_end|>\\n<|im_end|>" (all
3,770 completions of the P5 pairs, 11556254): two tokens the model never
generates, since it stops at the first <|im_end|>, inside every log-probability.

Here instead: prompt = the generation-prompt rendering (thinking on, as the
samples were generated), completion = the answer up to the template's turn end,
so the one eos TRL appends closes the turn. Given strings, TRL skips its own
rendering. Every row is checked to tokenize, prompt + completion + eos, to the
conversation through <|im_end|>, the check tools/check_pref_render.py ran on the
cluster; a row that does not is an error, not a skipped row.

    python -P -m b2e.data.pref_render --model $B/models/s_fin \\
        --data $B/data/processed/pref_fin/train.parquet $B/data/processed/pref_fin/test.parquet
writes <stem>_rendered.parquet beside each input: prompt, chosen, rejected as
strings, plus the input's metadata columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

END = "<|im_end|>"
THINKING = {"enable_thinking": True}


def render(tok, prompt: list[dict], answer: list[dict]) -> tuple[str, str]:
    """(prompt string, completion string) for one answer to one prompt."""
    gen = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **THINKING)
    full = tok.apply_chat_template(prompt + answer, tokenize=False, **THINKING)
    if not full.startswith(gen):
        raise ValueError(f"the generation prompt does not prefix the conversation: {gen[-60:]!r}")
    end = full.rfind(END)
    if end < len(gen):
        raise ValueError(f"no {END} after the prompt")
    return gen, full[len(gen):end]


def matches(tok, prompt: str, completion: str, conversation: str) -> bool:
    """prompt + completion tokenized apart, plus eos, = the conversation through <|im_end|>."""
    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]
    end = conversation.rfind(END)
    return ids(prompt) + ids(completion) + [tok.eos_token_id] == ids(conversation[:end + len(END)])


def render_rows(tok, df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in df.to_dict("records"):
        prompt = [dict(m) for m in r["prompt"]]
        row = {k: v for k, v in r.items() if k not in ("prompt", "chosen", "rejected")}
        for side in ("chosen", "rejected"):
            answer = [dict(m) for m in r[side]]
            p, c = render(tok, prompt, answer)
            if not matches(tok, p, c, tok.apply_chat_template(prompt + answer, tokenize=False, **THINKING)):
                raise ValueError(f"{r['data_source']} {r['id']} {side}: rendered tokens differ from the conversation")
            row["prompt"], row[side] = p, c
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir with the tokenizer and chat template")
    ap.add_argument("--data", nargs="+", required=True, help="b2e.data.pref_pairs parquet files")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.eos_token != END:
        raise SystemExit(f"eos is {tok.eos_token!r}, not {END}: the completion would not be closed by it")
    for f in map(Path, a.data):
        df = render_rows(tok, pd.read_parquet(f))
        dst = f.with_name(f.stem + "_rendered.parquet")
        df.to_parquet(dst, index=False)
        print(json.dumps({"input": str(f), "output": str(dst), "pairs": len(df), "model": a.model}))


if __name__ == "__main__":
    main()
