"""Render P5's preference pairs as TRL 0.27's DPOTrainer will, and check them
against the chat template before any DPO run: the preference-data counterpart of
tools/check_sft_render.py (handout 4.7).

DPOTrainer prepares a row in three steps (trl/trainer/dpo_trainer.py,
_prepare_dataset): data_utils.apply_chat_template renders prompt + generation
prompt and prompt + chosen (and + rejected), and keeps their common prefix as the
prompt without saying so; tokenize_row tokenizes prompt and completion apart
(add_special_tokens=False), appends eos_token_id to each completion, and keeps the
last max_prompt_length prompt tokens. Over every pair, both sides:

  prefix        the generation-prompt rendering is a prefix of prompt + answer, so
                the common prefix is the whole prompt (else the boundary moves)
  boundary      prompt and completion tokenized apart = the conversation tokenized
                whole
  ending        the last tokens of each completion once TRL appends eos (the
                template closes the turn with <|im_end|> and a newline; for S-fin's
                tokenizer eos is <|im_end|> again)
  pre-rendered  the alternative: hand TRL strings, the prompt rendering and the
                answer without the template's turn end, so the one eos TRL appends
                closes it; prompt + completion + eos must equal the conversation
                tokenized up to and including <|im_end|>, the sequence the model
                generated
  lengths       prompt and completion tokens (for max_prompt_length and
                max_length), and the completion against the tokens the model
                generated (chosen_tokens / rejected_tokens)

Exit 1 if a prefix, boundary or pre-rendered check fails on any pair.

    python -P tools/check_pref_render.py --model $B/models/s_fin \\
        --data $B/data/processed/pref_fin/train.parquet $B/data/processed/pref_fin/test.parquet
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter

import pandas as pd

END = "<|im_end|>"


def first_diff(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a model dir with the tokenizer and chat template")
    ap.add_argument("--data", nargs="+", required=True, help="b2e.data.pref_pairs parquet files")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from trl.data_utils import apply_chat_template

    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"enable_thinking": True}
    eos = tok.eos_token_id
    print(f"eos {tok.eos_token!r} = {eos}; bos {tok.bos_token!r}")

    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    df = pd.concat([pd.read_parquet(f) for f in a.data], ignore_index=True)
    fails: Counter = Counter()
    shown: set[str] = set()
    endings: Counter = Counter()
    prompt_len, comp_len, totals, gen_diff = [], [], [], []
    for r in df.itertuples(index=False):
        prompt = [dict(m) for m in r.prompt]
        sides = {"chosen": [dict(m) for m in r.chosen], "rejected": [dict(m) for m in r.rejected]}
        out = apply_chat_template({"prompt": prompt, **sides, "chat_template_kwargs": kw}, tok)
        gen = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kw)
        plen = len(ids(gen))
        prompt_len.append(plen)
        for side, msgs in sides.items():
            full = tok.apply_chat_template(prompt + msgs, tokenize=False, **kw)
            checks = {"prefix": full.startswith(gen) and out["prompt"] == gen}
            whole, apart = ids(full), ids(out["prompt"]) + ids(out[side])
            checks["boundary"] = whole == apart
            endings[tuple(tok.convert_ids_to_tokens((ids(out[side]) + [eos])[-3:]))] += 1
            end = full.rfind(END)
            completion = full[len(gen):end]
            pre = ids(gen) + ids(completion) + [eos]
            want = ids(full[:end + len(END)])
            checks["pre-rendered"] = pre == want
            n = len(ids(completion)) + 1
            comp_len.append(n)
            totals.append(plen + n)
            gen_diff.append(n - int(getattr(r, f"{side}_tokens")))
            for name, ok in checks.items():
                if ok:
                    continue
                fails[name] += 1
                if name not in shown:
                    shown.add(name)
                    x, y = (whole, apart) if name == "boundary" else (want, pre)
                    i = first_diff(x, y)
                    print(f"--- first {name} failure: {r.data_source} {r.id} {side}")
                    if name == "prefix":
                        print("generation prompt ends:", repr(gen[-80:]))
                        print("conversation there    :", repr(full[len(gen) - 80:len(gen) + 40]))
                    else:
                        print("expected:", tok.convert_ids_to_tokens(x[max(0, i - 3):i + 5]))
                        print("got     :", tok.convert_ids_to_tokens(y[max(0, i - 3):i + 5]))

    pairs = len(df)
    print(f"pairs {pairs}, completions {2 * pairs}")
    for name in ("prefix", "boundary", "pre-rendered"):
        print(f"{name:13} failures {fails[name]}")
    print("TRL's default completion ending (last 3 tokens):", dict(endings))
    q = statistics.quantiles
    print(f"prompt tokens: max {max(prompt_len)}, p99 {q(prompt_len, n=100, method='inclusive')[98]:.0f}, mean {statistics.fmean(prompt_len):.0f}")
    print(f"completion tokens with eos (pre-rendered): max {max(comp_len)}, p99 {q(comp_len, n=100, method='inclusive')[98]:.0f}")
    print(f"prompt + completion max {max(totals)}")
    print(f"completion minus generated tokens: mean {statistics.fmean(gen_diff):.2f}, min {min(gen_diff)}, "
          f"max {max(gen_diff)}, equal in {sum(d == 0 for d in gen_diff) / len(gen_diff):.1%}")
    sys.exit(1 if fails["prefix"] or fails["boundary"] or fails["pre-rendered"] else 0)


if __name__ == "__main__":
    main()
