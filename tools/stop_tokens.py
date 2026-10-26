"""Which tokens end a generation in vLLM for a model directory, and does the model write them.

run_eval stops where vLLM's defaults stop: the tokenizer's eos plus whatever
eos ids vLLM takes from the model's generation config (a generation_config.json,
or one derived from config.json). Base-chat and M0 declare different ones
(M0's config.json, re-saved by verl, has eos_token_id 248046 = <|im_end|> at the
top level; Base-chat's has 248044 = <|endoftext|> only inside text_config), so
the two may stop on different tokens. This prints the stop ids vLLM uses and
generates answers with special tokens kept in the text, counting the answers
that write <|endoftext|> or <|im_end|> and keep going.

    python -P tools/stop_tokens.py --model /tmp/b2e-models/m0 \\
        --prompts $B2E_ROOT/data/processed/if/ifeval_test.parquet --n 200 --out results/m0/stop_tokens_m0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ENDS = ("<|endoftext|>", "<|im_end|>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=8192, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    try:
        gen_cfg = llm.llm_engine.vllm_config.model_config.try_get_generation_config()
    except Exception as e:  # noqa: BLE001 - an internal API; the generations below still answer the question
        gen_cfg = {"error": repr(e)}
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=False)
    info = {"model": args.model, "tokenizer_eos": [tok.eos_token, tok.eos_token_id],
            "generation_config": {k: v for k, v in gen_cfg.items() if not k.startswith("_")},
            "default_sampling_params": str(llm.get_default_sampling_params())}
    msgs = [list(p) for p in pd.read_parquet(args.prompts).head(args.n)["prompt"]]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    per = []
    for o in outs:
        c = o.outputs[0]
        text = c.text
        first = {e: text.find(e) for e in ENDS if e in text}
        per.append({"finish": c.finish_reason, "stop_reason": c.stop_reason, "last_id": c.token_ids[-1] if c.token_ids else None,
                    "tokens": len(c.token_ids), "ends_written": first,
                    "chars_after_first_end": len(text) - min(first.values()) if first else 0})
    summary = {
        "n": len(per),
        "finish": {f: sum(p["finish"] == f for p in per) for f in {p["finish"] for p in per}},
        "last_token": {str(i): sum(p["last_id"] == i for p in per) for i in {p["last_id"] for p in per}},
        **{f"writes_{e}": sum(e in p["ends_written"] for p in per) for e in ENDS},
        "keeps_going_after_an_end": sum(p["chars_after_first_end"] > len(max(ENDS, key=len)) for p in per),
    }
    result = {**info, "summary": summary, "samples": per[:20]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({**info, "summary": summary}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
