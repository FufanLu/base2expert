"""Are the chat and thinking special tokens trained in a base checkpoint?

In sql-agent-rl, Qwen2.5-Coder's `<tool_call>` embedding turned out to be
untrained (all zeros in 7B/32B, a placeholder row in 3B). Every SFT target
started with that token, the model could never make it likely, and the
first answer position collapsed (sql-agent-rl docs/failures.md #29).
Qwen3.5-4B-Base's model card says its control tokens were trained for the
official chat template. This tool checks that for every token the SFT data
will use, from the weights alone (no model load, CPU only):

- the row's L2 norm against the spread of ordinary-token norms;
- whether the row equals another special token's row (a copied placeholder);
- the largest cosine similarity to any other special-token row.

    python -P tools/token_norms.py [--tokenizer DIR] <model dir> [<model dir> ...]

--tokenizer picks the tokenizer that maps token strings to ids (default: each
model's own). Qwen3.5's Base tokenizer does not register <think>, </think>,
<tool_response> or </tool_response> at all (the post-trained Qwen3.5-4B's
does, at ids 248066-248069), so a Base model's rows for them are found with
the post-trained tokenizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

TOKENS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
]


def embedding(d: Path) -> tuple[str, torch.Tensor]:
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                # Qwen3.5 checkpoints nest the text model: model.language_model.embed_tokens
                if k.endswith("embed_tokens.weight") and "mtp" not in k and "visual" not in k:
                    return k, h.get_tensor(k).float()
    raise SystemExit(f"no embed_tokens in {d}")


def main(dirs: list[str], tokenizer: str | None) -> None:
    for arg in dirs:
        d = Path(arg)
        tok = AutoTokenizer.from_pretrained(tokenizer or d)
        key, emb = embedding(d)
        norms = emb.norm(dim=-1)
        n_regular = len(tok) - len(tok.get_added_vocab())
        regular = norms[:n_regular]
        q = torch.quantile(regular, torch.tensor([0.01, 0.5, 0.99]))
        print(f"== {d.name} (ids from {Path(tokenizer).name if tokenizer else 'its own tokenizer'}): {key} {tuple(emb.shape)}; "
              f"ordinary-token norm p1/p50/p99 = "
              f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
        ids = {}
        for t in TOKENS:
            i = tok.convert_tokens_to_ids(t)
            if i is None or i == tok.unk_token_id:
                print(f"   {t:<18} not a single token in this vocabulary")
                continue
            ids[t] = i
        rows = emb[list(ids.values())]
        unit = torch.nn.functional.normalize(rows, dim=-1)
        cos = unit @ unit.T
        cos.fill_diagonal_(-1.0)
        for j, (t, i) in enumerate(ids.items()):
            same = [u for k, u in enumerate(ids) if k != j and torch.equal(rows[k], rows[j])]
            k = int(cos[j].argmax())
            flag = "  ZERO ROW" if float(norms[i]) == 0.0 else ""
            flag += f"  SAME ROW AS {same}" if same else ""
            print(f"   {t:<18} id {i:<7} norm {float(norms[i]):.4f}   "
                  f"max cos to another special token {float(cos[j, k]):+.3f} ({list(ids)[k]}){flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    main(a.models, a.tokenizer)
