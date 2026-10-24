"""Do packed sequences leak into each other in Qwen3.5 under verl?

Training with `use_remove_padding=True` packs a micro-batch into one row:
document A, then document B, with position ids restarting at 0 for B and
`cu_seqlens = [0, len(A), len(A)+len(B)]` marking the boundary.
- Full-attention layers stay inside each document through flash-attn's
  varlen kernel.
- Qwen3.5's Gated DeltaNet layers carry a recurrent state and run a short
  causal conv (kernel 4) over the whole row. Both must be cut at the
  boundary, or B's first tokens see A.

This runs A and B (1) one at a time and (2) packed, through verl's own
Qwen3.5 forward patch, and compares the log-prob the model assigns to each
token of B. Each kernel mode is run both ways, so kernel numerics cancel and
only the packing differs:
  kernels   flash-linear-attention delta rule + causal-conv1d (when installed)
  no_conv   flash-linear-attention delta rule; conv via verl's per-segment fallback
  no_fla    transformers' torch delta-rule fallback, which drops cu_seqlens

    python -P tools/pack_check.py --model /tmp/b2e-models/Qwen3.5-4B-Base --out pack_check.json
"""

from __future__ import annotations

import argparse
import json

import torch

DOC_A = (
    "2025年，公司实现营业收入128.6亿元，同比增长12.4%；归属于上市公司股东的净利润为15.3亿元，"
    "同比下降3.1%，主要原因是原材料价格上涨导致毛利率由31.2%下降至28.7%。报告期内，公司经营活动产生的"
    "现金流量净额为21.9亿元，资产负债率为46.5%。董事会建议每10股派发现金红利3.2元（含税）。"
)
DOC_B = (
    "The defendant was convicted of fraud under Article 266 of the Criminal Law. The court found that he "
    "obtained 480,000 yuan from eleven victims by promising high returns on a fictitious investment fund, "
    "and sentenced him to six years of imprisonment and a fine of 100,000 yuan. On appeal, the defendant "
    "argued that the amount should be reduced because part of the money had been returned before trial."
)


def token_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[t+1] | ids[:t+1]) for each position t, in float32."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp[:-1].gather(-1, ids[1:, None]).squeeze(-1)


def set_mode(model, mode: str, originals: dict) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    for name, mod in model.named_modules():
        if not isinstance(mod, m.Qwen3_5GatedDeltaNet):
            continue
        conv, rule = originals.setdefault(name, (mod.causal_conv1d_fn, mod.chunk_gated_delta_rule))
        if mode == "kernels":
            mod.causal_conv1d_fn, mod.chunk_gated_delta_rule = conv, rule
        elif mode == "no_conv":
            mod.causal_conv1d_fn, mod.chunk_gated_delta_rule = None, rule
        elif mode == "no_fla":
            mod.causal_conv1d_fn, mod.chunk_gated_delta_rule = None, m.torch_chunk_gated_delta_rule
        else:
            raise ValueError(mode)


@torch.no_grad()
def run(model, ids_a: torch.Tensor, ids_b: torch.Tensor, pass_cu: bool) -> dict:
    """pass_cu follows verl's rule (workers/engine/fsdp/transformer_impl.py): the
    packed row's cu_seqlens reach the model only if its forward() takes them."""
    dev = ids_a.device
    la, lb = len(ids_a), len(ids_b)

    def fwd(ids, pos, cu):
        kw = {} if cu is None or not pass_cu else {"cu_seqlens": cu, "cu_seqlens_cpu": cu.cpu()}
        # use_cache=False as in training: verl 0.9.1's patched Gated DeltaNet forward
        # still reads the pre-5.x cache API (cache_params.conv_states) and fails on
        # transformers 5.9's DynamicCache (docs/failures.md F#7).
        out = model(input_ids=ids[None], attention_mask=None, position_ids=pos[None], use_cache=False, **kw)
        return out.logits[0]

    alone_b = token_logprobs(fwd(ids_b, torch.arange(lb, device=dev), None), ids_b)
    alone_a = token_logprobs(fwd(ids_a, torch.arange(la, device=dev), None), ids_a)
    packed_ids = torch.cat([ids_a, ids_b])
    packed_pos = torch.cat([torch.arange(la, device=dev), torch.arange(lb, device=dev)])
    cu = torch.tensor([0, la, la + lb], device=dev, dtype=torch.long)
    packed = fwd(packed_ids, packed_pos, cu)
    packed_a = token_logprobs(packed[:la], ids_a)
    packed_b = token_logprobs(packed[la:], ids_b)
    da = (packed_a - alone_a).abs()
    db = (packed_b - alone_b).abs()
    return {
        "len_a": la, "len_b": lb,
        "doc_a_max_abs_diff": float(da.max()),
        "doc_b_max_abs_diff": float(db.max()),
        "doc_b_mean_abs_diff": float(db.mean()),
        "doc_b_first8_abs_diff": [round(float(x), 4) for x in db[:8]],
        "doc_b_last8_abs_diff": [round(float(x), 4) for x in db[-8:]],
        "doc_b_mean_logprob_alone": float(alone_b.mean()),
        "doc_b_mean_logprob_packed": float(packed_b.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="kernels,no_conv,no_fla")
    # cond: Qwen3_5ForConditionalGeneration, the class the checkpoint declares and
    # verl loads; causal: the text-only Qwen3_5ForCausalLM (what a text-only export
    # would be loaded as).
    ap.add_argument("--classes", default="cond,causal")
    ap.add_argument("--out", default="pack_check.json")
    args = ap.parse_args()

    import inspect

    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    tok = AutoTokenizer.from_pretrained(args.model)
    ids_a = torch.tensor(tok(DOC_A)["input_ids"], device="cuda")
    ids_b = torch.tensor(tok(DOC_B)["input_ids"], device="cuda")
    report = {
        "model": args.model,
        "flash_linear_attention": is_flash_linear_attention_available(),
        "causal_conv1d": is_causal_conv1d_available(),
        "classes": {},
    }
    classes = {"cond": Qwen3_5ForConditionalGeneration, "causal": Qwen3_5ForCausalLM}
    for cname in args.classes.split(","):
        model = classes[cname].from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
        apply_monkey_patch(model, use_remove_padding=True)
        pass_cu = "cu_seqlens" in inspect.signature(model.forward).parameters
        entry = report["classes"][cname] = {"class": classes[cname].__name__, "verl_passes_cu_seqlens": pass_cu,
                                            "modes": {}}
        originals: dict = {}
        for mode in args.modes.split(","):
            set_mode(model, mode, originals)
            r = entry["modes"][mode] = run(model, ids_a, ids_b, pass_cu)
            print(f"{cname:<6} {mode:<8} cu_seqlens passed={pass_cu}  doc B max|Δ| {r['doc_b_max_abs_diff']:.4f}  "
                  f"mean|Δ| {r['doc_b_mean_abs_diff']:.4f}  first 8 {r['doc_b_first8_abs_diff']}  "
                  f"(doc A max|Δ| {r['doc_a_max_abs_diff']:.4f})", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "classes"}))


if __name__ == "__main__":
    main()
