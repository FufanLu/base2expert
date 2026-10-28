"""Weight interpolation between a base model and its fine-tune (WiSE-FT).

Every tensor becomes base + alpha * (fine-tune - base), computed in fp32 and
stored in the fine-tune's dtype, with the fine-tune's config and tokenizer
files. alpha = 1 is the fine-tune, alpha = 0 the base. Why: the finance SFT
left the general scores 1-5 points below Base-chat (handout 4.12g), and moving
part of the way back to the base is the cheapest trade of some domain gain
for general ability (no training, one eval per alpha).

Tensors only the base has are left out: Qwen3.5's MTP head, which verl's
merger drops and the fine-tune's config declares absent (mtp_num_hidden_layers
0). A tensor only the fine-tune has, or one whose shape differs, is an error.

    python -P tools/interpolate.py --base DIR --ft DIR --check
    python -P tools/interpolate.py --base DIR --ft DIR --alpha 0.5 --out DIR

--check rebuilds alpha 0 and alpha 1 in memory and compares them with the
base (rounded to the fine-tune's dtype) and the fine-tune: both must match
exactly, or the tensor names are paired wrong.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensors(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                out[k] = h.get_tensor(k)
    return out


def interpolate(base: dict[str, torch.Tensor], ft: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    extra = sorted(set(ft) - set(base))
    if extra:
        raise SystemExit(f"{len(extra)} tensors only in the fine-tune: {extra[:5]}")
    out = {}
    for k, f in ft.items():
        b = base[k]
        if b.shape != f.shape:
            raise SystemExit(f"{k}: shape {tuple(b.shape)} in the base, {tuple(f.shape)} in the fine-tune")
        # lerp, not b + alpha * (f - b): that misses f at alpha = 1 by fp32
        # rounding where a weight moved far (11437914: 2.3e-10); lerp computes
        # from the end it is nearer, so both ends come back exactly.
        out[k] = torch.lerp(b.float(), f.float(), alpha).to(f.dtype)
    return out


def check(base: dict[str, torch.Tensor], ft: dict[str, torch.Tensor]) -> None:
    for alpha, ref in ((0.0, {k: base[k].to(v.dtype) for k, v in ft.items()}), (1.0, ft)):
        got = interpolate(base, ft, alpha)
        diffs = {k: float((got[k].float() - ref[k].float()).abs().max()) for k in ref}
        bad = sorted((k for k in diffs if diffs[k] != 0), key=diffs.get, reverse=True)
        print(f"alpha {alpha}: {len(got)} tensors, {len(bad)} differ, max |diff| {max(diffs.values()):.3g} {bad[:3]}")
        if bad:
            raise SystemExit("not exact")
        del got
    print(f"left out (only in the base): {len(set(base) - set(ft))} {sorted(set(base) - set(ft))[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--ft", type=Path, required=True)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    base, ft = tensors(args.base), tensors(args.ft)
    if args.check:
        check(base, ft)
        return
    if args.alpha is None or args.out is None:
        ap.error("--alpha and --out, or --check")
    part = args.out.with_name(args.out.name + ".partial")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    save_file(interpolate(base, ft, args.alpha), str(part / "model.safetensors"), metadata={"format": "pt"})
    # Config, tokenizer and processor files are the fine-tune's (its config has no MTP layer).
    for f in args.ft.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, part / f.name)
    (part / "PROVENANCE.txt").write_text(
        f"tools/interpolate.py: base + {args.alpha} * (ft - base)\nbase {args.base.resolve()}\nft {args.ft.resolve()}\n"
    )
    shutil.rmtree(args.out, ignore_errors=True)
    part.rename(args.out)
    (args.out / ".complete").touch()
    print(f"wrote {args.out} ({len(base)} base tensors, {len(ft)} written)")


if __name__ == "__main__":
    main()
