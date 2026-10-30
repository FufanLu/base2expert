"""How far a trained model's weights moved from its start (two HF model dirs).

Checks that both hold the same tensors (names, shapes, dtypes), then reports
the relative change ||B - A|| / ||A|| over the whole model and per kind of
tensor (the name with its layer number taken out), and the share of elements
whose stored value changed at all. For an exported RL checkpoint it shows that
the merge picked up the trained weights (no change would mean it exported the
start), and how sparse the stored update is: an fp32 step smaller than half a
bf16 step vanishes when the weights are saved in bf16.

    python -P tools/weight_delta.py $B2E_ROOT/models/s_fin $B2E_ROOT/models/grpo_fin_step60

A model against itself must print 0 changed everywhere (the check of the tool).
"""

from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

LAYER = re.compile(r"\.layers\.\d+\.")


def tensors(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                out[k] = h.get_tensor(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("start", type=Path)
    ap.add_argument("trained", type=Path)
    ap.add_argument("--top", type=int, default=12, help="kinds of tensor to list, largest change first")
    args = ap.parse_args()
    a, b = tensors(args.start), tensors(args.trained)
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    print(f"tensors: {len(a)} / {len(b)}; only in start {len(only_a)} {only_a[:3]}; only in trained {len(only_b)} {only_b[:3]}")
    bad = [k for k in sorted(set(a) & set(b)) if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype]
    if bad:
        raise SystemExit(f"shape or dtype differs in {len(bad)} tensors, e.g. {bad[0]}: "
                         f"{tuple(a[bad[0]].shape)} {a[bad[0]].dtype} vs {tuple(b[bad[0]].shape)} {b[bad[0]].dtype}")
    kinds: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])  # |d|^2, |a|^2, changed, n
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        acc = kinds[LAYER.sub(".layers.N.", k)]
        acc[0] += float((y.float() - x.float()).pow(2).sum())  # a bf16 difference is exact in fp32
        acc[1] += float(x.float().pow(2).sum())
        acc[2] += int((x != y).sum())
        acc[3] += x.numel()
    d2, a2, ch, n = (sum(v[i] for v in kinds.values()) for i in range(4))
    dtypes = collections.Counter(str(t.dtype) for t in a.values())
    print(f"dtypes {dict(dtypes)}; elements {int(n):,}; changed {int(ch):,} ({ch / n:.2%}); "
          f"relative change {math.sqrt(d2 / a2):.3e}")
    rows = sorted(kinds.items(), key=lambda kv: -(math.sqrt(kv[1][0] / kv[1][1]) if kv[1][1] else 0.0))
    print(f"{'relative':>10}  {'changed':>8}  {'elements':>13}  kind")
    for kind, (kd2, ka2, kch, kn) in rows[: args.top]:
        rel = math.sqrt(kd2 / ka2) if ka2 else float("nan")
        print(f"{rel:10.3e}  {kch / kn:8.2%}  {int(kn):13,}  {kind}")


if __name__ == "__main__":
    main()
