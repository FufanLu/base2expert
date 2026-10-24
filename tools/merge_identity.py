"""mergekit on Qwen3.5: merging a model with itself must give the same weights.

Runs a linear merge (0.5 / 0.5) and a TIES merge of a model with itself, then
compares every tensor of the result with the original. Any difference means
mergekit mishandles the architecture (tensor names, the vision tower, the
fp32 Gated DeltaNet parameters) before we trust it on the real experts.

    python -P tools/merge_identity.py <model dir> <work dir>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def tensors(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():  # noqa: SIM118 (a safetensors handle, not a dict)
                out[k] = h.get_tensor(k)
    return out


def main(model: str, work: str) -> None:
    w = Path(work)
    w.mkdir(parents=True, exist_ok=True)
    # mergekit rejects the same path twice ("Duplicate merge inputs"), so the
    # second input of the linear merge is a symlink to the same directory.
    twin = w / "twin"
    if not twin.exists():
        twin.symlink_to(Path(model).resolve(), target_is_directory=True)
    configs = {
        "linear": f"merge_method: linear\nmodels:\n  - model: {model}\n    parameters: {{weight: 0.5}}\n"
                  f"  - model: {twin}\n    parameters: {{weight: 0.5}}\ndtype: float32\n",
        "ties": f"merge_method: ties\nbase_model: {model}\nmodels:\n  - model: {model}\n"
                f"    parameters: {{weight: 1.0, density: 1.0}}\ndtype: float32\n",
    }
    ref = tensors(Path(model))
    for name, cfg in configs.items():
        (w / f"{name}.yaml").write_text(cfg)
        out = w / name
        r = subprocess.run(["mergekit-yaml", str(w / f"{name}.yaml"), str(out), "--copy-tokenizer"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{name}: mergekit failed\n{r.stderr[-3000:]}")
            continue
        got = tensors(out)
        missing = sorted(set(ref) - set(got))
        extra = sorted(set(got) - set(ref))
        worst = 0.0
        for k in set(ref) & set(got):
            worst = max(worst, float((ref[k].float() - got[k].float()).abs().max()))
        print(f"{name}: {len(got)} tensors, missing {len(missing)} {missing[:5]}, extra {len(extra)} {extra[:5]}, "
              f"max |diff| {worst:.3g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
