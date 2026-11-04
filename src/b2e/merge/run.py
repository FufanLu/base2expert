"""Weight-merging baselines for P8: the three experts merged into one model with mergekit 0.1.5.

Every expert comes from M0 (M0 -> S-<d> -> R-<d>), so each one's task vector is its weights minus
M0's, and M0 is the base of every merge. The grid (`GRID`), all with equal weights:

- avg: task arithmetic with weights 1/3, i.e. the plain average of the experts;
- ties_d<p>: TIES (trim each task vector to its largest-magnitude fraction p, elect a sign per
  parameter, average the values that agree; mergekit's `ties`, normalize on by default);
- dare_d<p>: DARE-TIES (drop a random 1 - p of each task vector and rescale the rest, then TIES'
  sign election and average). mergekit's `dare_ties` sums by default; normalize is turned on here
  so that it differs from TIES only in how it sparsifies. Seeded (`--random-seed`).

The merge is computed in fp32 and written in bf16, the dtype of every tensor of M0 and the
experts (their safetensors headers, 11-03). The merged model takes every file but the weights
from the first expert (tokenizer, chat template, config; mergekit would copy M0's, a base
model's) and keeps mergekit's weights and shard index. --check compares the result's tensor
names and dtypes with the first expert's, and for `avg` checks a few tensors against the mean of
the experts.

    python -P -m b2e.merge.run --base /tmp/b2e-models/m0 --experts /tmp/b2e-models/{dapo_fin_step90,s_law,s_med} \\
        --out $B2E_ROOT/models --prefix merge --names avg ties_d0.5 --check
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

GRID: dict[str, tuple[str, float | None]] = {
    "avg": ("task_arithmetic", None),
    "ties_d0.2": ("ties", 0.2),
    "ties_d0.5": ("ties", 0.5),
    "dare_d0.2": ("dare_ties", 0.2),
    "dare_d0.5": ("dare_ties", 0.5),
}
SEED = 0
WEIGHT_FILES = ("*.safetensors", "model.safetensors.index.json")


def merge_config(name: str, base: str, experts: list[str]) -> dict:
    method, density = GRID[name]
    if method == "task_arithmetic":
        params: dict = {"weight": 1 / len(experts)}
    else:
        params = {"weight": 1.0, "density": density}
    cfg: dict = {"merge_method": method, "base_model": base,
                 "models": [{"model": e, "parameters": dict(params)} for e in experts],
                 "dtype": "float32", "out_dtype": "bfloat16"}  # merged in fp32; every expert tensor is bf16
    if method == "dare_ties":
        cfg["parameters"] = {"normalize": True}
    return cfg


def to_yaml(cfg: dict) -> str:
    # JSON is valid YAML, and needs no YAML library in the env
    return json.dumps(cfg, indent=2) + "\n"


def is_weight_file(p: Path) -> bool:
    return any(p.match(g) for g in WEIGHT_FILES)


def copy_side_files(expert: Path, out: Path) -> list[str]:
    """Everything but the weights from the expert: tokenizer, chat template, configs."""
    copied = []
    for f in sorted(expert.iterdir()):
        if f.is_file() and not is_weight_file(f) and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    return copied


def tensor_dtypes(d: Path) -> dict[str, str]:
    """Each tensor's dtype, from the safetensors headers."""
    out: dict[str, str] = {}
    for f in d.glob("*.safetensors"):
        with f.open("rb") as h:
            header = json.loads(h.read(int.from_bytes(h.read(8), "little")))
        out.update({k: v["dtype"] for k, v in header.items() if k != "__metadata__"})
    return out


def load(d: Path, name: str):
    from safetensors import safe_open

    index = d / "model.safetensors.index.json"
    shard = json.loads(index.read_text())["weight_map"][name] if index.exists() else next(d.glob("*.safetensors")).name
    with safe_open(str(d / shard), "pt") as h:
        return h.get_tensor(name)


def check(name: str, out: Path, experts: list[Path], probes: int = 6) -> dict:
    ref_dtypes, got_dtypes = tensor_dtypes(experts[0]), tensor_dtypes(out)
    ref, got = set(ref_dtypes), set(got_dtypes)
    report: dict = {"tensors": len(got), "missing": sorted(ref - got)[:10], "n_missing": len(ref - got),
                    "extra": sorted(got - ref)[:10], "n_extra": len(got - ref),
                    "dtype_changed": sorted(k for k in ref & got if ref_dtypes[k] != got_dtypes[k])[:10]}
    if name == "avg":  # the average of the experts, up to bf16 rounding
        worst = 0.0
        for t in sorted(got & ref)[:: max(1, len(got & ref) // probes)][:probes]:
            mean = sum(load(e, t).float() for e in experts) / len(experts)
            worst = max(worst, float((load(out, t).float() - mean).abs().max() / (mean.abs().max() + 1e-12)))
        report["avg_max_rel_diff"] = worst
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="directory the merged models go into")
    ap.add_argument("--prefix", default="merge")
    ap.add_argument("--names", nargs="+", default=list(GRID), choices=list(GRID))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    experts = [Path(e) for e in args.experts]
    for name in args.names:
        out = args.out / f"{args.prefix}_{name}"
        if (out / ".complete").exists():
            print(f"{out}: done before, skipped", flush=True)
            continue
        partial = out.with_name(out.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        cfg_path = partial / "b2e_merge.yaml"
        cfg_path.write_text(to_yaml(merge_config(name, args.base, [str(e) for e in experts])))
        subprocess.run(["mergekit-yaml", str(cfg_path), str(partial), "--random-seed", str(SEED),
                        "--no-copy-tokenizer"], check=True)
        copied = copy_side_files(experts[0], partial)
        report = {"name": name, "config": json.loads(cfg_path.read_text()), "side_files_from": str(experts[0]),
                  "side_files": copied}
        if args.check:
            report["check"] = check(name, partial, experts)
        (partial / "b2e_merge_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"name": name, **report.get("check", {})}), flush=True)
        shutil.rmtree(out, ignore_errors=True)
        partial.rename(out)
        (out / ".complete").touch()


if __name__ == "__main__":
    main()
