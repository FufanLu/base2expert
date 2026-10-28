"""Load an RL run's train and val files the way verl's PPO trainer does before
it touches the GPU, so a run that would die at its data loader dies on a CPU
node instead. The GRPO smoke 11417519 queued, unpacked the env and staged the
model, then failed on its first dataset read: the file lists reached verl as
one string (F#52).

  1. compose configs/rl/<RUN>.yaml with the overrides cluster/rl.sbatch passes
     (data.train_files / data.val_files from rl_train.txt / rl_val.txt) and
     resolve it as main_ppo does (OmegaConf.resolve);
  2. build the train and val datasets with verl's create_rl_dataset, with the
     tokenizer and processor of the model config (as trainer_base does);
  3. print the row counts per data_source, and the end of one prompt rendered
     with data.apply_chat_template_kwargs (verl's agent loop renders it the
     same way);
  4. score every row's gold answer, written as \\boxed{...}, with the reward the
     trainer calls (b2e.reward.score.compute_score): every row must score 1.

Exits 1 if any step fails or any gold answer scores below 1.

    B2E_MODEL=/tmp/b2e-models/s_fin B2E_CKPT_DIR=/tmp/x B2E_CODE=$CODE \\
    python -P tools/check_rl_data.py --run grpo_fin --data $DATA/processed/finance_pool
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def to_list(path: Path) -> str:
    """The override value cluster/rl.sbatch builds from an rl_*.txt file."""
    return "[" + ",".join(repr(line.strip()) for line in path.read_text().splitlines() if line.strip()) + "]"


def gold_text(gt: dict) -> str:
    if gt["kind"] == "number":  # positional: "0.00001", not repr's "1e-05"
        import numpy as np
        return np.format_float_positional(float(gt["value"]), trim="-")
    return str(gt["value"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="config name in configs/rl")
    ap.add_argument("--data", type=Path, required=True, help="dir with rl_train.txt and rl_val.txt")
    ap.add_argument("overrides", nargs="*", help="further hydra overrides, as passed to rl.sbatch")
    a = ap.parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.ppo.utils import create_rl_dataset
    from verl.utils.config import omega_conf_to_dataclass

    from b2e.reward.score import compute_score

    conf_dir = Path(__file__).resolve().parents[1] / "configs" / "rl"
    overrides = [f"data.train_files={to_list(a.data / 'rl_train.txt')}",
                 f"data.val_files={to_list(a.data / 'rl_val.txt')}", *a.overrides]
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name=a.run, overrides=overrides)
    OmegaConf.resolve(cfg)
    print("train_files:", list(cfg.data.train_files))
    print("val_files:  ", list(cfg.data.val_files))
    model = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
    tok, proc = model.tokenizer, model.processor
    print("processor:", type(proc).__name__ if proc is not None else None)
    kwargs = OmegaConf.to_container(cfg.data.get("apply_chat_template_kwargs", {}))
    bad = 0
    for split, files, is_train, max_samples in (
            ("train", cfg.data.train_files, True, cfg.data.get("train_max_samples", -1)),
            ("val", cfg.data.val_files, False, cfg.data.get("val_max_samples", -1))):
        ds = create_rl_dataset(files, cfg.data, tok, proc, is_train=is_train, max_samples=max_samples)
        sources: Counter = Counter()
        wrong: Counter = Counter()
        for i in range(len(ds)):
            row = ds[i]
            src = row["data_source"]
            sources[src] += 1
            raw = row["reward_model"]["ground_truth"]
            gt = json.loads(raw)
            if compute_score(src, f"<think>\nx\n</think>\n\n\\boxed{{{gold_text(gt)}}}", raw) < 1:
                wrong[src] += 1
                if wrong[src] <= 3:
                    print(f"  {split} row {i} ({src}): gold {gt} does not score 1")
        print(f"{split}: {len(ds)} rows {dict(sources)}; gold scoring below 1: {dict(wrong) or 0}")
        bad += sum(wrong.values())
        renderer = proc if proc is not None else tok
        text = renderer.apply_chat_template(ds[0]["raw_prompt"], add_generation_prompt=True, tokenize=False, **kwargs)
        print(f"{split} row 0 rendered, last 120 chars: {text[-120:]!r}")
        print(f"{split} row 0 prompt tokens: {len(tok(text)['input_ids'])} (max_prompt_length {cfg.data.max_prompt_length})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
