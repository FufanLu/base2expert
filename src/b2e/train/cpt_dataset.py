"""Mid-training (continued pre-training) dataset for verl's SFT trainer.

verl's SFT trainer accepts a custom dataset class (`data.custom_cls`) that
returns, per sample, `input_ids`, `position_ids` and `loss_mask` (pad_mode
no_padding). Here every sample is one pre-tokenized chunk from
b2e.data.pretokenize and every token is a target, with one exception:

verl's sft_loss flattens the batch and rolls `loss_mask` left by one token to
line it up with next-token log-probs (verl/workers/utils/losses.py). The mask
of a sample's first token therefore ends up on the previous sample's last
position, i.e. on "predict the next sample's first token from the end of this
one". In SFT that first token is a prompt token with mask 0. Here it must be
set to 0 explicitly, or the loss would train across sample boundaries.

Position ids follow verl's MultiTurnSFTDataset: with a Qwen-VL style
processor (Qwen3.5 ships one) they have 4 rows (text positions + 3 M-RoPE
rows, identical for text), otherwise 1 row.
"""

from __future__ import annotations


def sample_fields(ids: list[int], mrope_rows: bool) -> dict[str, list]:
    n = len(ids)
    pos = list(range(n))
    return {
        "input_ids": ids,
        "position_ids": [pos, pos, pos, pos] if mrope_rows else pos,
        "loss_mask": [0] + [1] * (n - 1),
    }


def expand_files(spec) -> list[str]:
    """Parquet files from a path or list of paths; a directory contributes its own *.parquet only.

    Handing a directory to pd.read_parquet makes pyarrow read every file under
    it, recursively: in data/cpt/m0 that is manifest.json and the held-out
    documents in heldout/ (docs/failures.md F#30)."""
    import glob
    import os

    parts = [spec] if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for part in parts:
        for p in str(part).replace(",", " ").split():
            out += sorted(glob.glob(os.path.join(p, "*.parquet"))) if os.path.isdir(p) else [p]
    if not out:
        raise ValueError(f"no parquet files in {spec!r}")
    return out


class PretrainChunkDataset:
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        import numpy as np
        import pandas as pd

        files = expand_files(parquet_files)  # verl passes a str or an OmegaConf ListConfig
        frames = [pd.read_parquet(f, columns=["input_ids"]) for f in files]
        self.ids = pd.concat(frames, ignore_index=True)["input_ids"].tolist()
        if max_samples and 0 < max_samples < len(self.ids):
            rng = np.random.default_rng(config.get("seed", 0) if config else 0)
            keep = rng.choice(len(self.ids), size=max_samples, replace=False)
            self.ids = [self.ids[i] for i in sorted(keep)]
        self.max_length = config.get("max_length", 8192) if config else 8192
        self.mrope = processor is not None and "Qwen2VLImageProcessor" in type(
            getattr(processor, "image_processor", object())).__name__
        print(f"PretrainChunkDataset: {len(self.ids)} chunks from {len(files)} files, mrope rows={self.mrope}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        import torch

        ids = [int(x) for x in self.ids[i]][: self.max_length]
        f = sample_fields(ids, self.mrope)
        return {
            "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
            "position_ids": torch.tensor(f["position_ids"], dtype=torch.long),
            "loss_mask": torch.tensor(f["loss_mask"], dtype=torch.long),
        }
