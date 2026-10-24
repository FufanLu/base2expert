#!/bin/bash
#SBATCH --job-name=b2e-ar-prepare
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# One-time setup for the AutoResearch side track (P2.5): Karpathy's
# autoresearch at a pinned commit, its data and tokenizer. prepare.py keeps
# everything under ~/.cache/autoresearch; HOME points at /scratch so the
# shards stay off the home quota, and the ClimbMix shards nanochat already
# downloaded (same files) are linked in, so prepare.py only fetches the rest.
# Runs on short because prepare.py downloads from Hugging Face.
#   sbatch cluster/autoresearch_prepare.sbatch
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
AR_COMMIT=228791fb499afffb54b46200aca536f79142f117
AR=$ROOT/pretrain/autoresearch
export HOME=$ROOT/pretrain/ar_home
mkdir -p "$AR" "$HOME/.cache/autoresearch/data"
if [ ! -d "$AR/upstream/.git" ]; then
  git clone -q https://github.com/karpathy/autoresearch.git "$AR/upstream"
fi
git -C "$AR/upstream" fetch -q origin && git -C "$AR/upstream" checkout -q "$AR_COMMIT"
log "autoresearch at $(git -C "$AR/upstream" rev-parse HEAD)"
for f in "$ROOT"/pretrain/base/base_data_climbmix/shard_*.parquet; do
  ln -sf "$f" "$HOME/.cache/autoresearch/data/$(basename "$f")"
done
stage_env nanochat
cd "$AR/upstream"
python -P prepare.py --num-shards 10
ls -la "$HOME/.cache/autoresearch/data" "$HOME/.cache/autoresearch/tokenizer"
log "done"
