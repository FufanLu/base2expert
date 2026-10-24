#!/bin/bash
#SBATCH --job-name=b2e-nanochat-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Downloads nanochat's ClimbMix pre-training shards with nanochat's own
# downloader, into $ROOT/pretrain/base/base_data_climbmix (NANOCHAT_BASE_DIR).
#   sbatch cluster/nanochat_data.sbatch 8
# Runs on the short partition because compute nodes have no internet.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
N=${1:?number of shards}
stage_env nanochat
export NANOCHAT_BASE_DIR=$ROOT/pretrain/base
mkdir -p "$NANOCHAT_BASE_DIR"
cd "$ROOT/pretrain/nanochat"
t0=$SECONDS
python -P -m nanochat.dataset -n "$N"
log "$(ls "$NANOCHAT_BASE_DIR/base_data_climbmix" | wc -l) shards, $(du -sh "$NANOCHAT_BASE_DIR/base_data_climbmix" | cut -f1), in $((SECONDS - t0))s"
