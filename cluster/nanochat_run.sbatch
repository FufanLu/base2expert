#!/bin/bash
#SBATCH --job-name=b2e-nanochat
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Runs one nanochat module in the nanochat env, with NANOCHAT_BASE_DIR on
# /scratch (data, tokenizer, checkpoints, eval bundle).
#   sbatch -p short --gres=none cluster/nanochat_run.sbatch scripts.tok_train --max-chars=2000000000
#   sbatch cluster/nanochat_run.sbatch scripts.base_train --depth=12 --num-iterations=300 ...
# NC_DATA overrides the pre-training data dir (bilingual shards from
# pretrain/make_shards.py are linked in as base_data_climbmix).
# FETCH_EVAL_BUNDLE=1 downloads the CORE eval bundle first (short partition
# only: compute nodes have no internet).
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
MODULE=${1:?nanochat module, e.g. scripts.base_train}; shift
stage_env nanochat
command -v nvidia-smi >/dev/null && gpu_info
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$ROOT/pretrain/base}
# nanochat loads Flash Attention 3 at import time from the Hugging Face kernels
# hub (kernels.get_kernel("varunneal/flash-attention-3")). Compute nodes are
# offline, so without a cached copy it silently falls back to SDPA, which
# cannot run its sliding-window layers efficiently (d12 smoke 11347295: MFU 24%,
# docs/failures.md F#21). FETCH_KERNELS=1 (short partition) downloads the kernel
# into this persistent cache; GPU jobs then load it offline.
export HF_HOME=$ROOT/pretrain/hf_cache
mkdir -p "$HF_HOME"
if [ "${FETCH_KERNELS:-0}" = 1 ]; then
  HF_HUB_OFFLINE=0 python -P -c "
from kernels import get_kernel
k = get_kernel('varunneal/flash-attention-3')
print('cached FA3 kernel:', k.__name__)"
fi
export PYTHONPATH="$ROOT/pretrain/nanochat${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [ "${FETCH_EVAL_BUNDLE:-0}" = 1 ] && [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
  tmp=$(mktemp -d /tmp/b2e-evalbundle-XXXX)
  curl -sSL -o "$tmp/eval_bundle.zip" https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
  (cd "$tmp" && unzip -q eval_bundle.zip) && mv "$tmp/eval_bundle" "$NANOCHAT_BASE_DIR/eval_bundle"
  rm -rf "$tmp"
  log "eval bundle in $NANOCHAT_BASE_DIR/eval_bundle"
fi
cd "$ROOT/pretrain/nanochat"
t0=$SECONDS
python -P -m "$MODULE" "$@"
log "$MODULE done in $((SECONDS - t0))s"
