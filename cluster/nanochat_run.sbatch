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
