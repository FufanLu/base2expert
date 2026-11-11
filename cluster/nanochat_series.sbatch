#!/bin/bash
#SBATCH --job-name=b2e-nc-series
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Trains a list of nanochat base models one after another on one H200, each at
# nanochat's default data:param ratio (12), and scores each on the M0 held-out
# documents (pretrain/eval_bpb.py). RUNS is "BASE:DEPTH ...", BASE a dir under
# $ROOT/pretrain with tokenizer/, eval_bundle/ and base_data_climbmix/ (the
# shards of one mix, pretrain/make_shards.py):
#   RUNS="bl2_d5:6 bl2_d5:8 bl2_d5:10 bl2_d5:12 bl2_d5:14" sbatch cluster/nanochat_series.sbatch
# A run whose held-out JSON exists is skipped, so a resubmit after a time-out
# carries on with the first unfinished one (a run cut part way starts over:
# checkpoints are written only at the end). CORE is measured once, at the end
# of each run (500 examples a task). Results: $ROOT/results/nanochat/.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
RUNS=${RUNS:?BASE:DEPTH list}
stage_env nanochat
gpu_info
export HF_HOME=$ROOT/pretrain/hf_cache  # the cached FA3 kernel (cluster/nanochat_run.sbatch)
export PYTHONPATH="$ROOT/pretrain/nanochat${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
R=$ROOT/results/nanochat
mkdir -p "$R"
cd "$ROOT/pretrain/nanochat"
for run in $RUNS; do
  base=${run%%:*} d=${run##*:}
  out=$R/${base}_d${d}_heldout.json
  if [ -f "$out" ]; then log "skip $base d$d: $out exists"; continue; fi
  export NANOCHAT_BASE_DIR=$ROOT/pretrain/$base
  [ -d "$NANOCHAT_BASE_DIR/base_data_climbmix" ] || { log "no shards in $NANOCHAT_BASE_DIR"; exit 1; }
  # CORE runs after training; a missing bundle makes nanochat download it and
  # fail on the move out of /tmp, losing the run (F#121)
  [ -f "$NANOCHAT_BASE_DIR/eval_bundle/core.yaml" ] || { log "no eval_bundle in $NANOCHAT_BASE_DIR"; exit 1; }
  t0=$SECONDS
  python -P -m scripts.base_train --depth="$d" --model-tag="d$d" --run=dummy \
    --core-metric-every=999999 --core-metric-max-per-task=500 --sample-every=-1 --save-every=-1 \
    2>&1 | tee "$R/${base}_d${d}_train.log"
  log "$base d$d trained in $((SECONDS - t0))s"
  python -P "$CODE/pretrain/eval_bpb.py" --model-tag "d$d" --data "$DATA/cpt/m0/heldout" --out "$out.partial"
  mv "$out.partial" "$out"
  log "$base d$d done in $((SECONDS - t0))s"
done
