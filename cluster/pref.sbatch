#!/bin/bash
#SBATCH --job-name=b2e-po
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# One P5 preference-optimization run (configs/po/<RUN>.yaml, b2e.train.pref_train)
# from MODEL_NAME on the rendered pairs in $DATA/processed/<PREF_DATA>
# (b2e.data.pref_render). A finished run is published as $MODELS/<NAME> the way
# cluster/merge_step.sh publishes an RL step: the starting model's tokenizer and
# processor files (keep_tokenizer), check_arch, then .complete.
#   RUN=dpo_fin MODEL_NAME=s_fin PREF_DATA=pref_fin sbatch --job-name=b2e-dpo_fin cluster/pref.sbatch
# A smoke passes its own NAME (its own checkpoint dir, nothing published) and
# pref_train's smoke flags:
#   NAME=dpo_fin_smoke RUN=dpo_fin MODEL_NAME=s_fin PREF_DATA=pref_fin \
#     sbatch --partition=gpu-short --time=01:00:00 cluster/pref.sbatch --max-steps 3 --rows 32
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
RUN=${RUN:?config name in configs/po}
MODEL_NAME=${MODEL_NAME:?model dir name in $MODELS}
PREF_DATA=${PREF_DATA:?data dir in $DATA/processed}
NAME=${NAME:-$RUN}
gpu_info
stage_env train
export B2E_CODE=$CODE
B2E_MODEL=$(stage_model "$MODEL_NAME")
check_arch "$B2E_MODEL"
OUT=$CKPT/$NAME
mkdir -p "$OUT"
cp "$CODE/VERSION" "$OUT/code_version_$SLURM_JOB_ID.txt" 2>/dev/null || true
export WANDB_DIR=$OUT WANDB_RUN_ID=${WANDB_RUN_ID:-$NAME} WANDB_RESUME=allow
cd /tmp
python -P -m b2e.train.pref_train --config "$CODE/configs/po/$RUN.yaml" --model "$B2E_MODEL" \
  --data "$DATA/processed/$PREF_DATA" --out "$OUT" "$@" 2>&1 | tee -a "$LOGS/po_$NAME.log"
if [ -d "$OUT/final" ]; then
  rm -rf "$MODELS/$NAME.partial"
  mv "$OUT/final" "$MODELS/$NAME.partial"
  keep_tokenizer "$MODELS/$MODEL_NAME" "$MODELS/$NAME.partial"
  check_arch "$MODELS/$NAME.partial"
  rm -rf "${MODELS:?}/$NAME" && mv "$MODELS/$NAME.partial" "$MODELS/$NAME" && touch "$MODELS/$NAME/.complete"
  log "model at $MODELS/$NAME"
fi
log "run done"
