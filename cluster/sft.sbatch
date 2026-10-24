#!/bin/bash
#SBATCH --job-name=b2e-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# SFT with verl's SFT trainer (configs/sft/<cfg>.yaml), resumable:
# a resubmitted job with the same RUN continues from the last checkpoint.
# When the run reaches its last step the final checkpoint is merged into an HF
# model at $MODELS/$RUN.
#   MODEL_NAME=m0 SFT_DATA=$B2E_ROOT/data/sft/fin.parquet RUN=s_fin sbatch cluster/sft.sbatch s_dom
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
CFG=${1:?config name in configs/sft}; shift
MODEL_NAME=${MODEL_NAME:?}; RUN=${RUN:?}; SFT_DATA=${SFT_DATA:?}
export B2E_USE_EXT=1  # flash-attn (varlen full attention) + causal-conv1d
gpu_info
stage_env train
export B2E_MODEL=$(stage_model "$MODEL_NAME")
check_arch "$B2E_MODEL"
export B2E_CODE=$CODE RUN SFT_DATA
export B2E_CKPT_DIR=$CKPT/$RUN WANDB_DIR=$CKPT/$RUN
mkdir -p "$B2E_CKPT_DIR"
cp "$CODE/VERSION" "$B2E_CKPT_DIR/code_version_$SLURM_JOB_ID.txt" 2>/dev/null || true
ls -la $SFT_DATA >"$B2E_CKPT_DIR/data_listing_$SLURM_JOB_ID.txt"
cd /tmp
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer \
  --config-path "$CODE/configs/sft" --config-name "$CFG" "$@" 2>&1 | tee -a "$LOGS/sft_$RUN.log"

if [ "${PUBLISH:-1}" != 1 ]; then log "PUBLISH=0: not merging"; exit 0; fi
last=$(ls -d "$B2E_CKPT_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
[ -n "$last" ] || { log "no checkpoint"; exit 0; }
log "merging $last"
rm -rf "$MODELS/$RUN.partial"
python -P -m verl.model_merger merge --backend fsdp --local_dir "$last" --target_dir "$MODELS/$RUN.partial"
[ -d "$last/huggingface" ] && cp -n "$last"/huggingface/* "$MODELS/$RUN.partial/" 2>/dev/null || true
check_arch "$MODELS/$RUN.partial"
rm -rf "$MODELS/$RUN" && mv "$MODELS/$RUN.partial" "$MODELS/$RUN" && touch "$MODELS/$RUN/.complete"
log "model at $MODELS/$RUN"
