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
run_trainer "$LOGS/sft_$RUN.log" torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer \
  --config-path "$CODE/configs/sft" --config-name "$CFG" "$@"

if [ "${PUBLISH:-1}" != 1 ]; then log "PUBLISH=0: not merging"; exit 0; fi
last=$(ls -d "$B2E_CKPT_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
[ -n "$last" ] || { log "no checkpoint"; exit 0; }
# A leg that stopped at the time limit leaves a mid-run checkpoint; publish
# only when the run is complete (last step = rows / batch size x epochs).
total=$(python -P - "$SFT_DATA" "$CODE/configs/sft/$CFG.yaml" "$@" <<'PY'
import glob, os, sys
import pyarrow.parquet as pq
import yaml
spec, cfg_path, overrides = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = yaml.safe_load(open(cfg_path))
bs = cfg["data"]["train_batch_size"]
epochs = cfg["trainer"].get("total_epochs", 1)
for o in overrides:
    if o.startswith("data.train_batch_size="):
        bs = int(o.split("=", 1)[1])
    if o.startswith("trainer.total_epochs="):
        epochs = int(o.split("=", 1)[1])
paths = []
for part in spec.replace(",", " ").split():
    paths += sorted(glob.glob(os.path.join(part, "*.parquet"))) if os.path.isdir(part) else [part]
rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
fixed = cfg["trainer"].get("total_training_steps")
for o in overrides:
    if o.startswith("trainer.total_training_steps="):
        fixed = int(o.split("=", 1)[1])
# verl's SFT dataloader uses drop_last=True (verl/trainer/sft_trainer.py)
print(fixed if fixed else (rows // bs) * epochs)
PY
)
step=${last##*global_step_}
if [ "$step" -lt "$total" ]; then log "checkpoint at step $step of $total: run not finished, not publishing"; exit 0; fi
log "merging $last"
rm -rf "$MODELS/$RUN.partial"
python -P -m verl.model_merger merge --backend fsdp --local_dir "$last" --target_dir "$MODELS/$RUN.partial"
[ -d "$last/huggingface" ] && cp -n "$last"/huggingface/* "$MODELS/$RUN.partial/" 2>/dev/null || true
keep_tokenizer "$B2E_MODEL" "$MODELS/$RUN.partial"
check_arch "$MODELS/$RUN.partial"
rm -rf "$MODELS/$RUN" && mv "$MODELS/$RUN.partial" "$MODELS/$RUN" && touch "$MODELS/$RUN/.complete"
log "model at $MODELS/$RUN"
