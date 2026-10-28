#!/bin/bash
#SBATCH --job-name=b2e-rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# One 8-hour leg of an RL run (configs/rl/<RUN>.yaml). cluster/submit_rl.sh
# chains several legs; each resumes from the newest checkpoint in $CKPT/$RUN
# (trainer.resume_mode=auto) and one that finds the run finished exits.
#   RUN=grpo_fin MODEL_NAME=s_fin RL_DATA=finance_pool sbatch cluster/rl.sbatch [hydra overrides...]
# RL_DATA names a directory in $DATA/processed with <prefix>_train/_val files
# listed in $DATA/processed/<RL_DATA>/rl_train.txt and rl_val.txt (one path
# per line), so a run can mix several sources. NAME (default RUN) names the
# checkpoint dir, log and W&B run: a smoke of a config passes its own NAME so it
# never writes into, or resumes from, the real run's checkpoints.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
RUN=${RUN:?config name in configs/rl}
MODEL_NAME=${MODEL_NAME:?model dir name in $MODELS}
RL_DATA=${RL_DATA:?data dir in $DATA/processed}
NAME=${NAME:-$RUN}
export B2E_USE_EXT=1
gpu_info
stage_env train
export B2E_CODE=$CODE
export B2E_MODEL=$(stage_model "$MODEL_NAME")
check_arch "$B2E_MODEL"
D=$DATA/processed/$RL_DATA
to_list() { python -P -c "import sys; print('[' + ','.join(repr(l.strip()) for l in open(sys.argv[1]) if l.strip()) + ']')" "$1"; }
train=$(to_list "$D/rl_train.txt") val=$(to_list "$D/rl_val.txt")
export B2E_CKPT_DIR=$CKPT/$NAME
mkdir -p "$B2E_CKPT_DIR"
cp "$D"/rl_*.txt "$B2E_CKPT_DIR/" 2>/dev/null || true
cp "$CODE/VERSION" "$B2E_CKPT_DIR/code_version_$SLURM_JOB_ID.txt" 2>/dev/null || true
export WANDB_DIR=$B2E_CKPT_DIR RAY_TMPDIR=/tmp/b2e-ray-$SLURM_JOB_ID
export WANDB_RUN_ID=${WANDB_RUN_ID:-$NAME} WANDB_RESUME=allow
cd /tmp
python -P -m verl.trainer.main_ppo \
  --config-path "$CODE/configs/rl" --config-name "$RUN" \
  data.train_files="$train" data.val_files="$val" \
  trainer.experiment_name="$NAME" "$@" 2>&1 | tee -a "$LOGS/rl_$NAME.log"
log "leg done"
