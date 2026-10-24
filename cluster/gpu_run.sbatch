#!/bin/bash
#SBATCH --job-name=b2e-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Runs a command inside the training env on one H200 (override on the command
# line, e.g. --partition=gpu-short --gres=gpu:a100:1 --constraint=a100@80g).
#   sbatch cluster/gpu_run.sbatch python -P tools/pack_check.py ...
# ENV_KIND=nanochat picks the pre-training env instead of the training env.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
gpu_info
stage_env "${ENV_KIND:-train}"
# STAGE_MODELS="A B": copy these models to /tmp/b2e-models/ first (NFS mmap is slow).
for m in ${STAGE_MODELS:-}; do stage_model "$m" >/dev/null; done
cd "$CODE"
"$@"
