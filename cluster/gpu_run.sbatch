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
# A model the command names as /tmp/b2e-models/<name> is staged too: the IF
# retry 11360164 was submitted without STAGE_MODELS and died on a missing path.
for a in "$@"; do
  case $a in /tmp/b2e-models/*) STAGE_MODELS="${STAGE_MODELS:-} $(echo "${a#/tmp/b2e-models/}" | cut -d/ -f1)" ;; esac
done
for m in ${STAGE_MODELS:-}; do stage_model "$m" >/dev/null; done
cd "$CODE"
"$@"
