#!/bin/bash
#SBATCH --job-name=b2e-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Runs a command inside one of the envs on a CPU node (the login node kills
# anything that imports torch). ENV_KIND picks the env (default train).
#   ENV_KIND=tools sbatch cluster/cpu_run.sbatch python -P -m b2e.data.manifest ...
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
stage_env "${ENV_KIND:-train}"
# STAGE_MODELS="A B": copy these models to /tmp/b2e-models/ first (NFS mmap is slow).
for m in ${STAGE_MODELS:-}; do stage_model "$m" >/dev/null; done
cd "$CODE"
"$@"
