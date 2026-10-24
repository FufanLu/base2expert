#!/usr/bin/env bash
# Queues an RL run as a chain of 8-hour legs (cluster/rl.sbatch), each starting
# after the previous one ends, however it ended.
#   cluster/submit_rl.sh RUN MODEL_NAME RL_DATA LEGS [hydra overrides...]
# Run it on the cluster login node with SBATCH_ACCOUNT set. Every leg counts
# against the 8-job submit limit.
set -euo pipefail
RUN=${1:?run} MODEL_NAME=${2:?model} RL_DATA=${3:?data} LEGS=${4:?legs}
shift 4
cd "$(dirname "$0")/.."
dep=${AFTER:-}
for ((i = 1; i <= LEGS; i++)); do
  id=$(RUN=$RUN MODEL_NAME=$MODEL_NAME RL_DATA=$RL_DATA sbatch --parsable --job-name="b2e-$RUN" \
    ${dep:+--dependency=afterany:$dep} cluster/rl.sbatch "$@")
  echo "leg $i: $id"
  dep=$id
done
