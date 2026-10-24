#!/bin/bash
#SBATCH --job-name=b2e-hetjob-ray
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#SBATCH hetjob
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#
# Can one Ray cluster span the components of a heterogeneous job (one GPU
# each, usually on different nodes)? Starts the Ray head on component 0 and a
# worker on component 1, then runs one GPU task per node. If this works,
# verl's multi-teacher distillation can place teachers on other nodes by
# connecting to this cluster (P0, option a).
#   sbatch cluster/hetjob_ray_probe.sbatch
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
export RAY_TMPDIR=/tmp/b2e-ray-$SLURM_JOB_ID
# Ray 2.59 enables token authentication by default: the head made its own
# token and the other components' processes were rejected (InvalidAuthToken,
# docs/failures.md F#22). One token for the whole job, readable only by us,
# shared through /scratch and removed at the end. Auth stays on because the
# cluster network is shared with other users.
TOKEN_FILE=$ROOT/.ray_token_$SLURM_JOB_ID
(umask 077 && openssl rand -hex 32 >"$TOKEN_FILE")
trap 'rm -f "$TOKEN_FILE" "$LOGS/ray_worker_$SLURM_JOB_ID.ready"' EXIT
export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$TOKEN_FILE
SETUP="source $CODE/cluster/common.sh; stage_env train; export RAY_TMPDIR=$RAY_TMPDIR RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$TOKEN_FILE"
srun --het-group=0 --ntasks=1 bash -c "$SETUP; HEAD_IP=\$(hostname -I | awk '{print \$1}'); echo \$HEAD_IP > $LOGS/ray_head_$SLURM_JOB_ID.txt; ray start --head --node-ip-address=\$HEAD_IP --port=$PORT --num-gpus=1 --block" &
for _ in $(seq 240); do [ -s "$LOGS/ray_head_$SLURM_JOB_ID.txt" ] && break; sleep 5; done
HEAD_IP=$(cat "$LOGS/ray_head_$SLURM_JOB_ID.txt")
log "ray head on $HEAD_HOST ($HEAD_IP:$PORT)"
# The worker joins only after its own stage_env, which on a node without the
# env cached takes 205-905 s (12.5 GB tar over NFS; docs/runs.md, 11349055).
# It writes a marker when it starts Ray; the driver waits for the marker and
# then for both GPUs, and fails instead of running on one node.
READY=$LOGS/ray_worker_$SLURM_JOB_ID.ready
srun --het-group=1 --ntasks=1 bash -c "$SETUP; hostname > $READY; ray start --address=$HEAD_IP:$PORT --num-gpus=1 --block" &
for _ in $(seq 240); do [ -s "$READY" ] && break; sleep 5; done
[ -s "$READY" ] || { log "worker never got ready"; exit 1; }
log "worker ready on $(cat "$READY")"
srun --het-group=0 --ntasks=1 --overlap bash -c "$SETUP; python -P - <<'PY'
import os, socket, time
import ray
ray.init(address='$HEAD_IP:$PORT')
for _ in range(60):
    if ray.cluster_resources().get('GPU', 0) >= 2:
        break
    time.sleep(2)
print('cluster resources:', ray.cluster_resources())
assert ray.cluster_resources().get('GPU', 0) >= 2, 'worker GPU did not join'
@ray.remote(num_gpus=1)
def where():
    import torch
    return socket.gethostname(), torch.cuda.get_device_name(0), os.environ.get('CUDA_VISIBLE_DEVICES')
print('tasks:', ray.get([where.remote(), where.remote()]))
PY"
log "probe done"
kill %1 %2 2>/dev/null || true
