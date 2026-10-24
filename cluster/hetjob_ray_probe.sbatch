#!/bin/bash
#SBATCH --job-name=b2e-hetjob-ray
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#SBATCH hetjob
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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
SETUP="source $CODE/cluster/common.sh; stage_env train; export RAY_TMPDIR=$RAY_TMPDIR"
srun --het-group=0 --ntasks=1 bash -c "$SETUP; HEAD_IP=\$(hostname -I | awk '{print \$1}'); echo \$HEAD_IP > $LOGS/ray_head_$SLURM_JOB_ID.txt; ray start --head --node-ip-address=\$HEAD_IP --port=$PORT --num-gpus=1 --block" &
for _ in $(seq 60); do [ -s "$LOGS/ray_head_$SLURM_JOB_ID.txt" ] && break; sleep 5; done
HEAD_IP=$(cat "$LOGS/ray_head_$SLURM_JOB_ID.txt")
log "ray head on $HEAD_HOST ($HEAD_IP:$PORT)"
sleep 20
srun --het-group=1 --ntasks=1 bash -c "$SETUP; ray start --address=$HEAD_IP:$PORT --num-gpus=1 --block" &
sleep 40
srun --het-group=0 --ntasks=1 --overlap bash -c "$SETUP; python -P - <<'PY'
import os, socket, time
import ray
ray.init(address='$HEAD_IP:$PORT')
for _ in range(30):
    if ray.cluster_resources().get('GPU', 0) >= 2:
        break
    time.sleep(2)
print('cluster resources:', ray.cluster_resources())
@ray.remote(num_gpus=1)
def where():
    import torch
    return socket.gethostname(), torch.cuda.get_device_name(0), os.environ.get('CUDA_VISIBLE_DEVICES')
print('tasks:', ray.get([where.remote(), where.remote()]))
PY"
log "probe done"
kill %1 %2 2>/dev/null || true
