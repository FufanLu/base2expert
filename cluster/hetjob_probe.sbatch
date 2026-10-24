#!/bin/bash
#SBATCH --job-name=b2e-hetjob-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#SBATCH hetjob
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#
# Does the gpu QoS (MaxGRESPerJob = 1 GPU) accept a heterogeneous job whose
# components ask for one GPU each? If yes, multi-teacher distillation can get
# 4 GPUs that start together as one job (P0, option a).
#   sbatch cluster/hetjob_probe.sbatch
set -euo pipefail
srun --het-group=0 bash -c 'echo "component 0 on $(hostname): $(nvidia-smi -L | head -1)"' &
srun --het-group=1 bash -c 'echo "component 1 on $(hostname): $(nvidia-smi -L | head -1)"' &
wait
