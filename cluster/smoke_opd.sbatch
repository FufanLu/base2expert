#!/bin/bash
#SBATCH --job-name=b2e-smoke-opd
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# P0 check of on-policy distillation in verl on two GPUs: the student trains
# on GPU 0 (actor + vLLM rollout), the teacher runs as a vLLM server on GPU 1
# and scores the student's own samples token by token. GSM8K, a few steps.
#   sbatch cluster/smoke_opd.sbatch
# Student Qwen3.5-0.8B-Base-chat, teacher Qwen3.5-4B (official post-trained):
# same tokenizer ids (docs/runs.md). LOSS=k3 (reverse KL estimate, default)
# or forward_kl_topk. TASK_REWARDS=False: distillation loss only.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
gpu_info
stage_env train
STUDENT=$(stage_model "${STUDENT_NAME:-Qwen3.5-0.8B-Base-chat}")
TEACHER=$(stage_model "${TEACHER_NAME:-Qwen3.5-4B}")
check_arch "$STUDENT"
LOSS=${LOSS:-k3}
PG=${PG:-True}
TASK_REWARDS=${TASK_REWARDS:-False}
STEPS=${STEPS:-10}

GSM=$DATA/processed/gsm8k
RUN=smoke-opd-$LOSS-pg$PG-$SLURM_JOB_ID
export WANDB_DIR=$CKPT/$RUN RAY_TMPDIR=/tmp/b2e-ray-$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
MAXLEN=$((512 + 1024 + 1))

cd /tmp
t0=$SECONDS
python -P -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$GSM/train.parquet" data.val_files="$GSM/test.parquet" \
  data.train_batch_size=32 data.max_prompt_length=512 data.max_response_length=1024 \
  data.val_max_samples=128 data.dataloader_num_workers=2 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$STUDENT" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=$MAXLEN \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER" \
  distillation.teacher_models.teacher_model.num_replicas=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
  distillation.teacher_models.teacher_model.inference.max_model_len=$MAXLEN \
  distillation.distillation_loss.loss_mode=$LOSS \
  distillation.distillation_loss.use_policy_gradient=$PG \
  distillation.distillation_loss.use_task_rewards=$TASK_REWARDS \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.logger='[console,wandb]' trainer.project_name=base2expert trainer.experiment_name="$RUN" \
  trainer.total_training_steps=$STEPS trainer.test_freq=5 trainer.val_before_train=True \
  trainer.save_freq=-1 trainer.default_local_dir="$CKPT/$RUN" \
  2>&1 | tee "$LOGS/$RUN.log"
log "smoke $RUN done in $((SECONDS - t0))s"
