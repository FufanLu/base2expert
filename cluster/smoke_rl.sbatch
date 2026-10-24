#!/bin/bash
#SBATCH --job-name=b2e-smoke-rl
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# P0 stack check for Qwen3.5 in verl: GSM8K, one GPU, a few steps. It proves
# vLLM rollout + FSDP2 training + weight sync + (for PPO) a critic work with
# the hybrid Gated DeltaNet model before any project data is involved.
#   MODEL_NAME=Qwen3.5-0.8B-Base-chat ALGO=grpo sbatch cluster/smoke_rl.sbatch
#   MODEL_NAME=Qwen3.5-4B-Base-chat   ALGO=ppo  OFFLOAD=True sbatch cluster/smoke_rl.sbatch
# ALGO=grpo: group-relative advantage, no critic. ALGO=ppo: GAE with a critic
# initialised from the same weights. RMPAD=True packs each micro-batch into one
# row (needs fla, see tools/pack_check.py); RMPAD=False pads instead.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
gpu_info
stage_env train
MODEL_NAME=${MODEL_NAME:-Qwen3.5-0.8B-Base-chat}
ALGO=${ALGO:-grpo}
RMPAD=${RMPAD:-True}
OFFLOAD=${OFFLOAD:-False}
STEPS=${STEPS:-10}
MODEL=$(stage_model "$MODEL_NAME")
[ "$RMPAD" != True ] || check_arch "$MODEL"
python -P -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f, is_causal_conv1d_available as c; import flash_attn; print('fla', f(), 'causal_conv1d', c(), 'flash_attn', flash_attn.__version__)"

GSM=$DATA/processed/gsm8k
if [ ! -f "$GSM/train.parquet" ]; then
  mkdir -p "$GSM" && cp /scratch/$USER/sar/data/processed/gsm8k/*.parquet "$GSM/"
fi

RUN=smoke-$ALGO-$MODEL_NAME-rmpad$RMPAD-$SLURM_JOB_ID
export WANDB_DIR=$CKPT/$RUN RAY_TMPDIR=/tmp/b2e-ray-$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"

ALGO_ARGS=(algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5)
if [ "$ALGO" = ppo ]; then
  ALGO_ARGS=(
    algorithm.adv_estimator=gae algorithm.gamma=1.0 algorithm.lam=0.95
    actor_rollout_ref.rollout.n=1
    critic.model.path="$MODEL" critic.model.use_remove_padding=$RMPAD
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=1e-5 critic.ppo_micro_batch_size_per_gpu=8
    critic.fsdp.param_offload=$OFFLOAD critic.fsdp.optimizer_offload=$OFFLOAD
    trainer.critic_warmup=0
  )
fi

cd /tmp
t0=$SECONDS
python -P -m verl.trainer.main_ppo \
  "${ALGO_ARGS[@]}" \
  data.train_files="$GSM/train.parquet" data.val_files="$GSM/test.parquet" \
  data.train_batch_size=32 data.max_prompt_length=512 data.max_response_length=1024 \
  data.val_max_samples=128 data.dataloader_num_workers=2 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=$RMPAD \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.logger='[console,wandb]' trainer.project_name=base2expert trainer.experiment_name="$RUN" \
  trainer.total_training_steps=$STEPS trainer.test_freq=5 trainer.val_before_train=True \
  trainer.save_freq=-1 trainer.default_local_dir="$CKPT/$RUN" \
  2>&1 | tee "$LOGS/$RUN.log"
log "smoke $RUN done in $((SECONDS - t0))s"
