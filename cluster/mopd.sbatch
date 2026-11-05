#!/bin/bash
#SBATCH --job-name=b2e-mopd
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Multi-teacher on-policy distillation (P8) on two GPUs: the student on the first, every teacher
# on the second (b2e.distill.share_gpu, via cluster/mopd_site/sitecustomize.py). verl's own
# placement gives each teacher its own GPU, four in all, which no job gets under the QoS.
#   RUN=mopd_smoke STUDENT=s_mix TEACHERS="fin=dapo_fin_step90 law=s_law med=s_med" RL_DATA=mopd_smoke \
#     sbatch cluster/mopd.sbatch
# TEACHERS maps each routing key (the `teacher` column of b2e.data.mopd_mix) to a model dir in
# $MODELS; each teacher's vLLM gets TEACHER_MEM of the GPU (default 0.27, three to a GPU) and at most
# TEACHER_SEQS sequences at once (default 256): at 0.27 a teacher has 755 Mamba cache blocks, one per
# sequence, and vLLM refuses to start when max_num_seqs (verl's default 1,024) is more (11611081).
# A teacher added with + does not inherit the teacher_model template of verl's distillation.yaml but
# RolloutConfig's defaults (F#104), so the template's settings that matter are given here: load_format
# auto (the default, dummy, served random weights in the smoke 11626637), enforce_eager true (the
# default captured CUDA graphs, which is where 11611081 stopped), and the student's lengths.
# MOPD_LOSS / MOPD_PG pick the loss (k3 / true by default; forward_kl_topk wants PG false), and
# a top-k loss raises vLLM's max_logprobs to distillation_loss.topk.
# MOPD_BALANCE=share|gap turns on Open-MOPD's per-domain loss weights (b2e.distill.balance),
# which need verl's per-token clamp off (the patch clamps before weighting). LEG_STEPS as in
# rl.sbatch.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
RUN=${RUN:?config name in configs/rl}
STUDENT=${STUDENT:?student model dir}
TEACHERS=${TEACHERS:?key=model ...}
RL_DATA=${RL_DATA:?data dir in $DATA/processed}
NAME=${NAME:-$RUN}
TEACHER_MEM=${TEACHER_MEM:-0.27}
TEACHER_SEQS=${TEACHER_SEQS:-256}
export B2E_USE_EXT=1
gpu_info
stage_env train
export B2E_CODE=$CODE
export B2E_MODEL=$(stage_model "$STUDENT")
check_arch "$B2E_MODEL"
export B2E_TEACHERS_SHARE_GPU=1 PYTHONPATH="$CODE/cluster/mopd_site${PYTHONPATH:+:$PYTHONPATH}"
D=$DATA/processed/$RL_DATA
to_list() { python -P -c "import sys; print('[' + ','.join(repr(l.strip()) for l in open(sys.argv[1]) if l.strip()) + ']')" "$1"; }
train=$(to_list "$D/rl_train.txt") val=$(to_list "$D/rl_val.txt")
resp=$(python -P -c "from omegaconf import OmegaConf; c = OmegaConf.load('$CODE/configs/rl/$RUN.yaml'); print(c.data.max_response_length)")
maxlen=$((4096 + resp + 1))
topk=$(python -P -c "from omegaconf import OmegaConf; c = OmegaConf.load('$CODE/configs/rl/$RUN.yaml'); print(c.distillation.distillation_loss.topk)")
tov=()
for kv in $TEACHERS; do
  key=${kv%%=*} model=$(stage_model "${kv#*=}")
  p=+distillation.teacher_models.$key
  tov+=("$p.key=$key" "$p.model_path=$model" "$p.num_replicas=1" "$p.inference.name=vllm"
        "$p.inference.tensor_model_parallel_size=1" "$p.inference.gpu_memory_utilization=$TEACHER_MEM"
        "$p.inference.max_num_seqs=$TEACHER_SEQS" "$p.inference.max_model_len=$maxlen"
        "$p.inference.load_format=auto" "$p.inference.enforce_eager=true"
        "$p.inference.prompt_length=4096" "$p.inference.response_length=$resp")
  case ${MOPD_LOSS:-k3} in *topk*) tov+=("$p.inference.engine_kwargs.vllm.max_logprobs=$topk") ;; esac
  log "teacher $key: $model"
done
loss=()
[ -n "${MOPD_LOSS:-}" ] && loss+=("distillation.distillation_loss.loss_mode=$MOPD_LOSS")
[ -n "${MOPD_PG:-}" ] && loss+=("distillation.distillation_loss.use_policy_gradient=$MOPD_PG")
if [ -n "${MOPD_BALANCE:-}" ]; then
  export B2E_MOPD_BALANCE=$MOPD_BALANCE
  loss+=("distillation.distillation_loss.loss_max_clamp=null")
  log "loss weights: $MOPD_BALANCE"
fi
export B2E_CKPT_DIR=$CKPT/$NAME
leg=()
if [ -n "${LEG_STEPS:-}" ]; then
  done_steps=$(cat "$B2E_CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  final=$(python -P -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CODE/configs/rl/$RUN.yaml').trainer.total_training_steps)")
  stop=$((done_steps + LEG_STEPS < final ? done_steps + LEG_STEPS : final))
  log "leg: steps $done_steps -> $stop of $final"
  leg=(trainer.total_training_steps=$stop)
fi
mkdir -p "$B2E_CKPT_DIR"
cp "$D"/rl_*.txt "$B2E_CKPT_DIR/" 2>/dev/null || true
cp "$CODE/VERSION" "$B2E_CKPT_DIR/code_version_$SLURM_JOB_ID.txt" 2>/dev/null || true
export WANDB_DIR=$B2E_CKPT_DIR RAY_TMPDIR=/tmp/b2e-ray-$SLURM_JOB_ID
export WANDB_RUN_ID=${WANDB_RUN_ID:-$NAME} WANDB_RESUME=allow
cd /tmp
python -P -m verl.trainer.main_ppo \
  --config-path "$CODE/configs/rl" --config-name "$RUN" \
  data.train_files="$train" data.val_files="$val" \
  trainer.experiment_name="$NAME" "${tov[@]}" "${loss[@]}" "${leg[@]}" "$@" 2>&1 | tee -a "$LOGS/rl_$NAME.log"
log "leg done"
