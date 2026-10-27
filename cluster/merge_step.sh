#!/usr/bin/env bash
# Merges a mid-run SFT checkpoint into an HF model, the way sft.sbatch merges
# the last one (it publishes only a finished run), and stages it for this job:
#   cluster/merge_step.sh s_fin_bc 600 Qwen3.5-4B-Base-chat
# → $MODELS/s_fin_bc_step600 and /tmp/b2e-models/s_fin_bc_step600. The third
# argument is the model the run started from; the merged model gets its
# tokenizer and processor files (keep_tokenizer). For measuring a run part way,
# e.g. the val loss after epoch 1 (F#49). Run inside a job in the training env
# (cluster/gpu_run.sbatch bash -c "cluster/merge_step.sh ... && ...").
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
RUN=${1:?run name in ckpt/}; STEP=${2:?step}; FROM=${3:?model the run started from}
src=$CKPT/$RUN/global_step_$STEP
name=${RUN}_step$STEP
[ -d "$src" ] || { log "no $src"; exit 1; }
rm -rf "$MODELS/$name.partial"
python -P -m verl.model_merger merge --backend fsdp --local_dir "$src" --target_dir "$MODELS/$name.partial"
[ -d "$src/huggingface" ] && cp -n "$src"/huggingface/* "$MODELS/$name.partial/" 2>/dev/null || true
keep_tokenizer "$MODELS/$FROM" "$MODELS/$name.partial"
check_arch "$MODELS/$name.partial"
rm -rf "${MODELS:?}/$name" && mv "$MODELS/$name.partial" "$MODELS/$name" && touch "$MODELS/$name/.complete"
log "model at $MODELS/$name"
stage_model "$name" >/dev/null
