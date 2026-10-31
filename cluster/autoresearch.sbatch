#!/bin/bash
#SBATCH --job-name=b2e-ar
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# One stretch of an AutoResearch run (P2.5): pretrain/autoresearch/agent.py
# proposes a change to train.py, trains 5 minutes, keeps or resets, repeats.
# PROPOSER=llm serves Qwen3.6-35B-A3B with vLLM on the same H200 (it idles
# while train.py runs); PROPOSER=random is the baseline and needs no model.
# Each TAG is its own clone on branch autoresearch/<TAG>; rerunning a TAG
# continues it (results.tsv says how many rounds are done).
# NOISE_RUNS (new run only) repeats the unchanged baseline to measure run-to-run
# spread; MIN_GAIN is the margin a change must beat the best by to be kept.
# CONFIRM=2 re-runs a change whose first run beats the best by SCREEN_SD standard
# deviations twice, alternating with the best, before keeping it (agent.py).
# HOURS=1.3 leaves 42 minutes, enough for a round started at 1.3 h to finish
# its 5 runs.
#   TAG=oct25-llm PROPOSER=llm ROUNDS=40 NOISE_RUNS=2 MIN_GAIN=0.002 sbatch cluster/autoresearch.sbatch
#   TAG=oct31-llm PROPOSER=llm NOISE_RUNS=2 CONFIRM=2 MIN_GAIN=0.001 sbatch cluster/autoresearch.sbatch
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
TAG=${TAG:?run tag}; PROPOSER=${PROPOSER:-llm}; ROUNDS=${ROUNDS:-40}
HOURS=${HOURS:-1.3}  # stop starting rounds before the 2-hour limit
AR=$ROOT/pretrain/autoresearch
RUN=$AR/runs/$TAG
gpu_info
stage_env nanochat
NC_PY=/tmp/$(env_name nanochat)/bin/python
stage_env train
TR_PY=/tmp/$(env_name train)/bin/python
export HOME=$ROOT/pretrain/ar_home HF_HOME=$ROOT/pretrain/hf_cache HF_HUB_OFFLINE=1
if [ ! -d "$RUN/.git" ]; then
  git clone -q "$AR/upstream" "$RUN"
  git -C "$RUN" checkout -q -b "autoresearch/$TAG"
  git -C "$RUN" config user.name "b2e autoresearch" && git -C "$RUN" config user.email "b2e@localhost"
fi
URL=""
if [ "$PROPOSER" = llm ]; then
  MODEL=$(stage_model Qwen3.6-35B-A3B)
  # At 0.5 (70.2 of the H200's 140.4 GiB) the KV cache came out at -1.66 GiB and
  # vLLM refused (11360202): weights 65.5 GiB plus graphs and activations need 71.9.
  # 0.62 = 87.0 GiB leaves ~15 GiB of KV cache; with the d8 run's 44 GB peak
  # (results.tsv) the GPU holds both. --max-num-seqs: Qwen3.6 is a hybrid (Gated
  # DeltaNet) model and every running sequence needs one Mamba-state block; 15 GiB
  # holds 750, below the default 1,024, and vLLM refused (11377316). The
  # researcher sends one request at a time.
  PORT=$((20000 + SLURM_JOB_ID % 20000))
  "$TR_PY" -P -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name qwen3.6 \
    --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization 0.62 --max-model-len 40960 --max-num-seqs 4 \
    --enable-prefix-caching >"$LOGS/ar_vllm_$SLURM_JOB_ID.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  for _ in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { log "vLLM did not come up"; tail -30 "$LOGS/ar_vllm_$SLURM_JOB_ID.log"; exit 1; }
  log "vLLM up on port $PORT"
  URL=http://127.0.0.1:$PORT
fi
"$TR_PY" -P "$CODE/pretrain/autoresearch/agent.py" --repo "$RUN" --python "$NC_PY" --proposer "$PROPOSER" \
  ${URL:+--llm-url "$URL"} --rounds "$ROUNDS" --hours "$HOURS" --seed "${SEED:-0}" \
  --noise-runs "${NOISE_RUNS:-0}" --min-gain "${MIN_GAIN:-0}" --confirm "${CONFIRM:-0}" \
  --screen-sd "${SCREEN_SD:-1}" 2>&1 | tee -a "$LOGS/ar_$TAG.log"
log "stretch done: $(($(wc -l <"$RUN/results.tsv") - 1)) rows in results.tsv"
