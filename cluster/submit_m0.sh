#!/usr/bin/env bash
# Builds the M0 mid-training mix and queues M0 as chained 8-hour legs.
#   cluster/submit_m0.sh [TOTAL_TOKENS] [LEGS]      (default 5e8 tokens, 2 legs)
# Run on the login node with SBATCH_ACCOUNT set, after the eight cleaning
# jobs have written $DATA/clean/<source>.
set -euo pipefail
TOTAL=${1:-5e8} LEGS=${2:-2}
cd "$(dirname "$0")/.."
source cluster/common.sh
mix=$(sbatch --parsable --job-name=b2e-m0-mix --cpus-per-task=8 --mem=64G --time=04:00:00 \
  cluster/cpu_run.sbatch python -P -m b2e.data.m0_mix --recipe configs/data/m0.yaml --total-tokens "$TOTAL" \
  --data-root "$DATA" --tokenizer "$MODELS/Qwen3.5-4B-Base-chat" --decontam-index "$DATA/processed/eval_items" \
  --out "$DATA/cpt/m0")
echo "mix: $mix"
dep=$mix
for ((i = 1; i <= LEGS; i++)); do
  # the first leg needs the mix to have succeeded; later legs follow whatever happened before
  kind=$([ "$i" = 1 ] && echo afterok || echo afterany)
  id=$(MODEL_NAME=Qwen3.5-4B-Base-chat RUN=m0 CPT_DATA=$DATA/cpt/m0 sbatch --parsable --job-name=b2e-m0 \
    --dependency=$kind:$dep cluster/cpt.sbatch m0)
  echo "leg $i: $id"
  dep=$id
done
