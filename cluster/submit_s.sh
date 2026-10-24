#!/usr/bin/env bash
# Builds one SFT mix and queues its SFT run on top of M0.
#   cluster/submit_s.sh fin [LEGS] [--after JOBID ...]
# fin | law | med | mix picks configs/data/sft_<d>.yaml; the run is called s_<d>
# and its model is published to $MODELS/s_<d>. The mix waits for the given
# jobs (the syntheses it reads); the SFT legs wait for the mix and for M0
# (models/m0 must exist when the first leg starts, see cpt.sbatch).
# Run on the login node with SBATCH_ACCOUNT set.
set -euo pipefail
D=${1:?fin|law|med|mix}; LEGS=${2:-1}; shift $(( $# >= 2 ? 2 : 1 ))
AFTER=""
if [ "${1:-}" = --after ]; then shift; AFTER=$(IFS=:; echo "$*"); fi
cd "$(dirname "$0")/.."
source cluster/common.sh
mix=$(sbatch --parsable --job-name=b2e-mix-$D --cpus-per-task=8 --mem=64G --time=02:00:00 \
  ${AFTER:+--dependency=afterok:$AFTER} \
  cluster/cpu_run.sbatch python -P -m b2e.data.sft_mix --recipe configs/data/sft_$D.yaml --data-root "$DATA" \
  --tokenizer "$MODELS/Qwen3.5-4B-Base-chat" --decontam-index "$DATA/processed/eval_items" --out "$DATA/sft/mix_$D")
echo "mix: $mix"
dep=$mix
for ((i = 1; i <= LEGS; i++)); do
  kind=$([ "$i" = 1 ] && echo afterok || echo afterany)
  id=$(MODEL_NAME=m0 RUN=s_$D SFT_DATA=$DATA/sft/mix_$D/train.parquet sbatch --parsable --job-name=b2e-s_$D \
    --dependency=$kind:$dep cluster/sft.sbatch s_dom)
  echo "sft leg $i: $id"
  dep=$id
done
