#!/usr/bin/env bash
# Builds one SFT mix and queues its SFT run on top of M0.
#   cluster/submit_s.sh fin [LEGS] [--after JOBID ...]
# fin | law | med | mix picks configs/data/sft_<d>.yaml; the run is called s_<d>
# and its model is published to $MODELS/s_<d>. The mix waits for the given
# jobs (the syntheses it reads); the SFT legs wait for the mix, and for M0
# when M0_JOB (its last leg) is set (models/m0 must exist when the first leg
# starts, see cpt.sbatch).
# Run on the login node with SBATCH_ACCOUNT set.
set -euo pipefail
D=${1:?fin|law|med|mix}; shift
# LEGS is optional: `fin --after J` used to take "--after" as LEGS and drop it
LEGS=1
if [ $# -gt 0 ] && [ "$1" != --after ]; then LEGS=$1; shift; fi
M0_JOB=${M0_JOB:-}
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
  # leg 1: the mix succeeded and M0 is over (cpt.sbatch publishes models/m0 only from a complete run;
  # if M0 is not complete, stage_model fails and the leg ends at once); later legs follow the previous one
  deps=$([ "$i" = 1 ] && echo "afterok:$dep${M0_JOB:+,afterany:$M0_JOB}" || echo "afterany:$dep")
  id=$(MODEL_NAME=m0 RUN=s_$D SFT_DATA=$DATA/sft/mix_$D/train.parquet sbatch --parsable --job-name=b2e-s_$D \
    --dependency=$deps cluster/sft.sbatch s_dom)
  echo "sft leg $i: $id"
  dep=$id
done
