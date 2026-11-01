#!/usr/bin/env bash
# Queues the thinking evals on the law and medicine sets (P7), one H200 job for
# the models given, as cluster/submit_eval_r.sh does for finance:
#   CAIL2018 test, MedQA test, MedMCQA dev, CMExam test with thinking, greedy,
#     8,192 tokens (finance's thinking evals use the same cap); without
#     thinking these sets are already in the A100 job's general evals;
#   the P7 RL val rows (`{law,med}_pool/p7_split/*_val.parquet`, 600 per
#     domain) greedy with thinking at each domain's RL cap (law 4,096,
#     medicine 8,192), as the trainer's val_kwargs: for a starting model the
#     step-0 val, for an exported checkpoint a check against its training log.
# A failed eval does not stop the others in its job; the job then exits 1.
#   AFTER=<jobid> cluster/submit_eval_lawmed.sh s_law s_med
# ~55 min per model (CMExam's 6,810 questions are most of it); on gpu-short
# (PARTITION, the default) the job must stay within 2 hours.
# Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
(($#)) || { echo "usage: $0 MODEL..." >&2; exit 1; }
PARTITION=${PARTITION:-gpu-short}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
P=$B/data/processed L=$B/data/processed/law_med
for M in "$@"; do [ -f "$B/models/$M/.complete" ] || [ -n "${AFTER:-}" ] || { echo "no $B/models/$M" >&2; exit 1; }; done
law_val=$P/law_pool/p7_split/cail_train_val.parquet
med_val="$P/med_pool/p7_split/medqa_train_val.parquet $P/med_pool/p7_split/medmcqa_train_val.parquet \
$P/med_pool/p7_split/cmexam_train_val.parquet"
for f in $law_val $med_val; do [ -f "$f" ] || { echo "no $f" >&2; exit 1; }; done
dep=${AFTER:+--dependency=afterok:$AFTER}

cmd="rc=0"
for M in "$@"; do
  R=$B/results/$M
  ev="python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M --thinking true --repeats 1"
  cmd+="
$ev --data $L/cail_test.parquet --out $R/$M-think-law --max-tokens 8192 --max-model-len 16384 || rc=1
$ev --data $L/medqa_test.parquet $L/medmcqa_dev.parquet $L/cmexam_test.parquet --out $R/$M-think-med --max-tokens 8192 --max-model-len 16384 || rc=1
$ev --data $law_val --out $R/valchk-law --max-tokens 4096 --max-model-len 8192 || rc=1
$ev --data $med_val --out $R/valchk-med --max-tokens 8192 --max-model-len 16384 || rc=1"
done
cmd+="
exit \$rc"

minutes=$((15 + 55 * $#))
if [ "$PARTITION" = gpu-short ] && ((minutes > 120)); then
  echo "$# models need ~$minutes min, over gpu-short's 2 h: one model per job, or PARTITION=gpu" >&2; exit 1
fi
echo "$cmd" >&2
[ -n "${DRY:-}" ] && exit 0
STAGE_MODELS="$*" sbatch --parsable $dep --partition="$PARTITION" --gres=gpu:h200:1 --mem=96G --time="$minutes" \
  --job-name="b2e-evalLM-$1" cluster/gpu_run.sbatch bash -c "$cmd"
