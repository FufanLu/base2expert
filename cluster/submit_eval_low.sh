#!/usr/bin/env bash
# R-fin-low asked the way it was trained (P7): FinQA test and TAT-QA test with
# thinking at 8,192 tokens, as cluster/submit_eval_r.sh's thinking evals, with
# the low-effort system message before every prompt (run_eval --low-effort, the
# text of b2e.data.effort_pool) → results/<m>/<m>-think-low. For a model whose
# name holds fin_low, also its RL val rows (`finance_low/rl_val512_low`, which
# carry the message already; greedy, 4,096 tokens, as its val_kwargs) →
# results/<m>/valchk-low, the export checked against its training log. Without
# the message the same model goes through submit_eval_r.sh, so the two sit next
# to R-fin's own evals and budget curve (submit_eval_budget.sh).
#   cluster/submit_eval_low.sh dapo_fin_low_step90 dapo_fin_step90
# Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
(($#)) || { echo "usage: $0 MODEL..." >&2; exit 1; }
PARTITION=${PARTITION:-gpu-short}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
P=$B/data/processed
for f in finance/finqa_test finance/tatqa_test finance_low/rl_val512_low; do
  [ -f "$P/$f.parquet" ] || { echo "no $P/$f.parquet" >&2; exit 1; }
done
dep=${AFTER:+--dependency=afterok:$AFTER}
cmd="rc=0"
minutes=10
for M in "$@"; do
  [ -f "$B/models/$M/.complete" ] || [ -n "${AFTER:-}" ] || { echo "no $B/models/$M" >&2; exit 1; }
  R=$B/results/$M
  ev="python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M --thinking true --repeats 1"
  cmd+="
$ev --data $P/finance/finqa_test.parquet $P/finance/tatqa_test.parquet --out $R/$M-think-low --max-tokens 8192 --max-model-len 16384 --low-effort || rc=1"
  minutes=$((minutes + 20))
  if [[ $M == *fin_low* ]]; then
    cmd+="
$ev --data $P/finance_low/rl_val512_low.parquet --out $R/valchk-low --max-tokens 4096 --max-model-len 8192 || rc=1"
    minutes=$((minutes + 8))
  fi
done
cmd+="
exit \$rc"
if [ "$PARTITION" = gpu-short ] && ((minutes > 120)); then
  echo "$# models need ~$minutes min, over gpu-short's 2 h: split them or set PARTITION=gpu" >&2; exit 1
fi
STAGE_MODELS="$*" sbatch --parsable $dep --partition="$PARTITION" --gres=gpu:h200:1 --mem=96G --time="$minutes" \
  --job-name="b2e-evalLow-$1" cluster/gpu_run.sbatch bash -c "$cmd"
