#!/usr/bin/env bash
# P7's accuracy against the thinking budget: FinQA test and TAT-QA test with
# thinking, the thinking cut at each of BUDGETS tokens and closed by us, then
# --answer-tokens for the answer (b2e.eval.budget, s1's budget forcing without
# "Wait"). The unforced point is the model's finance eval with thinking at
# 8,192 (<m>-think, <m>-think-tatqa_test). Each job first runs 16 prompts of
# each set at the smallest budget and stops if a set has fewer than half its
# answers boxed, before spending the slot on the full run. Results:
# results/<m>/<m>-think-budget (or results/baseline/<m>-think-budget).
#   cluster/submit_eval_budget.sh dapo_fin_step90
#   BUDGETS="256 1024" AFTER=<jobid> cluster/submit_eval_budget.sh dapo_fin_low_step90
# Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
(($#)) || { echo "usage: $0 MODEL..." >&2; exit 1; }
PARTITION=${PARTITION:-gpu-short}
BUDGETS=${BUDGETS:-128 256 512 1024 2048 4096}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
F=$B/data/processed/finance
for t in finqa_test tatqa_test; do
  [ -f "$F/$t.parquet" ] || { echo "no $F/$t.parquet" >&2; exit 1; }
done
dep=${AFTER:+--dependency=afterok:$AFTER}
smallest=$(tr ' ' '\n' <<<"$BUDGETS" | sort -n | head -1)
cmd="rc=0"
for M in "$@"; do
  [ -f "$B/models/$M/.complete" ] || [ -n "${AFTER:-}" ] || { echo "no $B/models/$M" >&2; exit 1; }
  if [ -d "$B/results/baseline/$M-nothink" ]; then R=$B/results/baseline/$M; else R=$B/results/$M/$M; fi
  ev="python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M --data $F/finqa_test.parquet $F/tatqa_test.parquet --thinking true --answer-tokens 1024 --max-model-len 16384"
  cmd+="
if $ev --out $R-think-budget-smoke --limit 16 --budgets $smallest &&
  python -P -c 'import json, sys; s = json.load(open(sys.argv[1]))[\"sets\"]; print({k: v[\"runs\"][0] for k, v in s.items()}); sys.exit(any(v[\"runs\"][0][\"boxed_rate\"] < 0.5 for v in s.values()))' $R-think-budget-smoke/summary.json; then
  $ev --out $R-think-budget --budgets $BUDGETS || rc=1
else
  echo '$M: budget smoke failed, full run skipped'; rc=1
fi"
done
cmd+="
exit \$rc"
nb=$(wc -w <<<"$BUDGETS")
minutes=$((15 + $# * (5 + 4 * nb)))
if [ "$PARTITION" = gpu-short ] && ((minutes > 120)); then
  echo "$# models need ~$minutes min, over gpu-short's 2 h: split them or set PARTITION=gpu" >&2; exit 1
fi
STAGE_MODELS="$*" sbatch --parsable $dep --partition="$PARTITION" --gres=gpu:h200:1 --mem=96G --time="$minutes" \
  --job-name="b2e-evalB-$1" cluster/gpu_run.sbatch bash -c "$cmd"
