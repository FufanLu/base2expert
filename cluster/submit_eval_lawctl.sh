#!/usr/bin/env bash
# P9's control for the articles prompt (F#92): CAIL test's 339 articles rows with
# the prompt as trained (`orig`, `\boxed{264、67}`) and with the example swapped
# for a label from the pool (`ex1` `\boxed{133}`, `ex2` `\boxed{196、264}`;
# b2e.data.law_control, job 11608867), with thinking at 8,192 tokens as the law
# evals. Results: results/<m>/<m>-think-lawctl.
#   cluster/submit_eval_lawctl.sh s_law
# Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
(($#)) || { echo "usage: $0 MODEL..." >&2; exit 1; }
PARTITION=${PARTITION:-gpu-short}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
C=$B/data/processed/law_med/control
data=""
for v in orig ex1 ex2; do
  [ -f "$C/cail_test_articles_$v.parquet" ] || { echo "no $C/cail_test_articles_$v.parquet" >&2; exit 1; }
  data+=" $C/cail_test_articles_$v.parquet"
done
dep=${AFTER:+--dependency=afterok:$AFTER}
cmd="rc=0"
for M in "$@"; do
  [ -f "$B/models/$M/.complete" ] || [ -n "${AFTER:-}" ] || { echo "no $B/models/$M" >&2; exit 1; }
  cmd+="
python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M --data$data --out $B/results/$M/$M-think-lawctl --thinking true --max-tokens 8192 --max-model-len 16384 || rc=1"
done
cmd+="
exit \$rc"
minutes=$((15 + 10 * $#))
STAGE_MODELS="$*" sbatch --parsable $dep --partition="$PARTITION" --gres=gpu:h200:1 --mem=96G --time="$minutes" \
  --job-name="b2e-evalC-$1" cluster/gpu_run.sbatch bash -c "$cmd"
