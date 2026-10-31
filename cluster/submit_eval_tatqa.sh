#!/usr/bin/env bash
# Back-fills TAT-QA test (the gold file, F#66) for models whose finance evals
# ran before it was processed, with the finance commands (no thinking x2 at
# 4,096 tokens, thinking at 8,192) on H200, as those evals ran. Results go
# next to the model's finance dirs: results/baseline/<m>-{nothink,think}-tatqa_test
# for a baseline (it has results/baseline/<m>-nothink), else
# results/<m>/<m>-{nothink,think}-tatqa_test. cluster/submit_eval_s.sh and
# cluster/submit_eval_r.sh include TAT-QA test from now on.
#   cluster/submit_eval_tatqa.sh Qwen3.5-4B-Base-chat Qwen3.5-4B m0 s_fin
# ~8 min per model; on gpu-short (PARTITION, the default) the job must stay
# within 2 hours. Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
(($#)) || { echo "usage: $0 MODEL..." >&2; exit 1; }
PARTITION=${PARTITION:-gpu-short}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
T=$B/data/processed/finance/tatqa_test.parquet
[ -f "$T" ] || { echo "no $T (b2e.data.finance --only tatqa_test)" >&2; exit 1; }
dep=${AFTER:+--dependency=afterok:$AFTER}
cmd="rc=0"
for M in "$@"; do
  [ -f "$B/models/$M/.complete" ] || [ -n "${AFTER:-}" ] || { echo "no $B/models/$M" >&2; exit 1; }
  if [ -d "$B/results/baseline/$M-nothink" ]; then R=$B/results/baseline/$M; else R=$B/results/$M/$M; fi
  [ -d "$R-nothink" ] || { echo "no finance results at $R-nothink" >&2; exit 1; }
  ev="python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M --data $T"
  cmd+="
$ev --out $R-nothink-tatqa_test --thinking false --repeats 2 --max-tokens 4096 || rc=1
$ev --out $R-think-tatqa_test --thinking true --repeats 1 --max-tokens 8192 --max-model-len 16384 || rc=1"
done
cmd+="
exit \$rc"
minutes=$((15 + 8 * $#))
if [ "$PARTITION" = gpu-short ] && ((minutes > 120)); then
  echo "$# models need ~$minutes min, over gpu-short's 2 h: split them or set PARTITION=gpu" >&2; exit 1
fi
STAGE_MODELS="$*" sbatch --parsable $dep --partition="$PARTITION" --gres=gpu:h200:1 --mem=96G --time="$minutes" \
  --job-name="b2e-evalT-$1" cluster/gpu_run.sbatch bash -c "$cmd"
