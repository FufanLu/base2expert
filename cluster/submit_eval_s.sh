#!/usr/bin/env bash
# Queues the evaluations of one SFT model with the commands S-fin's ran
# (11400096-101), bundled into two jobs to save submit slots, each on the
# hardware type that eval has always used:
#   A100 80 GB (gpu-short): bpb on the M0 held-out documents, general +
#     law/medicine, IF, and the SFT val loss of the model and of M0 on the
#     same rows (`--exclude-train` its own train file, F#47);
#   H200 (gpu-short): finance without thinking (x2) and with thinking.
# A failed eval does not stop the others in its job; the job then exits 1.
#   AFTER=<jobid> cluster/submit_eval_s.sh s_law
# MIX (default mix_<domain>) names the train mix, BASE (default m0) the model
# the run started from, scored on the same val rows; for the Base-chat ablation:
#   MIX=mix_fin BASE=Qwen3.5-4B-Base-chat AFTER=<jobid> cluster/submit_eval_s.sh s_fin_bc
# Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
M=${1:?model dir name in models/, e.g. s_law}
MIX=${MIX:-mix_${M#s_}}
BASE=${BASE:-m0}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
D=$B/data P=$B/data/processed S=$B/data/sft R=$B/results/$M
[ -f "$S/$MIX/train.parquet" ] || { echo "no $S/$MIX/train.parquet" >&2; exit 1; }
dep=${AFTER:+--dependency=afterok:$AFTER}
ev="python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M"
val="$S/mix_fin/val.parquet $S/mix_law/val.parquet $S/mix_med/val.parquet $S/mix_mix/val.parquet"
seen="--exclude-train $S/$MIX/train.parquet"
general="$P/general/gsm8k_test.parquet $P/general/mmlu_pro_test.parquet $P/general/ceval_val.parquet \
$P/general/cmmlu_test.parquet $P/law_med/medqa_test.parquet $P/law_med/medmcqa_dev.parquet \
$P/law_med/cmexam_test.parquet $P/law_med/cail_test.parquet"
finance="$P/finance/finqa_test.parquet $P/finance/tatqa_dev.parquet"

a100="rc=0
python -P -m b2e.eval.bpb --model /tmp/b2e-models/$M --data $D/cpt/m0/heldout --out $R/bpb_$M.json || rc=1
$ev --data $general --out $R/$M-nothink-general --thinking false --repeats 1 || rc=1
$ev --data $P/if/ifeval_test.parquet $P/if/if_multi_val.parquet --out $R/$M-nothink-if --thinking false --repeats 1 || rc=1
python -P tools/sft_val_loss.py --model /tmp/b2e-models/$M --data $val $seen --out $R/sft_val_loss_${M}_unseen.json || rc=1
python -P tools/sft_val_loss.py --model /tmp/b2e-models/$BASE --data $val $seen --out $B/results/$BASE/sft_val_loss_${BASE}_vs_${M#s_}.json || rc=1
exit \$rc"
h200="rc=0
$ev --data $finance --out $R/$M-nothink --thinking false --repeats 2 --max-tokens 4096 || rc=1
$ev --data $finance --out $R/$M-think --thinking true --repeats 1 --max-tokens 8192 --max-model-len 16384 || rc=1
exit \$rc"

STAGE_MODELS="$M $BASE" sbatch --parsable $dep --partition=gpu-short --gres=gpu:a100:1 --constraint=a100@80g \
  --time=02:00:00 --job-name="b2e-evalA-$M" cluster/gpu_run.sbatch bash -c "$a100"
STAGE_MODELS="$M" sbatch --parsable $dep --partition=gpu-short --gres=gpu:h200:1 --mem=96G \
  --time=01:00:00 --job-name="b2e-evalH-$M" cluster/gpu_run.sbatch bash -c "$h200"
