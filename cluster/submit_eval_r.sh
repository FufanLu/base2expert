#!/usr/bin/env bash
# Queues the evaluations of RL checkpoints exported by cluster/merge_step.sh
# (models/<run>_step<N>) with the commands S-fin's ran (cluster/submit_eval_s.sh),
# on the same hardware types, several models in one job to save submit slots:
#   PART=h200: finance without thinking (x2) and with thinking, and the 512
#     val rows the RL trainer scored (`finance_pool/rl_val512.parquet` from
#     tools/rl_val_rows.py; greedy, thinking, 4,096 tokens, as its val_kwargs),
#     which checks the export against the log;
#   PART=a100 (80 GB): bpb on the M0 held-out documents, general +
#     law/medicine, IF, and the SFT val loss on the val rows of MIX (default
#     mix_fin, S-fin's mix), `--exclude-train` its train file as for S-fin.
# A failed eval does not stop the others in its job; the job then exits 1.
#   PART=h200 AFTER=<jobid> cluster/submit_eval_r.sh grpo_fin_step90 dapo_fin_step90 gspo_fin_step90
# The time limit is set from the number of models (H200 ~25 min each, A100
# ~40); on gpu-short (PARTITION, the default) it must stay within 2 hours.
# Run it on the cluster login node with SBATCH_ACCOUNT set.
set -euo pipefail
PART=${PART:?h200 or a100}
(($#)) || { echo "usage: PART=h200|a100 $0 MODEL..." >&2; exit 1; }
MIX=${MIX:-mix_fin}
PARTITION=${PARTITION:-gpu-short}
cd "$(dirname "$0")/.."
B=/scratch/$USER/b2e
D=$B/data P=$B/data/processed S=$B/data/sft
[ -f "$S/$MIX/train.parquet" ] || { echo "no $S/$MIX/train.parquet" >&2; exit 1; }
for M in "$@"; do [ -f "$B/models/$M/.complete" ] || [ -n "${AFTER:-}" ] || { echo "no $B/models/$M" >&2; exit 1; }; done
dep=${AFTER:+--dependency=afterok:$AFTER}
val="$S/mix_fin/val.parquet $S/mix_law/val.parquet $S/mix_med/val.parquet $S/mix_mix/val.parquet"
general="$P/general/gsm8k_test.parquet $P/general/mmlu_pro_test.parquet $P/general/ceval_val.parquet \
$P/general/cmmlu_test.parquet $P/law_med/medqa_test.parquet $P/law_med/medmcqa_dev.parquet \
$P/law_med/cmexam_test.parquet $P/law_med/cail_test.parquet"
finance="$P/finance/finqa_test.parquet $P/finance/tatqa_dev.parquet"

cmd="rc=0"
for M in "$@"; do
  R=$B/results/$M
  ev="python -P -m b2e.eval.run_eval --model /tmp/b2e-models/$M"
  case $PART in
    h200) cmd+="
$ev --data $finance --out $R/$M-nothink --thinking false --repeats 2 --max-tokens 4096 || rc=1
$ev --data $finance --out $R/$M-think --thinking true --repeats 1 --max-tokens 8192 --max-model-len 16384 || rc=1
$ev --data $P/finance_pool/rl_val512.parquet --out $R/valchk --thinking true --repeats 1 --max-tokens 4096 --max-model-len 8192 || rc=1" ;;
    a100) cmd+="
python -P -m b2e.eval.bpb --model /tmp/b2e-models/$M --data $D/cpt/m0/heldout --out $R/bpb_$M.json || rc=1
$ev --data $general --out $R/$M-nothink-general --thinking false --repeats 1 || rc=1
$ev --data $P/if/ifeval_test.parquet $P/if/if_multi_val.parquet --out $R/$M-nothink-if --thinking false --repeats 1 || rc=1
python -P tools/sft_val_loss.py --model /tmp/b2e-models/$M --data $val --exclude-train $S/$MIX/train.parquet --out $R/sft_val_loss_${M}_unseen.json || rc=1" ;;
    *) echo "PART is h200 or a100, not $PART" >&2; exit 1 ;;
  esac
done
cmd+="
exit \$rc"

case $PART in
  h200) gres=(--gres=gpu:h200:1 --mem=96G) per=25 tag=H ;;
  a100) gres=(--gres=gpu:a100:1 --constraint=a100@80g) per=40 tag=A ;;
esac
minutes=$((15 + per * $#))
if [ "$PARTITION" = gpu-short ] && ((minutes > 120)); then
  echo "$# models need ~$minutes min, over gpu-short's 2 h: split them or set PARTITION=gpu" >&2; exit 1
fi
STAGE_MODELS="$*" sbatch --parsable $dep --partition="$PARTITION" "${gres[@]}" --time="$minutes" \
  --job-name="b2e-eval$tag-$1" cluster/gpu_run.sbatch bash -c "$cmd"
