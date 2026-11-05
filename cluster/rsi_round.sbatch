#!/bin/bash
#SBATCH --job-name=b2e-rsi
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# One stage of a self-improvement round (P7.5) into $DATA/rsi/round<ROUND>; b2e.rsi.round says
# why the stages run in this order. MODEL is the round's proposer, solver and policy (a dir in
# $MODELS).
#   propose  MODEL writes N grounded problems without thinking, on one GPU (--gres=gpu:h200:1);
#            PRIOR lists earlier problems files, whose problems are not proposed again
#   pass     MODEL answers each problem 8 times (T 1.0, top-p 1.0, 4,096 tokens: the RL rollout's
#            settings), half the problems on each GPU in chunks of 500 that a rerun of the stage
#            skips; then the pass rates and the mixed problems (right and wrong among finished)
#   gate     Qwen3.6-35B-A3B at 32,768 tokens (F#100) on the mixed problems, a shard on each GPU,
#            then the shards joined: gate/problems_gated.parquet holds the round's usable problems
#   STAGE=propose ROUND=1 MODEL=dapo_fin_step90 N=10000 PRIOR="a/problems.parquet b/problems.parquet" \
#     sbatch --gres=gpu:h200:1 --time=01:00:00 cluster/rsi_round.sbatch
#   STAGE=pass ROUND=1 MODEL=dapo_fin_step90 sbatch cluster/rsi_round.sbatch
#   STAGE=gate ROUND=1 sbatch cluster/rsi_round.sbatch
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
STAGE=${STAGE:?propose, pass or gate}
ROUND=${ROUND:?round}
R=$DATA/rsi/round$ROUND
gpu_info
stage_env train
IFS=, read -ra DEV <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
two() { [ ${#DEV[@]} -ge 2 ] || { echo "need 2 GPUs, got $CUDA_VISIBLE_DEVICES"; exit 1; }; }
mkdir -p "$R"
cd /tmp
case $STAGE in
  propose)
    m=$(stage_model "${MODEL:?}")
    prior=()
    [ -n "${PRIOR:-}" ] && read -ra prior <<< "$PRIOR" && prior=(--prior "${prior[@]}")
    python -P -m b2e.rsi.propose --model "$m" --docs "$DATA/rsi/docs.parquet" --n "${N:?}" --round "$ROUND" \
      --seed 0 --temperature 1.0 --no-thinking --max-tokens 4096 --decontam-index "$DATA/processed/eval_items" \
      "${prior[@]}" --out "$R"
    ;;
  pass)
    two
    m=$(stage_model "${MODEL:?}")
    python -P -m b2e.rsi.round halves --problems "$R/problems.parquet" --n 2 --out "$R/ids"
    pids=()
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=${DEV[$i]} python -P -m b2e.data.synth --teacher "$m" --k 8 --temperature 1.0 \
        --top-p 1.0 --max-tokens 4096 --per-source 1000000 --chunk 500 --only-ids "$R/ids/ids_${i}of2.txt" \
        --out "$R/pass/h$i.parquet" "$R/problems.parquet" > "$LOGS/rsi_r${ROUND}_pass_h${i}_$SLURM_JOB_ID.log" 2>&1 &
      pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ $rc -eq 0 ] || { log "a half failed or ran out of time: rerun the stage, it resumes from its chunks"; exit 1; }
    python -P -m b2e.rsi.round join --out "$R/passrate.parquet" "$R/pass/h0_passrate.parquet" "$R/pass/h1_passrate.parquet"
    python -P -m b2e.rsi.round mixed --problems "$R/problems.parquet" --passrate "$R/passrate.parquet" \
      --out "$R/mixed.parquet"
    ;;
  gate)
    two
    j=$(stage_model Qwen3.6-35B-A3B)
    pids=()
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=${DEV[$i]} python -P -m b2e.rsi.gate --judge "$j" --max-tokens 32768 --max-model-len 40960 \
        --shard $i --shards 2 --problems "$R/mixed.parquet" --out "$R/gate" \
        > "$LOGS/rsi_r${ROUND}_gate_${i}of2_$SLURM_JOB_ID.log" 2>&1 &
      pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ $rc -eq 0 ] || exit 1
    for f in gate problems_gated; do
      python -P -m b2e.rsi.round join --out "$R/gate/$f.parquet" "$R/gate/${f}_0of2.parquet" "$R/gate/${f}_1of2.parquet"
    done
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
log "stage $STAGE done"
