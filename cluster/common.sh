# Sourced by every cluster job. Paths, versions, and staging helpers live here.
# Adapted from sql-agent-rl's cluster/common.sh.
#
# Layout on the shared filesystem (/scratch is NFS: fine for big sequential
# reads, slow for the thousands of small files a Python env imports):
#   $ROOT/code     this repo (cluster/push.sh)
#   $ROOT/envs     packed Python environments (setup_env.sbatch)
#   $ROOT/wheels   wheels built here (flash-attn, causal-conv1d)
#   $ROOT/models   model weights (download_model.sbatch)
#   $ROOT/data     datasets
#   $ROOT/ckpt     training checkpoints
#   $ROOT/logs     job logs
# Jobs unpack the env and copy the weights to node-local /tmp, once per node.
# /scratch is purged after 28 days: final weights must be copied elsewhere.

ROOT=${B2E_ROOT:-/scratch/$USER/b2e}
CODE=${B2E_CODE:-$ROOT/code}
MODELS=$ROOT/models
DATA=$ROOT/data
CKPT=$ROOT/ckpt
LOGS=$ROOT/logs
WHEELS=$ROOT/wheels

# The H200 nodes run driver 570 (CUDA 12.8). vLLM 0.26.0 is the newest release
# on torch 2.11.0 and has a cu129 build that runs on driver 570 through CUDA
# minor-version compatibility (verified in sql-agent-rl and vllm-sglang-infra).
# verl 0.9.1 is installed without its backend extras, which pin cu130 builds.
VERL_VERSION=${VERL_VERSION:-0.9.1}
VLLM_VERSION=${VLLM_VERSION:-0.26.0}
CUDA_TAG=${CUDA_TAG:-cu129}
# Three environments, packed separately because their pins conflict:
#   train     GPU: verl, vLLM, TRL, flash-linear-attention, lm-eval
#   tools     CPU: datatrove, fastText, mergekit, MLflow (mergekit pins its own transformers)
#   nanochat  GPU: the from-scratch pre-training side track (torch 2.9.1 cu128)
# Bump the REV of an env when its package set changes: nodes keep unpacked
# envs, and a new name is what makes them unpack the new one.
TRAIN_REV=${TRAIN_REV:-r1}
TOOLS_REV=${TOOLS_REV:-r2}  # r2: + spacy (datatrove word tokenizers need it)
NANOCHAT_REV=${NANOCHAT_REV:-r1}

env_name() {
  case "$1" in
    train) echo "b2e-train-verl${VERL_VERSION}-vllm${VLLM_VERSION}-${CUDA_TAG}-${TRAIN_REV}" ;;
    tools) echo "b2e-tools-${TOOLS_REV}" ;;
    nanochat) echo "b2e-nanochat-${NANOCHAT_REV}" ;;
    *) echo "unknown env $1" >&2; return 1 ;;
  esac
}

mkdir -p "$ROOT/envs" "$MODELS" "$DATA" "$CKPT" "$LOGS" "$WHEELS"

# Compute nodes have no internet, and caches must not land in $HOME.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HOME=/tmp/b2e-hf-$USER
export XDG_CACHE_HOME=/tmp/b2e-cache-$USER
export VLLM_CACHE_ROOT=/tmp/b2e-vllm-cache-$USER
export TRITON_CACHE_DIR=/tmp/b2e-triton-$USER
export PYTHONUNBUFFERED=1
# Compute nodes route HTTP through a site proxy (squid), and Python HTTP
# clients send even 127.0.0.1 there, which answers "Access Denied". Local
# services (vLLM, judges, Ray's endpoints) must bypass it.
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"
# W&B cannot reach its server from a compute node: runs log offline into the
# checkpoint directory and are synced from the login node.
export WANDB_MODE=${WANDB_MODE:-offline}
# The mid-training smoke (11347590) had 90 GB allocated but 137 GB reserved
# on a 141 GB H200: allocator fragmentation. Expandable segments let the
# caching allocator grow blocks instead of reserving new ones.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# Every job records itself in $ROOT/runs.tsv when it starts: id, name, node,
# code version and the variables that say what it is running. The outcome
# comes from sacct; results and notes go into docs/runs.md.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${B2E_RUN_RECORDED:-}" ]; then
  export B2E_RUN_RECORDED=1
  {
    printf '%s\t%s\t%s\t%s\t%s\t' "$SLURM_JOB_ID" "${SLURM_JOB_NAME:-}" "$(date '+%F %T')" "$(hostname -s)" \
      "$(head -1 "$CODE/VERSION" 2>/dev/null | cut -c1-10)"
    for v in RUN MODEL_NAME MODEL_DIR TAG CONFIG STAGE DOMAIN; do
      [ -n "${!v:-}" ] && printf '%s=%s ' "$v" "${!v}"
    done
    printf '\n'
  } >>"$ROOT/runs.tsv"
fi

# stage_env KIND unpacks the packed env to /tmp/<name> unless this node already
# has it, then activates it. A lock serializes jobs that land on the same node;
# unpacking into a scratch directory and renaming makes a half-finished unpack
# impossible to pick up. The venv was built at that same /tmp path, so it
# needs no path rewriting.
stage_env() {
  local kind=${1:-train} name
  name=$(env_name "$kind") || return 1
  local tarball=$ROOT/envs/$name.tar dir=/tmp/$name
  [ -f "$tarball" ] || { log "missing $tarball: run setup_env.sbatch $kind first"; return 1; }
  (
    flock 9
    if [ ! -f "$dir/.complete" ]; then
      local t0=$SECONDS stage=/tmp/.b2e-stage-$$
      rm -rf "$dir" "$stage" && mkdir -p "$stage"
      tar -C "$stage" -xf "$tarball"
      mv "$stage/$name" "$dir" && rmdir "$stage"
      touch "$dir/.complete"
      log "unpacked $name in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/.$name.lock"
  activate_env "$dir"
}

# activate_env DIR puts the env's tools on PATH and loads a CUDA 12.8 toolkit,
# the newest the H200 nodes' driver runs (Triton JIT needs it).
activate_env() {
  local dir=$1
  export PATH="$dir/bin:$PATH" VIRTUAL_ENV="$dir"
  # Compiled extensions built here (flash-attn, causal-conv1d) live in their
  # own directory and are added only when a job asks for them, so a run that
  # started without them keeps the same kernels when a later leg resumes it.
  if [ "${B2E_USE_EXT:-0}" = 1 ]; then
    [ -d "$ROOT/envs/ext-$TRAIN_REV" ] || { log "B2E_USE_EXT=1 but $ROOT/envs/ext-$TRAIN_REV is missing"; return 1; }
    export PYTHONPATH="$ROOT/envs/ext-$TRAIN_REV${PYTHONPATH:+:$PYTHONPATH}"
  fi
  # B2E_EXTRA_PYTHONPATH: extra package directories for one job, e.g. a
  # prebuilt extension from another project while ours is still building.
  if [ -n "${B2E_EXTRA_PYTHONPATH:-}" ]; then
    export PYTHONPATH="$B2E_EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if module load cuda/12.8.0 >/dev/null 2>&1 && command -v nvcc >/dev/null; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    export CUDA_HOME
  fi
  # Slurm here sets ROCR_VISIBLE_DEVICES (AMD's variable) next to
  # CUDA_VISIBLE_DEVICES, and verl's workers refuse to start when both are set.
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
  export FLASHINFER_WORKSPACE_BASE=/tmp/b2e-flashinfer-$USER
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
  # vllm-sglang-infra saw NCCL segfault while loading the node's network
  # plugin and its InfiniBand transport.
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
  # The repo's package (src/b2e) is importable without installing.
  export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
  # Pure-Python additions for the instruction-following verifiers
  # (langdetect, immutabledict) and nltk's sentence-splitter data, built by
  # cluster/setup_addons.sbatch. Appended, so an env's own packages win.
  if [ -d "$ROOT/envs/addons-r1" ]; then
    export PYTHONPATH="$PYTHONPATH:$ROOT/envs/addons-r1"
  fi
  if [ -d "$ROOT/envs/nltk_data" ]; then
    export NLTK_DATA=$ROOT/envs/nltk_data
  fi
}

# stage_model NAME copies $MODELS/NAME to node-local disk unless this node
# already has it, and prints the local path. Loading safetensors straight off
# NFS goes through mmap page faults and is far slower than one sequential copy.
stage_model() {
  local name=$1 src=$MODELS/$1 dst=/tmp/b2e-models/$1
  [ -f "$src/.complete" ] || { log "model $name not downloaded: run download_model.sbatch"; return 1; }
  mkdir -p /tmp/b2e-models
  (
    flock 9
    if [ ! -f "$dst/.complete" ]; then
      local t0=$SECONDS
      rm -rf "$dst" "$dst.partial"
      cp -r "$src" "$dst.partial" && mv "$dst.partial" "$dst"
      log "staged $name ($(du -sh "$dst" | cut -f1)) in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/b2e-models/.$name.lock"
  echo "$dst"
}

# gpu_info logs the GPU, driver, and memory of this node.
gpu_info() {
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2 || true
}

# check_arch DIR refuses to train from a checkpoint whose config declares a
# class verl would not pass packed-sequence boundaries to. verl hands
# `cu_seqlens` only to models whose forward() takes it (its patched
# Qwen3_5ForConditionalGeneration); a text-only Qwen3_5ForCausalLM export
# would train with packed sequences leaking into each other (tools/pack_check.py,
# job 11343348: mean |log-prob diff| 0.47 vs 0.02 noise).
check_arch() {
  local arch
  arch=$(python -P -c "import json,sys; print(json.load(open(sys.argv[1]))['architectures'][0])" "$1/config.json")
  if [ "$arch" != Qwen3_5ForConditionalGeneration ] && [ "${B2E_ALLOW_ARCH:-}" != "$arch" ]; then
    log "$1 declares $arch, not Qwen3_5ForConditionalGeneration: packed training would leak (set B2E_ALLOW_ARCH=$arch to override)"
    return 1
  fi
}
