#!/bin/bash
#SBATCH --job-name=b2e-build-ext
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Builds the compiled extensions the training env needs and installs them into
# $ROOT/envs/ext-$TRAIN_REV, which jobs add to PYTHONPATH with B2E_USE_EXT=1.
#   sbatch /scratch/$USER/b2e/code/cluster/build_ext.sbatch
# - causal-conv1d: Qwen3.5's Gated DeltaNet layers run a short causal conv
#   before the delta rule. transformers uses this package's kernel when it is
#   installed, and only that kernel takes `seq_idx`, i.e. stops at packed-
#   sequence boundaries; the fallback torch conv1d runs across them.
# - flash-attn: reused from sql-agent-rl, which built 2.8.3 for torch 2.11 +
#   CUDA 12 (no upstream wheel exists for that pair).
# Compiling needs nvcc, not a GPU. Kernels are built for A100 (sm_80) and
# H200 (sm_90) only.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
CC1D_VERSION=${CC1D_VERSION:-1.7.0}
stage_env train
module load cuda/12.8.0
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export CUDA_HOME
nvcc --version | tail -2 >&2

BUILD=/tmp/b2e-ext-build-$SLURM_JOB_ID
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/out"
export UV_CACHE_DIR=$BUILD/uv-cache UV_LINK_MODE=copy UV_NO_PROGRESS=1
# Build tools go into a throwaway target so the packed env stays untouched.
uv pip install --python "$(command -v python)" --target "$BUILD/tools" ninja packaging setuptools wheel
export PYTHONPATH="$BUILD/tools${PYTHONPATH:+:$PYTHONPATH}" PATH="$BUILD/tools/bin:$PATH"

whl=$(ls "$WHEELS"/causal_conv1d-"$CC1D_VERSION"-*.whl 2>/dev/null | tail -1 || true)
if [ -z "$whl" ]; then
  export TORCH_CUDA_ARCH_LIST="8.0;9.0" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-16}
  t0=$SECONDS
  pip wheel --no-build-isolation --no-deps -w "$BUILD/out" "causal-conv1d==$CC1D_VERSION"
  cp "$BUILD"/out/causal_conv1d-*.whl "$WHEELS/"
  whl=$(ls "$WHEELS"/causal_conv1d-"$CC1D_VERSION"-*.whl | tail -1)
  log "built $(basename "$whl") in $((SECONDS - t0))s"
fi
fa=$(ls "$WHEELS"/flash_attn-*.whl 2>/dev/null | tail -1 || true)
if [ -z "$fa" ]; then
  fa=$(ls /scratch/$USER/sar/wheels/flash_attn-*.whl | tail -1)
  cp "$fa" "$WHEELS/" && fa=$WHEELS/$(basename "$fa")
fi

dst=$ROOT/envs/ext-$TRAIN_REV
rm -rf "$dst.partial"
uv pip install --python "$(command -v python)" --target "$dst.partial" --no-deps "$whl" "$fa"
rm -rf "$dst" && mv "$dst.partial" "$dst"
ls "$dst" >&2
# No GPU here, so only check that the packages and their shared objects exist.
ls "$dst"/causal_conv1d_cuda*.so "$dst"/flash_attn_2_cuda*.so >&2
log "installed $(basename "$whl") $(basename "$fa") -> $dst"
