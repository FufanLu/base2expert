#!/bin/bash
#SBATCH --job-name=b2e-setup-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Builds one of the three environments and packs it for later jobs.
#   sbatch /scratch/$USER/b2e/code/cluster/setup_env.sbatch train
#   sbatch /scratch/$USER/b2e/code/cluster/setup_env.sbatch tools
#   sbatch /scratch/$USER/b2e/code/cluster/setup_env.sbatch nanochat
# Runs on the short partition because its nodes can reach PyPI and GitHub;
# GPU nodes cannot. Each venv is built at the node-local path jobs unpack it to.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
KIND=${1:?usage: setup_env.sbatch train|tools|nanochat}
NAME=$(env_name "$KIND")
DIR=/tmp/$NAME
TAR=$ROOT/envs/$NAME.tar

module load anaconda3/2024.06 >/dev/null 2>&1 || true
PYBASE=$(command -v python3)
"$PYBASE" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
free_gb=$(df -Pk /tmp | awk 'NR == 2 { print int($4 / 1048576) }')
[ "$free_gb" -ge 40 ] || { log "only ${free_gb}G free in /tmp on $(hostname)"; exit 1; }

export UV_CACHE_DIR=/tmp/b2e-uv-cache-$SLURM_JOB_ID UV_LINK_MODE=copy UV_NO_PROGRESS=1
trap 'rm -rf "$UV_CACHE_DIR"' EXIT
rm -rf "$DIR"
"$PYBASE" -m venv "$DIR"
"$DIR/bin/pip" install -q --upgrade pip uv
UV=("$DIR/bin/uv" pip install --python "$DIR/bin/python")
PY=("$DIR/bin/python" -P)

case "$KIND" in
train)
  # 1. vLLM first: its wheel pins torch 2.11.0, and --torch-backend picks the
  #    cu129 build of torch to match.
  WHEEL=https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl
  log "installing vllm $VLLM_VERSION ($CUDA_TAG)"
  "${UV[@]}" --torch-backend "$CUDA_TAG" "vllm @ $WHEEL"
  # 2. verl without its backend extras (they pin cu130 builds), the same set
  #    sql-agent-rl runs on, plus what this project adds:
  #    flash-linear-attention: Triton kernels for Qwen3.5's Gated DeltaNet
  #      layers. Without it transformers falls back to a slow torch loop that
  #      also ignores packed-sequence boundaries (see docs/runs.md, 10-24).
  #      0.5.2 rather than 0.6.0, which came out the day before this env.
  #    math-verify (verl[math]): answer checking for math/finance numbers.
  log "installing verl $VERL_VERSION and training deps"
  "${UV[@]}" --torch-backend "$CUDA_TAG" \
    "verl[verl-core,math]==$VERL_VERSION" \
    "torch==2.11.0" "transformers==5.9.0" \
    "flash-linear-attention==0.5.2" \
    "liger-kernel>=0.8.2" "cupy-cuda12x" \
    "trl==0.27.0" "peft" "accelerate" "deepspeed" \
    "wandb" "tensorboard" "hydra-core" \
    "fastapi" "uvicorn" "httpx" "pydantic>=2" "pandas" "pyarrow" "jsonschema" \
    "lm-eval" "sentencepiece" "tiktoken"
  # 3. Compiled extensions (flash-attn, causal-conv1d) are not part of this env:
  #    cluster/build_ext.sbatch builds their wheels and installs them into
  #    $ROOT/envs/ext-$TRAIN_REV, which jobs add with B2E_USE_EXT=1.
  "${PY[@]}" -c "
import torch, vllm, transformers, verl, ray, trl, peft, fla
print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)
print('vllm', vllm.__version__, 'verl', verl.__version__, 'transformers', transformers.__version__)
print('ray', ray.__version__, 'trl', trl.__version__, 'peft', peft.__version__, 'fla', fla.__version__)
from transformers.utils.import_utils import is_flash_linear_attention_available as f
print('transformers sees fla:', f())
from transformers import Qwen3_5ForCausalLM; print('Qwen3_5ForCausalLM ok')
"
  ;;
tools)
  # CPU-only data and model tools. mergekit pins its own transformers range
  # and pulls torch, so it gets a CPU torch here instead of touching the
  # training env.
  log "installing data and model tools"
  "${UV[@]}" --torch-backend cpu \
    "torch" "mergekit==0.1.5" \
    "datatrove[processing,io]==0.10.1" "fasttext-numpy2-wheel" \
    "mlflow==3.17.2" "polars" "duckdb" "pandas" "pyarrow" \
    "huggingface_hub[hf_xet]" "transformers" "sentencepiece" "tiktoken" \
    "datasketch" "regex" "pydantic>=2" "pyyaml" "spacy"
  "${PY[@]}" -c "
import mergekit, datatrove, fasttext, mlflow, polars, duckdb, transformers
print('datatrove', datatrove.__version__ if hasattr(datatrove, '__version__') else '?', 'mlflow', mlflow.__version__,
      'polars', polars.__version__, 'duckdb', duckdb.__version__, 'transformers', transformers.__version__)
"
  ;;
nanochat)
  # The pre-training side track. nanochat pins torch 2.9.1 on the cu128
  # index, which the H200 driver (570, CUDA 12.8) runs natively. The repo is
  # pinned to a commit and kept on /scratch; the env holds only packages.
  NC_COMMIT=${NC_COMMIT:?set NC_COMMIT to the nanochat commit to pin}
  SRC=$ROOT/pretrain/nanochat
  if [ ! -d "$SRC/.git" ]; then
    git clone -q https://github.com/karpathy/nanochat.git "$SRC"
  fi
  git -C "$SRC" fetch -q origin && git -C "$SRC" checkout -q "$NC_COMMIT"
  log "nanochat at $(git -C "$SRC" rev-parse HEAD)"
  # uv sync into our venv path instead of the repo's .venv.
  (cd "$SRC" && UV_PROJECT_ENVIRONMENT="$DIR" "$DIR/bin/uv" sync --frozen --extra gpu --python "$DIR/bin/python")
  "${PY[@]}" -c "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)"
  ;;
*)
  log "unknown env kind $KIND"; exit 1 ;;
esac

"$DIR/bin/uv" pip freeze --python "$DIR/bin/python" >"$ROOT/envs/$NAME.freeze.txt"
log "packing $TAR"
tar -C /tmp -cf "$TAR.partial" "$NAME"
mv "$TAR.partial" "$TAR"
du -sh "$TAR" >&2
log "done"
