#!/bin/bash
#SBATCH --job-name=b2e-download-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Downloads files of a Hugging Face dataset repo into $ROOT/data/hf/<repo>.
#   sbatch cluster/download_data.sbatch <repo> [glob] [max_files]
#   sbatch cluster/download_data.sbatch BAAI/IndustryCorpus2_finance_economics 'chinese/high/*' 1
# glob selects files (fnmatch, default: everything); max_files keeps the
# first N matches in sorted order, so reruns pick the same shards. Big corpora
# are sampled this way instead of downloaded whole. A manifest line per file
# (repo, path, size, sha256) goes to $ROOT/data/hf/manifests/<job id>.tsv: one
# file per job, because appends to one shared file on NFS are not atomic.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
REPO=${1:?usage: download_data.sbatch <repo> [glob] [max_files]}
GLOB=${2:-*}
MAX=${3:-0}
stage_env tools
export HF_HUB_OFFLINE=0 HF_XET_CACHE=/tmp/b2e-xet-$SLURM_JOB_ID HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'rm -rf "$HF_XET_CACHE"' EXIT
t0=$SECONDS
mkdir -p "$DATA/hf/manifests"
python -P - "$REPO" "$GLOB" "$MAX" "$DATA/hf" "$DATA/hf/manifests/$SLURM_JOB_ID.tsv" <<'PY'
import fnmatch, hashlib, os, sys
from huggingface_hub import HfApi, hf_hub_download
repo, glob, max_files, root, manifest = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, glob))
if max_files:
    files = files[:max_files]
dest = os.path.join(root, repo)
os.makedirs(dest, exist_ok=True)
with open(manifest, "w") as man:
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        size = os.path.getsize(p)
        man.write(f"{repo}\t{f}\t{size}\t{h.hexdigest()}\n")
        print(f"{f}\t{size / 1e9:.2f} GB", flush=True)
PY
rm -rf "$DATA/hf/$REPO/.cache"
log "downloaded $REPO [$GLOB, max $MAX] in $((SECONDS - t0))s"
