#!/bin/bash
#SBATCH --job-name=b2e-download-url
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# Downloads URLs into $ROOT/data/raw/<name>/ and records their sha256 in
# $ROOT/data/raw/manifests/<job id>.tsv. For datasets that live outside the
# Hugging Face Hub (FinQA and ConvFinQA are on the authors' GitHub).
#   sbatch cluster/download_url.sbatch finqa https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json ...
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
NAME=${1:?name}; shift
dest=$DATA/raw/$NAME
mkdir -p "$dest" "$DATA/raw/manifests"
for url in "$@"; do
  f=$dest/$(basename "$url")
  curl -sSfL --retry 3 -o "$f.partial" "$url" && mv "$f.partial" "$f"
  printf '%s\t%s\t%s\t%s\n' "$NAME" "$url" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)" >>"$DATA/raw/manifests/$SLURM_JOB_ID.tsv"
  log "$(basename "$f") $(du -h "$f" | cut -f1)"
done
