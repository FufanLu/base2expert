#!/bin/bash
#SBATCH --job-name=b2e-setup-addons
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/b2e/logs/%x_%j.out
#
# The instruction-following verifiers (src/b2e/reward/ifevalg) need two small
# pure-Python packages the training env lacks, and they need nltk's
# sentence-splitter data, which compute nodes cannot download.
# Both go next to the envs instead of into the training env, which stays as
# validated; activate_env appends the package dir to PYTHONPATH and sets
# NLTK_DATA. Runs on short because its nodes reach PyPI and GitHub.
#   sbatch cluster/setup_addons.sbatch
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
stage_env train
DST=$ROOT/envs/addons-r1
rm -rf "$DST.partial" && mkdir -p "$DST.partial"
python -m pip install -q --no-deps --target "$DST.partial" "langdetect==1.0.9" "immutabledict==4.3.1"
rm -rf "$DST" && mv "$DST.partial" "$DST"
# nltk 3.10 refuses to download through a proxy (SSRF protection), and the
# short nodes reach the internet only through one (docs/failures.md F#28).
# Rather than opting out of that check, the data is copied from a machine
# that downloaded it directly (nltk.download('punkt'), ('punkt_tab')):
#   tar -C .nltk_data -cf - tokenizers | ssh explorer 'tar -C $ROOT/envs/nltk_data -xf -'
for d in punkt punkt_tab; do
  [ -d "$ROOT/envs/nltk_data/tokenizers/$d" ] || { log "missing $ROOT/envs/nltk_data/tokenizers/$d (see above)"; exit 1; }
done
# check from a clean shell the way jobs will see it
bash -c "source $CODE/cluster/common.sh; stage_env train; cd /tmp && python -P -c '
from b2e.reward.ifcheck import if_score
g = {\"kind\": \"ifeval\", \"prompt\": \"p\", \"constraints\": [{\"instruction_id\": [\"punctuation:no_comma\", \"language:response_language\"], \"kwargs\": [None, {\"language\": \"en\"}]}]}
print(\"if_score\", if_score(\"This answer is written in plain English without any commas at all.\", g))
from b2e.reward.ifevalg import instructions_util as u; print(\"sentences\", u.count_sentences(\"One. Two. Three.\"))
'"
log "addons in $DST, nltk data in $ROOT/envs/nltk_data"
