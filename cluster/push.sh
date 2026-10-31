#!/usr/bin/env bash
# Copies this repo (code only, no data or checkpoints) to
# /scratch/$USER/b2e/code on the cluster.
#   cluster/push.sh            # host alias "explorer"
#   B2E_HOST=myhpc cluster/push.sh
#   B2E_DEST=/scratch/me/b2e/jobs/pushtest/code cluster/push.sh   # another dir, to test the push
#
# It needs an already-open ssh master connection and will not open one:
# repeated failed logins get this machine's IP blocked at the login nodes.
# Open it yourself first with `ssh explorer true`.
set -euo pipefail
HOST=${B2E_HOST:-explorer}
if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "no ssh master connection to $HOST; run: ssh $HOST true" >&2
  exit 1
fi
# BatchMode: if the master drops mid-run, fail instead of trying a password.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
cd "$(dirname "$0")/.."

remote_user=$("${SSH[@]}" "$HOST" 'echo $USER')
dest=${B2E_DEST:-/scratch/$remote_user/b2e/code}
# tar over ssh: scp to this cluster drops connections intermittently. On macOS,
# COPYFILE_DISABLE and the two flags keep AppleDouble ._ files and extended
# attributes out of the archive.
tar_flags=()
if [ "$(uname)" = Darwin ]; then
  export COPYFILE_DISABLE=1
  tar_flags=(--no-mac-metadata --no-xattrs)
fi
# The archive is unpacked into a temporary dir next to $dest first and each
# file is then moved over with mv, a rename on the same file system: a job
# reading the code sees the old file or the new one, never part of one, and a
# transfer cut off midway leaves $dest as it was (F#67: a push that stalled on
# a dead connection left a file there empty). rsync --delay-updates would do
# the same but is killed on the login node when it copies the whole tree.
git ls-files -co --exclude-standard -z |
  tar "${tar_flags[@]}" --null -T - -cf - |
  "${SSH[@]}" "$HOST" "set -e; mkdir -p $dest /scratch/$remote_user/b2e/logs
tmp=\$(mktemp -d $dest.push.XXXXXX); trap 'rm -rf \"\$tmp\"' EXIT
tar -C \"\$tmp\" -xf -
cd \"\$tmp\"
find . -type d -exec mkdir -p $dest/{} \\;
find . -type f -exec mv -f {} $dest/{} \\;"
# The cluster copy has no .git: record which commit (and whether the tree
# was dirty) it came from, so every run can log its code version.
{ git rev-parse HEAD 2>/dev/null || echo "no-commit"; git status --porcelain | head -20; } |
  "${SSH[@]}" "$HOST" "cat > $dest/VERSION"
echo "pushed to $HOST:$dest"
