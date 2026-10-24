#!/usr/bin/env bash
# Makes $MODELS/<out> = a Base checkpoint's weights and config + the tokenizer
# files of a post-trained model of the same family.
#   bash cluster/make_chat_model.sh Qwen3.5-4B-Base Qwen3.5-4B Qwen3.5-4B-Base-chat
# Why: Qwen3.5's Base tokenizer does not register <think>, </think>,
# <tool_response>, </tool_response> and its eos is <|endoftext|>; the
# post-trained tokenizer registers them (same vocab.json and merges.txt, so
# the same ids) and ends a turn at <|im_end|>. Their embedding rows are trained
# in the Base weights (docs/runs.md, job 11343276), so only the tokenizer
# files change. No generation_config.json is written: vLLM would take
# sampling defaults from it.
set -euo pipefail
source "${B2E_CODE:-/scratch/$USER/b2e/code}/cluster/common.sh"
BASE=${1:?base model dir name}; DONOR=${2:?tokenizer donor dir name}; OUT=${3:?output dir name}
src=$MODELS/$BASE tok=$MODELS/$DONOR dst=$MODELS/$OUT
[ -f "$src/.complete" ] && [ -f "$tok/.complete" ] || { log "missing $src or $tok"; exit 1; }
cmp -s "$src/vocab.json" "$tok/vocab.json" && cmp -s "$src/merges.txt" "$tok/merges.txt" ||
  { log "vocab/merges differ between $BASE and $DONOR: ids would not match"; exit 1; }
rm -rf "$dst.partial" && mkdir -p "$dst.partial"
cp "$src"/*.safetensors "$src"/model.safetensors.index.json "$src"/config.json "$dst.partial/"
cp "$src"/preprocessor_config.json "$src"/video_preprocessor_config.json "$dst.partial/" 2>/dev/null || true
for f in tokenizer.json tokenizer_config.json chat_template.jinja vocab.json merges.txt; do
  cp "$tok/$f" "$dst.partial/"
done
{
  echo "weights and config: $BASE"
  echo "tokenizer files:    $DONOR"
  sha256sum "$dst.partial"/{tokenizer.json,tokenizer_config.json,chat_template.jinja} | sed "s#$dst.partial/##"
} >"$dst.partial/PROVENANCE.txt"
rm -rf "$dst" && mv "$dst.partial" "$dst" && touch "$dst/.complete"
log "made $dst ($(du -sh "$dst" | cut -f1))"
cat "$dst/PROVENANCE.txt" >&2
