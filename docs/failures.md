# Failures and fixes

Every failure gets an entry, including my own command mistakes. Status is
`open` until a later run verifies the fix (the job id is noted). If a cause
has not been established, the entry says so.

| # | Date | What failed | Cause | Fix | Status |
|---|------|-------------|-------|-----|--------|
| 1 | 2026-10-24 | A local `curl … ; echo =====; curl …` printed `(eval):1: ==== not found` and the second fetch (verl's Qwen3.5 example at the v0.9.1 tag) never ran | My command: in zsh a word starting with `=` is expanded as a command path (`=cmd`), so `=====` is an error that stops the line | Quote separators (`echo '====='`) | fixed (later commands) |
| 2 | 2026-10-24 | `bash -n cluster/*.sh cluster/*.sbatch` reported no errors, but it had only checked the first file | My command: `bash -n` takes one script; the other names become that script's arguments | Loop over the files: `for f in …; do bash -n "$f"; done` | fixed (all scripts checked) |
| 3 | 2026-10-24 | `grep -rn … --include=*.py` failed with `no matches found` | My command: zsh expands the unquoted glob before grep sees it | Quote the pattern or drop the flag | fixed |
| 4 | 2026-10-24 | Analysis error: docs/runs.md said both flash-linear-attention and causal-conv1d are required for packed training of Qwen3.5 | I concluded it from transformers' code alone. verl replaces the Gated DeltaNet forward, and without causal-conv1d it runs the conv one packed segment at a time, which is correct | Corrected in docs/runs.md: under verl only fla is required (the torch delta-rule fallback silently drops `cu_seqlens`); causal-conv1d only adds speed | open: to be confirmed on a GPU by `tools/pack_check.py` (11343302) |
| 5 | 2026-10-24 | `ruff` failed on `tools/pack_check.py` (I001, import order) | My code: verl's import was in its own block | `ruff --fix` | fixed |
| 6 | 2026-10-24 | `scontrol update job=11343221 MinMemoryNode=64G` was rejected (`Invalid MinMemoryNode value`) | My command: scontrol takes memory in MB | `MinMemoryNode=65536` | fixed (job now asks for 16 CPUs / 64G) |
| 7 | 2026-10-24 | `tools/pack_check.py` (11343302) died at the first forward: `AttributeError: 'DynamicCache' object has no attribute 'conv_states'` in verl's `qwen3_5_gated_delta_net_forward` | My script left `use_cache` at its default (True), so transformers 5.9 passed a `DynamicCache`; verl 0.9.1's replacement forward still reads the older cache attributes (`cache_params.conv_states[layer]`). Training never hits it (no cache), but it means anything using HF cached generation on a verl-patched Qwen3.5 breaks | `use_cache=False` in the script. Evaluation and generation go through vLLM, never HF `generate()` on a patched model | open: rerun pending |
