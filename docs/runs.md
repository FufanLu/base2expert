# Run log

Every cluster job, successful or not, in submission order. Checks run on
the login node or locally and design decisions are logged here too, as their
own rows. Every job also records itself in `$B2E_ROOT/runs.tsv` when it
starts (`cluster/common.sh`). Failures are explained in
[failures.md](failures.md) (F#). Times are US Eastern.

Results are copied from the job's own output files or from the source that
was read. Nothing here is estimated.

## 2026-10-24

| Job | What | Partition / node | State | Elapsed | Result |
|---|---|---|---|---|---|
| local | Decision: project scope and plan (docs/plan.md), approved by the user | — | done | — | Base model Qwen3.5-4B-Base; three domains (finance, law, medicine) fused by multi-teacher on-policy distillation; side track pre-trains a model from scratch with nanochat |
| local | Decision: one shared mid-training run (M0) instead of one per domain | — | done | — | Merging (TIES/DARE) and distillation baselines need the experts and the student to start from the same checkpoint. It also saves two mid-training runs. A finance-only M0 is kept as an ablation |
| local | Decision: no LLaMA-Factory | — | done | — | In sql-agent-rl it required transformers ≤5.6 (the env has 5.9) and produced non-finite gradients on half its steps (sql-agent-rl F#19, F#24). Mid-training and SFT use verl's SFT trainer, which ran cleanly there |
| local | Read Qwen3.5-4B-Base `config.json` and file list on the Hub | — | done | — | `Qwen3_5ForConditionalGeneration` (vision tower, depth 24, plus 1 MTP layer). Text: 32 layers in the pattern 3 × Gated DeltaNet then 1 × gated full attention; hidden 2560; vocab 248,320; tied embeddings; `mamba_ssm_dtype` float32; checkpoint 9.3 GB (2 shards) |
| login node | Package list of sql-agent-rl's packed env (`sar-verl0.9.1-vllm0.26.0-cu129-r1.freeze.txt`) | — | done | — | torch 2.11.0+cu129, vLLM 0.26.0, verl 0.9.1, transformers 5.9.0, TRL 0.27.0, lm-eval 0.4.13. **Not installed: flash-linear-attention, causal-conv1d, mergekit, datatrove, MLflow.** No Qwen3.5 weights downloaded |
| local | Read transformers 5.9.0 `models/qwen3_5/modeling_qwen3_5.py` (Gated DeltaNet forward, lines 437-560) | — | done | — | `Qwen3_5ForCausalLM` exists and skips `mtp.*` and `model.visual.*` on load, so a text-only fine-tune saved through it has no MTP head. **Packed sequences:** the delta-rule kernel resets its state at boundaries only through `cu_seqlens`, and only the flash-linear-attention kernel takes it. The torch fallback `torch_chunk_gated_delta_rule(..., **kwargs)` silently drops it. The short causal conv stops at boundaries only with the `causal_conv1d` kernel (`seq_idx`); otherwise it is a plain `conv1d` over the whole packed row, and the last 3 tokens of one document leak into the next. Both packages are therefore required before any packed (remove-padding) training |
| local | Read verl's `examples/grpo_trainer/run_qwen3_5_2b_openr1_fsdp.sh` | — | done | — | verl's own Qwen3.5 recipe runs with `use_remove_padding=False`, micro-batch 1, FSDP2 with param and optimizer offload. It was tested with vllm 0.18.0 and a transformers main commit |
| local | Versions on PyPI | — | done | — | flash-linear-attention 0.6.0 was released 2026-10-23, the day before this env. The env pins 0.5.2 (2026-07-27). causal-conv1d 1.7.0 has no wheel for torch 2.11, so it is built on a CPU node (`cluster/build_ext.sbatch`) |
