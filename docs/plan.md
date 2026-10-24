# Plan

Stages run in order. Each one is recorded before it starts (docs/runs.md) and
is written up in the handout the day it finishes. The first milestone is a
minimal end-to-end line: Base → M0 → S-fin → R-fin (GRPO) → eval. Everything
else fans out from there.

## P0 Environment and feasibility

Short smoke jobs, mostly on `gpu-short`.

- Qwen3.5-4B-Base in the verl 0.9.1 / vLLM 0.26 / transformers 5.9 stack:
  - text-only class (`Qwen3_5ForCausalLM` drops `model.visual.*` and `mtp.*`);
  - Gated DeltaNet kernels (flash-linear-attention, causal-conv1d);
  - packed sequences must not leak into each other. Check: per-token
    log-probs of two packed documents vs. each run alone.
- Special-token embeddings (`<think>`, `</think>`, `<|im_start|>`,
  `<tool_call>`) are trained, not placeholders. This is the lesson from
  sql-agent-rl, where `<tool_call>` turned out untrained.
- 20-step smoke runs: GRPO (0.8B and 4B), PPO with a critic (4B, FSDP
  offload), single-teacher OPD (2 GPUs on gpu-short).
- 4-GPU multi-teacher distillation, three options tried in order:
  - (a) heterogeneous Slurm job;
  - (b) a Ray cluster spanning four single-GPU jobs;
  - (c) a remote-teacher plugin: three 4B teachers on one GPU behind vLLM
    `prompt_logprobs`, checked against built-in single-teacher OPD.
- mergekit on Qwen3.5 (self-merge must be the identity); whether Qwen3.6 and
  Qwen3.5 share a tokenizer; Qwen3.6-35B-A3B throughput in vLLM 0.26;
  nanochat d12 smoke on one H200.
- Fallback if Qwen3.5 training does not work: Qwen3-4B-Base (standard attention).

## P1 Data

- Corpus pipeline (datatrove): normalize → language ID → MinHash dedup →
  fastText quality classifier (trained on ~50k teacher labels) → PII scrub →
  13-gram decontamination against every eval set → tokenize → manifest with
  hashes and licenses.
- SFT synthesis with Qwen3.6-35B-A3B:
  - rejection sampling (only verifier-accepted chains of thought);
  - AutoIF-style constraint data (instruction plus checker code, kept when
    the code passes);
  - paired thinking / non-thinking targets;
  - system-prompt business dialogues;
  - selection by difficulty, judge quality score and embedding diversity.
- RL prompt pools: keep prompts the SFT model solves 10-90% of the time.

## P2 Side track: pre-training from scratch (nanochat)

- Bilingual 64K tokenizer, with its compression on domain text compared to
  the GPT-2 and Qwen tokenizers.
- Scaling-law fit (d8-d16, under 2 h each) that predicts the d20 loss,
  checked against the real run.
- d12 data-mix ablation (general only vs +10% / +20% domain text).
- d20 (~0.56B) over 2-3 chained 8-hour jobs; mid-training (domain + chat,
  context 2K→8K); SFT.
- Comparison with Qwen3.5-0.8B-Base in bits per byte (tokenizer-independent).

## P3 M0 mid-training

- From Qwen3.5-4B-Base on ~0.5-1B tokens, set after measuring throughput:
  ~55% domain text (three domains, zh+en), 30% general replay, 15% long-CoT
  anneal.
- LR re-warmup, then WSD decay.
- Ablations: replay 0% vs 30% (forgetting); finance-only M0-fin (optional).

## P4 SFT

- S-fin, S-law, S-med: each domain's data plus general data, instruction
  following and thinking modes.
- S-mix: all three domains. It is the student for MOPD and the start of the
  mixed-RL baseline.
- Checks: chat template token-equal to `apply_chat_template`; loss on
  assistant tokens only.
- Optional: self-distilled SFT; selected vs random data.

## P5 Rewards and preference optimization

- Verifiers with unit tests:
  - numbers (units, percent, separators, tolerance);
  - multiple choice;
  - sets (charges, articles);
  - prison-term tolerance;
  - constraint checks;
  - JSON schema, length, thinking budget;
  - safety gate.
- Rubric judge (Qwen3.6-35B-A3B on SGLang, separate GPU job, HTTP). Its
  agreement with HealthBench physician grades is reported.
- Bradley-Terry reward model (TRL), scored on RewardBench 2 and held-out
  domain pairs.
- DPO vs SimPO vs KTO vs ORPO from S-fin, on S-fin's own samples paired by
  verifier or judge.

## P6 RL algorithms (finance)

- From S-fin, at equal total samples:
  - PPO (critic, GAE λ=0.95);
  - GRPO (n=8);
  - DAPO (clip-higher, dynamic sampling, token-level loss, overlong shaping);
  - GSPO (optional).
- Hybrid reward: verifiers, constraint checks, thinking budget, and the reward
  model for open prompts.
- Measure the vLLM-vs-FSDP log-prob gap first, then ablate IcePop / TIS.
- Read 50 rollouts per run by hand.

## P7 Law and medicine experts

- R-law and R-med with the recipe chosen in P6. R-fin is the best P6 run.
- Medicine adds RaR-Medicine rubric rewards.
- Accuracy-vs-thinking-budget curves.

## P8 Fusion

- MOPD: student S-mix, teachers R-fin / R-law / R-med, routed by
  `data_source`; reverse KL (k3, policy-gradient form). Forward top-k KL is
  optional.
- Baselines:
  - (a) mixed RL from S-mix at equal GPU-hours;
  - (b) TIES / DARE merge;
  - (c) off-policy distillation (optional);
  - (d) each expert alone as the per-domain ceiling.
- Report the seesaw across domains, general ability, instruction following and
  controllability, and GPU-hours.

## P9 Evaluation, serving, demo

- Results table: every model × every benchmark, thinking on and off.
- Trident-4B behind the vllm-sglang-infra gateway, compared with three
  experts behind a domain router on GPU cost and quality.
- Gradio demo: domain detection, thinking switch, budget slider, JSON mode.
- Optional: FP8; distillation to Qwen3.5-0.8B on a Mac with MLX; MTP
  speculative decoding.
