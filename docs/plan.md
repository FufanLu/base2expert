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

## P2.5 Side track: agentic auto-research (recursive self-improvement, research level)

- Follows Karpathy's AutoResearch protocol (2026-03): data prep and
  evaluation are fixed, the agent edits only the training script, every trial
  trains for a fixed 5 minutes, and val_bpb decides keep or discard.
- The researcher is Qwen3.6-35B-A3B served by vLLM on the same H200 as the
  small training run (compute nodes are offline). Our own `program.md`
  instructs it.
- About 15 trials per 2-hour gpu-short job, 3-4 jobs.
- Baselines: the default config, and random search with the same number of
  trials (LR, warmdown, data mix).
- The best change found is carried to the d20 run.
- Upgrade (2026-11-03, the user's choice after the review in docs/runs.md):
  - two trials at a time, one on each of a gpu-short job's two GPUs, so a
    stretch gets about twice the trials (14 per 1.3 h on one GPU were too few
    against the node's noise, 11378056; SkyPilot's 16-GPU run screened in
    parallel waves of 10-13 and confirmed the best);
  - a candidate pool instead of one greedy chain: the researcher edits a
    parent drawn from the archive of trained commits (ShinkaEvolve-style
    parent sampling, with a novelty check that rejects a proposal repeating an
    earlier one), not always the current best;
  - the keep rule stays `--confirm` (screen, then re-run against the best).

## P3 M0 mid-training

- From Qwen3.5-4B-Base on ~0.5-1B tokens, set after measuring throughput:
  ~55% domain text (three domains, zh+en), 30% general replay, 15% long-CoT
  anneal.
- LR re-warmup, then cosine decay to 10% of the peak (verl 0.9.1's FSDP engine has only constant and cosine schedules; WSD is used in the nanochat side track).
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
- ~~Bradley-Terry reward model (TRL), scored on RewardBench 2 and held-out
  domain pairs.~~ Dropped 2026-11-03: Kimi K3's post-training (2026-07) has no
  separate reward-model or DPO stage and scores non-verifiable tasks with a
  generative judge. The rubric judge follows its protocol instead:
  - a rubric per prompt (RaR-Medicine's own where the prompt has one, else
    written by the judge for that prompt);
  - each candidate scored against it, and the group's candidates compared in
    pairs (a tournament), which gives the group reward;
  - a candidate over a verbosity limit loses its comparison, so length does
    not buy reward;
  - used as the reward for medicine's open prompts (P7) only after its
    agreement with HealthBench physician grades is measured.
- DPO vs SimPO vs KTO vs ORPO from S-fin, on S-fin's own samples paired by
  verifier or judge. Done (11-02/03): kept as the offline baseline, not
  extended.

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
- Medicine adds RaR-Medicine rubric rewards (the P5 rubric judge).
- Accuracy-vs-thinking-budget curves.
- Effort control (2026-11-03, Kimi K3's recipe in place of an L1-style
  length reward): a low-effort finance teacher, R-fin-low, continued from
  R-fin. Each prompt's budget is S-fin's (the cold-start model's) length on it
  from the pool sampling times a multiplier; a rollout over its budget gets
  reward -1 even when right; the multiplier is annealed down over the run.
  Its prompts carry a low-effort flag, so P8 can route on it and the student
  learns two effort levels.

## P7.5 Model self-improvement loop (recursive self-improvement, model level)

From R-fin, each round:

1. The model writes new finance problems, each with a Python program that
   computes the answer (FinQA's own "program → answer" form), executed in a
   sandbox to give the gold answer.
2. Dedup and decontaminate; keep problems the model solves 10-90% of the time.
3. Sample 8 answers per problem and check them against the program's result.
4. GRPO on the new problems gives the next round's model.

Redesign (2026-11-03, after the pilots' hand audits: 9 of 20 kept no-thinking
problems flawed, and `--majority` does not catch them because S-fin follows a
flawed gold; R-Quest (2026-10) reports invalid self-made questions growing over
rounds and answer-consistency filtering raising their share, CoVerRL (2026-03)
calls it the consensus trap, DARC (2026-01) and SPICE (2025-10) ground the
questions in documents):

1. Grounded proposals: the proposer is given a real filing's text and table
   (FinQA / TAT-QA train documents, or EDGAR 10-K tables from P1) and asks a
   new question about it; the program reads its numbers from that table, so
   text and table cannot disagree and no figure is invented.
2. Validity gate before the pass-rate filter: Qwen3.6-35B-A3B judges whether
   the question is answerable from the document and unambiguous, and answers
   it independently; a problem is kept only if the judge calls it valid and
   its answer matches the program's. The judge is external, not the solver's
   own majority.
3. Dedup by meaning: as built (bd129ca), two problems on the same excerpt
   with the same answer (or one in percent) are the same task, and a problem
   repeating one of the dataset's own answers on its excerpt is dropped; a
   frozen-model pair judge, the first idea, is not needed once problems are
   grounded (problems on different excerpts use different figures). Then
   decontamination, then the 10-90 % filter with a cut answer counted as no
   answer (F#89, `rl_pool --finished`).
4. The pilot audits' cheap fixes stay: the program prints its unrounded result
   (FinQA's own gold convention; this replaces "4 significant digits"),
   "percentage change" vs "percentage points" and the sign of a change defined
   in the prompt, a larger cap (8,192) for a thinking proposer.
5. Each round, 20 kept problems are read by hand, giving the flawed share per
   round.

- Three rounds.
- Open-ended prompts use the model as its own judge, compared against the
  external judge (Qwen3.6), to measure whether self-judging drifts.
- Measured: per-round scores on fixed eval sets, diversity of the generated
  problems, saturation, and whether problems drift towards easy ones
  (generator gaming).
- Baseline: the same compute spent on RL over the fixed dataset.

## P8 Fusion

- MOPD: student S-mix, teachers R-fin / R-law / R-med, routed by
  `data_source`; reverse KL (k3, policy-gradient form). Forward top-k KL is
  optional.
- Two core comparisons added 2026-11-03 (multi-teacher on-policy
  distillation is now the consolidation step of DeepSeek-V4, Kimi K3 and
  MiMo-V2-Flash; the MOPD paper, 2026-06, uses our baselines):
  - the KL: the sampled per-token reverse KL against a full-vocabulary (or
    top-k) logit KL, which DeepSeek-V4 is reported to use because the sampled
    one is noisy; which of the two verl supports at a 248K vocabulary is
    checked first;
  - the token budget: plain MOPD against Open-MOPD's (2026-08) token-share
    balancing and gap-aware allocation across domains. Our domains differ in
    cap (4,096 law, 8,192 medicine) and answer length, the imbalance it
    diagnoses; instruction following, the shortest answers, is the one
    expected to suffer.
    As built (92412cc, `b2e.distill.balance`, `MOPD_BALANCE=share|gap`): the
    weights are per routing key and per step; the gap is an EMA (decay 0.9, ours)
    of each key's rollout-time per-token loss, so gap weights need a sampled-token
    loss (k3), and a top-k loss gets share weights only; the paper's per-mini-batch
    reward refresh is left out.
- Teachers include R-fin-low (P7), routed on the effort flag.
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
