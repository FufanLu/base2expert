# base2expert

From a pre-trained-only base model to three domain experts (finance, law,
medicine), and back to one model that keeps all three.

The base model is [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base).
It has seen no instruction tuning, so every stage of post-training is done here:

1. **Mid-training** on a mix of finance, law and medical text, general replay
   data and long chain-of-thought data.
2. **SFT** per domain, with thinking and non-thinking modes, on data
   synthesized by a teacher model and kept only when a verifier accepts it.
3. **Preference optimization and RL**: DPO / SimPO / KTO / ORPO offline, and
   PPO (with a critic), GRPO, DAPO and GSPO online. The algorithms are
   compared under the same data and sampling budget.
4. **Fusion** of the three RL experts into one model by multi-teacher
   on-policy distillation (MOPD). Two baselines are compared against it: RL
   on all three domains at once, and TIES / DARE weight merging. A low-effort finance
   teacher (per-prompt token budgets, Kimi K3's recipe) gives the student a second effort level.

Two recursive self-improvement loops are added on top:

- **Model level**: the finance expert writes new questions about real
  filings, each with a program that computes the answer; an external judge
  rejects invalid questions and must reproduce the answer; the expert trains on
  the ones it sometimes solves, for three rounds.
- **Research level**: following Karpathy's AutoResearch, a local Qwen3.6
  agent edits the small model's training script, trains for 5 minutes, and
  keeps a change only after re-runs beat the best by the node's measured noise;
  two lanes (one per GPU) share a candidate pool.

A side track pre-trains a 0.56B bilingual model from scratch with
[nanochat](https://github.com/karpathy/nanochat). It covers the tokenizer, a
small scaling-law fit, data-mix ablations, and chained 8-hour jobs.

> **Status (2026-11-04):** main line P0, P1, P3, P4 and P6 done; P5 done (four offline methods,
> and the rubric judge, which agrees with HealthBench physicians as well as they agree with each
> other: class-balanced F1 0.686 vs 0.689, difference 95 % [−0.021, +0.017]); P7 (law and medicine
> RL) in progress, law and medicine at step 60; P7.5's grounded pilot has proposed (83 % / 95 % of
> proposals kept without / with thinking; 7 / 20 and 1 / 20 of the kept flawed by hand audit); its
> validity gate, at 8,192 tokens, was cut inside its thinking on half the problems and so dropped 20 of
> the 32 sound audited ones along with all 8 flawed (F#100), and is queued again at 32,768; the P8 code (teacher sharing, Open-MOPD weights, weight merging)
> are written, their GPU runs queued; R-fin's thinking-budget curve is measured (below); P9 not started. Side track: tokenizer, data shards and a d12 smoke so far. Only measured numbers appear
> below, each with its job in [docs/runs.md](docs/runs.md); every failure, including my own
> mistakes, is in [docs/failures.md](docs/failures.md); the stage plan is in
> [docs/plan.md](docs/plan.md).

## Model lineage

```
Qwen3.5-4B-Base
  └─ M0  mid-training (finance + law + medicine + general replay + long-CoT anneal)     done
       ├─ S-fin SFT → R-fin RL (DAPO, P6)  ─┐                                            done
       │    └─ R-fin-low (low reasoning effort, per-prompt token budgets)               running
       ├─ S-law SFT → R-law RL (DAPO, P7)  ─┼─ teachers ─┐                              running
       ├─ S-med SFT → R-med RL (DAPO, P7)  ─┘            │                              running
       └─ S-mix SFT (all three) ────────────────────── MOPD ──► Trident-4B              smoke queued
             ├─ baseline: RL on the three-domain mix
             └─ baseline: TIES / DARE merge of R-fin, R-law, R-med
R-fin → self-written problems on real filings, validity-gated, three rounds (P7.5)    pilot: gate rerun queued
Side track: nanochat from scratch (tokenizer, d12 smoke); AutoResearch on d8 (two-lane search)
```

## Results so far

**Finance, P5 / P6** (from S-fin; thinking on, 8,192 tokens; FinQA test 1,147 + TAT-QA dev 738 =
1,885 items; paired against S-fin, se in brackets):

| Model | Accuracy | vs S-fin | Notes |
|---|---|---|---|
| S-fin (SFT) | 0.8297 | — | |
| SimPO | 0.8350 | +0.53 (0.39) | offline, S-fin's own pairs |
| DPO | 0.8350 | +0.53 (0.38) | |
| KTO | 0.8361 | +0.64 (0.40) | |
| ORPO | 0.8382 | +0.85 (0.44) | vetoed: MMLU-Pro −2.07 (0.95) |
| GRPO, step 90 | 0.8462 | +1.64 (0.52) | 14.92 H200-hours |
| PPO (critic, GAE), step 90 | 0.8493 | +1.96 (0.53) | 25.04 H200-hours |
| GSPO, step 90 | 0.8509 | +2.12 (0.48) | 14.72 H200-hours |
| **DAPO, step 90 (R-fin)** | **0.8568** | **+2.71 (0.52)** | 12.35 H200-hours; chosen for P7 |

The offline methods leave general ability unchanged (pooled within ±0.13 points) and cost 0.4-0.7
points of instruction following; DAPO is about 2 points ahead of all of them.

**Law and medicine, P7, first leg** (DAPO from S-law / S-med; validation accuracy at steps 0 → 30):
CAIL articles 0.530 → 0.710, charges 0.714 → 0.729, prison term 0.541 → 0.576; MedQA 0.740 → 0.790,
MedMCQA 0.755 → 0.755, CMExam 0.805 → 0.800. Most of the articles gain is the model no longer adding
general-part articles (article 67 above all), which CAIL never labels and the prompt's own example
named (F#92), not better legal reading; the English medicine gain is answers that now finish within
the cap. Law's second leg: articles 0.760, charges 0.779, term 0.564 at step 60 (all 599 rows: 0.595 at
step 0, 0.672 at 30, 0.701 at 60). The control for F#92 measured what the example cost before RL: on CAIL
test's 339 articles questions S-law scores 0.490 with the prompt as trained and 0.596 / 0.575 with an
example drawn from the labels (`\boxed{133}`, `\boxed{196、264}`; paired, p ≈ 1e-5 each, unchanged without
the rows whose label holds the example), and writes general-part articles in 129 answers against 81 / 92.

**R-fin against its thinking budget** (budget forcing: the thinking is closed at B tokens, then the
answer is written; FinQA test 1,147 / TAT-QA test 739):

| Thinking budget | 128 | 256 | 512 | 1,024 | 2,048 | 4,096 | no thinking |
|---|---|---|---|---|---|---|---|
| FinQA | 0.790 | 0.802 | 0.809 | 0.814 | 0.812 | 0.813 | 0.790 |
| TAT-QA | 0.919 | 0.927 | 0.939 | 0.943 | 0.941 | 0.942 | 0.921-0.924 |

The answer after `</think>` is ~220 tokens at every budget and holds the computation; the whole
thinking block is worth about 2.3 points, all of it within 512 tokens (paired against 4,096: 512 p = 0.34
and 0.62, 128 p < 0.001). R-fin thinks ~900 tokens where 512 do as well, which sets the bar for R-fin-low:
it has to beat R-fin cut at the same budget. R-fin-low's first step (multiplier 0.5) put 18.9 % of its
rollouts over budget, not the 42 % R-fin's own rollouts predicted: on the same 72 prompts the
"Reasoning effort: low." flag alone makes R-fin's answers 0.88× as long at the median, before any training.
After leg 1 (30 steps at 0.5), val answers are 29 % shorter at the same accuracy (FinQA 0.8136 → 0.8136,
1,063 → 752 tokens; TAT-QA 0.9399 → 0.9356, 894 → 635), and the rollouts sit at 0.27× their budget, so leg 2
runs at 0.35 (9.1 % of step 30's rollouts over).

**Findings worth knowing** (details in docs/failures.md):
- F#86: the CAIL charges verifier never paid a selective charge (`[走私、贩卖、运输、制造]毒品`);
  S-law's charges accuracy was 0.783, not 0.575.
- F#71 / F#89: pass-rate filters keep problems that are "mixed" only because some samples were cut at
  the length limit; selection now counts a cut as no answer.
- F#92: a prompt's example answer is a training signal; ours contradicted the label convention.
- F#95: two results are comparable only when scored by the same verifier version.

## What is measured

| Capability | How |
|---|---|
| Domain reasoning (multi-step) | FinQA / TAT-QA / ConvFinQA (numeric), CAIL (charge, article, prison term), MedQA / CMExam |
| Instruction following | IFEval, IFBench, CFBench, and a held-out suite of domain constraint templates |
| Controllability | Thinking on/off compliance, thinking-budget error, JSON-schema validity, length limits, over-refusal (XSTest) |
| Open-ended business tasks | Rubric scores from a judge model, with the judge's agreement with physicians on HealthBench reported |
| General ability kept | MMLU-Pro, C-Eval / CMMLU, GSM8K, MATH-500 |

## Stack

| Layer | Used for |
|---|---|
| verl 0.9.1 (Ray, FSDP2, vLLM rollout) | Mid-training and SFT (its SFT trainer), PPO / GRPO / DAPO / GSPO, rollout correction (IcePop / TIS), single- and multi-teacher on-policy distillation |
| TRL | DPO, SimPO, KTO, ORPO |
| Rubric judge (Qwen3.6-35B-A3B in vLLM) | Per-item rubric grading in the HealthBench / Kimi K3 style, calibrated against HealthBench physician labels; replaces a Bradley-Terry reward model |
| nanochat | The from-scratch pre-training side track |
| vLLM 0.26, SGLang | Teacher data synthesis, evaluation, the judge service |
| datatrove, fastText, MinHash | Corpus cleaning, quality classifier, dedup, 13-gram decontamination |
| mergekit | TIES / DARE baselines |
| lm-evaluation-harness, OpenCompass / EvalScope, own verifiers | Evaluation |
| W&B, MLflow | Training curves; model registry and lineage |
| Slurm (Northeastern Explorer), Hydra, uv, pytest, ruff, GitHub Actions | Jobs, configs, envs, tests |

## Layout

| Path | What |
|---|---|
| `cluster/` | Slurm jobs for the Explorer cluster: env builds, model downloads, runs. `common.sh` holds paths and staging helpers |
| `src/b2e/` | The package: data pipeline (`data/`), verifiers, rewards, token budgets and the rubric judge (`reward/`), evaluation (`eval/`), self-written problems and the validity gate (`rsi/`), teacher sharing for MOPD (`distill/`) |
| `pretrain/` | Side-track changes to nanochat (nanochat itself is cloned on the cluster at a pinned commit) |
| `configs/` | Hydra configs, one file per run |
| `tools/` | One-off checks (packing, token embeddings, chat template, train/inference log-prob gap) |
| `docs/` | Plan, run log, failures, results |

## Cluster constraints

The cluster is Northeastern's Explorer. Each GPU job gets one GPU for at most
8 hours, and each user gets at most 4 GPUs at once; the short partition gives
two GPUs for two hours. The H200 driver supports CUDA 12.8. Compute nodes have
no internet. Long runs are therefore chained 8-hour legs that resume from
checkpoints. Multi-teacher distillation, which verl would place on four GPUs
(student plus one per teacher), runs on two: the three teachers share one GPU
through a small patch to verl's teacher placement (`b2e.distill.share_gpu`).
