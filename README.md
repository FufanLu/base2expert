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
   on all three domains at once, and TIES / DARE weight merging.

Two recursive self-improvement loops are added on top:

- **Model level**: the finance expert writes new problems, each with a
  program that computes the answer. It answers them, keeps what the verifier
  accepts, and is trained again, for three rounds.
- **Research level**: following Karpathy's AutoResearch, a local Qwen3.6
  agent edits the small model's training script, trains for 5 minutes, and
  keeps or discards each change by validation loss.

A side track pre-trains a 0.56B bilingual model from scratch with
[nanochat](https://github.com/karpathy/nanochat). It covers the tokenizer, a
small scaling-law fit, data-mix ablations, and chained 8-hour jobs.

> **Status: P0 (environment and feasibility).** There are no results yet.
> Numbers are added only once they have been measured. Every job is in
> [docs/runs.md](docs/runs.md) and every failure in
> [docs/failures.md](docs/failures.md). The stage plan is in
> [docs/plan.md](docs/plan.md).

## Model lineage (planned)

```
Qwen3.5-4B-Base
  └─ M0  mid-training (finance + law + medicine + general replay + long-CoT anneal)
       ├─ S-fin SFT → R-fin RL ─┐
       ├─ S-law SFT → R-law RL ─┼─ teachers ─┐
       ├─ S-med SFT → R-med RL ─┘            │
       └─ S-mix SFT (all three) ──────────── MOPD ──► Trident-4B
             ├─ baseline: RL on the three-domain mix
             └─ baseline: TIES / DARE merge of R-fin, R-law, R-med
Side track: nanochat d20 (~0.56B, zh+en) from scratch → mid-training → SFT
```

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
| TRL | DPO, SimPO, KTO, ORPO; Bradley-Terry reward model |
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
| `src/b2e/` | The package: data pipeline, verifiers and rewards, evaluation, distillation routing, merging, lineage, serving |
| `pretrain/` | Side-track changes to nanochat (nanochat itself is cloned on the cluster at a pinned commit) |
| `configs/` | Hydra configs, one file per run |
| `tools/` | One-off checks (packing, token embeddings, chat template, train/inference log-prob gap) |
| `docs/` | Plan, run log, failures, results |

## Cluster constraints

The cluster is Northeastern's Explorer. Each GPU job gets one GPU for at most
8 hours, and each user gets at most 4 GPUs at once. The H200 driver supports
CUDA 12.8. Compute nodes have no internet. Long runs are therefore chained
8-hour legs that resume from checkpoints. Multi-GPU stages (multi-teacher
distillation) are assembled from several single-GPU jobs.
