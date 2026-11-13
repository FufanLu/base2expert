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

A side track pre-trains small bilingual models from scratch with
[nanochat](https://github.com/karpathy/nanochat) (depth 6 to 14, 0.14B to 0.66B
parameters). It covers the tokenizer, a scaling-law fit checked on a held-out
size, and a data-mix ablation.

> **Status (2026-11-12, closed):** the main line ran P0-P8; P9 (serving behind the gateway, the
> demo) was not run, serving being the vllm-sglang-infra project's subject. **P8's answer: RL on the
> three-domain mix from S-mix gives one model within about a point of every expert** (finance −0.80,
> law −0.62, medicine −0.53 points, paired); multi-teacher distillation, at the same GPU-hours,
> matches it on finance only. The side track trained the bilingual tokenizer, ran AutoResearch on d8,
> and fit a scaling law on d6-d12 that predicted d14's bpb within 0.7 %; at d12, up to 20 % domain
> data lowered the domain bpb without moving the general bpb. What was not run, and
> why, is in [docs/plan.md](docs/plan.md) § Closing. Only measured numbers appear below, each with its
> job in [docs/runs.md](docs/runs.md); every failure, including my own mistakes, is in
> [docs/failures.md](docs/failures.md).

## Model lineage

```
Qwen3.5-4B-Base
  └─ M0  mid-training (finance + law + medicine + general replay + long-CoT anneal)
       ├─ S-fin SFT → R-fin RL (DAPO, P6) ──┐
       │    └─ R-fin-low (low reasoning effort, per-prompt token budgets)
       ├─ S-law SFT → R-law RL (DAPO, P7) ──┼─ four teachers ─┐
       ├─ S-med SFT → R-med RL (DAPO, P7) ──┘                 │
       └─ S-mix SFT (all three) ─────────────────────────── MOPD (11.80 H200-hours)
             ├─ baseline: RL on the three-domain mix (10.78 H200-hours)   ← best single model
             └─ baseline: weight merges of R-fin, R-law, R-med (avg picked on val)
R-fin → self-written problems on real filings, validity-gated (P7.5): rounds 1-2 trained, round 3 proposed
Side track: nanochat from scratch (bilingual tokenizer, AutoResearch on d8, scaling d6-d14, d12 data mix)
```

## Results

**P8, fusion: one model in place of four teachers** (every arm starts from S-mix; each model run
through the scripts the experts ran; thinking, greedy, 8,192 tokens unless noted; mean answer
tokens in brackets; the final steps and evals were fixed on 11-09 before any arm's final number):

| Eval | S-mix (start) | merge_avg | MOPD | Mixed RL | Expert (ceiling) |
|---|---|---|---|---|---|
| FinQA test | 0.7855 (2,325) | 0.7864 (1,491) | 0.7995 (1,060) | 0.8012 (551) | R-fin 0.8134 (1,129) |
| TAT-QA test | 0.9066 (2,150) | 0.9256 (1,319) | 0.9350 (886) | 0.9418 (458) | R-fin 0.9432 (988) |
| FinQA test, low-effort message | 0.7847 (2,305) | 0.7820 (1,408) | 0.8003 (422) | 0.7960 (321) | R-fin-low 0.8126 (344) |
| TAT-QA test, low-effort message | 0.9134 (2,093) | 0.9188 (1,245) | 0.9323 (340) | 0.9364 (284) | R-fin-low 0.9391 (261) |
| CAIL2018 test | 0.6236 (2,471) | 0.6386 (1,533) | 0.6956 (1,620) | **0.7116** (1,338) | R-law 0.7178 (1,522) |
| MedQA test | 0.7636 (3,705) | 0.8201 (1,494) | 0.8013 (2,300) | 0.8350 (1,091) | R-med 0.8594 (2,097) |
| MedMCQA dev | 0.6644 (2,152) | 0.6829 (945) | 0.6538 (1,318) | 0.6875 (695) | R-med 0.6900 (1,316) |
| CMExam test | 0.7775 (1,725) | 0.7906 (1,237) | 0.7725 (1,399) | 0.7953 (1,088) | R-med 0.7982 (1,487) |
| General + IF, no thinking (10 sets, 19,509 items, pooled) | 0.7268 | 0.7306 | 0.7262 | **0.7339** | — |
| GSM8K, no thinking | 0.8870 | 0.9052 | 0.8650 | 0.8901 | — |
| P8 val (128 per teacher, of 512) | 387.2 | 394.2 | 389.8 | **401.0** | — |
| Training compute (sacct) | — | none (CPU merge) | 11.80 H200-hours | 10.78 H200-hours | — |

Paired differences (points ± se) that decide the reading:

| | Finance, thinking (2,624) | Low-effort (1,886) | Law + medicine, thinking (11,899) | General + IF (19,509) |
|---|---|---|---|---|
| MOPD − S-mix | +1.71 ± 0.45 | +1.70 ± 0.55 | +0.47 ± 0.35 (CAIL +7.20 ± 1.26) | −0.07 ± 0.26 (GSM8K −2.20 ± 0.80) |
| Mixed RL − S-mix | +2.02 ± 0.46 | +1.59 ± 0.55 | **+3.07 ± 0.34** (CAIL +8.80 ± 1.20, MedQA +7.15 ± 1.12) | +0.71 ± 0.25 |
| MOPD − expert | −1.11 ± 0.36 | −1.01 ± 0.46 | CAIL −2.22 ± 0.92; medicine −3.22 ± 0.36 | — |
| Mixed RL − expert | −0.80 ± 0.36 | −1.11 ± 0.49 | CAIL −0.62 ± 0.90; medicine −0.53 ± 0.35 | — |
| MOPD − mixed RL | −0.30 ± 0.35 | +0.11 ± 0.47 | **−2.60 ± 0.34** | −0.78 ± 0.26 |

- **Mixed RL is the one model to keep.** Pooled per domain it is within 1.1 points of each expert (the
  largest single-set gap is MedQA, −2.44 ± 1.03), it gains on general + IF without thinking (+0.71 ±
  0.25 over S-mix), and it answers in 1.6-4.7× fewer tokens than S-mix. On the low-effort message it
  writes 284-321 tokens, R-fin-low's length; in training those rows carried R-fin-low's budget reward.
- **MOPD ties mixed RL on finance and low effort, not elsewhere.** It took most of law (−2.2 from R-law)
  and little of medicine (MedQA −5.8 from R-med, 43 of 1,273 answers cut at the cap against R-med's 6),
  and without thinking it loses GSM8K (−2.20 ± 0.80 against S-mix). It moves furthest from the SFT
  text: bpb +0.03 to +0.04 on every held-out source and SFT val loss 2.5-4× S-mix's (mixed RL: bpb
  +0.002); why is not established (neither arm has a KL term to S-mix).
- **The plan expected MOPD to cost less than RL and to avoid the seesaw; here it did neither.** One MOPD
  variant (the sampled token's k3 as the reward, plain weighting) and one seed; the top-k KL and Open-MOPD weighting were
  smoked, not run at budget. Both arms used about the same GPU-hours, MOPD on two GPUs plus the four
  trained teachers.
- **MOPD's per-token signal was k3 used as a reward, which is not the reverse-KL gradient** (F#130,
  found 11-13 while tracing one question through every stage). With r = p_teacher / p_student,
  −k3 = log r + 1 − r: the extra −r punishes the tokens the teacher rates above the student, and the
  expected update is ∇KL(teacher‖student) − ∇KL(student‖teacher) instead of −∇KL(student‖teacher)
  (equal to 4e-16 on 2,000 random distributions; against the reverse-KL direction in 1,111 of them).
  verl's docs recommend `loss_mode=k1` with `use_policy_gradient`; its defaults, k3 with the policy
  gradient, are what P8 ran. The numbers above are measured and stand as "MOPD with k3 as the
  reward"; a k1 arm (`MOPD_LOSS=k1`) was not run, so whether it closes the gap to mixed RL is open.
- **Averaging the experts' weights costs no GPU** and gains on medicine (MedQA +5.66 ± 1.19 over S-mix),
  not on law (CAIL +1.50 ± 1.32) or low effort (R-fin-low is not in the merge, which writes 1,245-1,408
  tokens). Of the five merges, DARE at density 0.2 broke the model (378 of 512 val answers repeat until cut at the cap).

Jobs: arms' evals 11819217 (finance), 11819220 (low effort), 11819218 (general, IF, bpb, SFT val loss),
11819219 (law, medicine), 11819456 (P8 val); S-mix and merge_avg 11771666-11771671, 11804775, 11804776,
11728733; GPU-hours by sacct over the eight training legs (docs/runs.md, 11-11).

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
step 0, 0.672 at 30, 0.701 at 60). Third leg: 0.687 / 0.696 / 0.701 at steps 70 / 80 / 90; their mean,
0.6946, is 0.0099 over the mean at 40 / 50 / 60, and the rule fixed before training runs on to 150 only
above +0.0100, so law stops at step 90 (a miss of 0.06 of a row, well inside the noise: the rule decides). Medicine (600 rows): 0.767 at step 0, 0.803 at 60, 0.807 / 0.815 / 0.815 at 70 / 80 / 90; that mean, 0.8122, is 0.0172 over the one at 40 / 50 / 60 (CMExam +0.030, MedQA +0.023, MedMCQA −0.002), so medicine goes on to step 150. Steps 100 / 110 / 120: 0.800 / 0.812 / 0.803, a mean 0.0072 under 70-90 (4.3 rows, inside the noise); the plan holds. The control for F#92 measured what the example cost before RL: on CAIL
test's 339 articles questions S-law scores 0.490 with the prompt as trained and 0.596 / 0.575 with an
example drawn from the labels (`\boxed{133}`, `\boxed{196、264}`; paired, p ≈ 1e-5 each, unchanged without
the rows whose label holds the example), and writes general-part articles in 129 answers against 81 / 92. R-law, the same test: 0.758 / 0.767 / 0.767
(p 0.63, 0.58), general-part articles in 14 / 4 / 7: after RL the example no longer matters, and 0.162 of
R-law's 0.268 over S-law is beyond what a better example gives S-law. With thinking, R-law scores CAIL2018 test
0.718 against S-law's 0.542 at 1,522 tokens against 2,454, and medicine, untrained, holds (MedQA 0.797,
MedMCQA 0.669, CMExam 0.773 against 0.784 / 0.660 / 0.766); its law val rows score 0.7005, the training log's.

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
runs at 0.35 (9.1 % of step 30's rollouts over; step 31 had 9.0 %). After leg 2, val answers are 56-59 %
shorter than at step 0 (FinQA 0.8065 at 467 tokens, TAT-QA 0.9442 at 370; 445 of 512 right, 446 at step 0)
and the rollouts sit at 0.17× their budget; leg 3 runs at 0.25, which 3 % of step 60's rollouts exceed.
After leg 3, 444 of 512 right at step 90 (446 at step 0); 5.3 % of step 61's rollouts went over × 0.25 and
0.3 % of step 70's, and on shared prompts R-fin-low's answers are 0.24-0.27× R-fin's length at the median.
**Asked as trained (the low-effort message), R-fin-low scores FinQA test 0.813 at 344 tokens and TAT-QA test
0.939 at 261, against R-fin's 0.813 at 1,129 and 0.943 at 988 (paired p 1 and 0.61), and beats R-fin cut at
128 thinking tokens, as long or longer, 0.790 and 0.919 (p 0.0016 and 0.0026).** The message alone shortens R-fin
by 5-6 %.

**Rubric judge, P5**: Qwen3.6-35B-A3B grading per-item rubrics agrees with HealthBench physicians as well
as they agree with each other (class-balanced F1 0.686 against 0.689, difference 95 % [−0.021, +0.017]);
it replaces a Bradley-Terry reward model.

**Self-written problems, P7.5** (R-fin writes questions on real filings with a program for the answer;
an external validity gate; RL on the problems it sometimes solves): round 1 kept 5,688 of 10,000
proposals (2,638 duplicates); R-fin's pass rate left 1,128 mixed, the gate kept 346 (a hand audit of
20 found 1 flawed and 3 ambiguous), and 20 RL steps moved the val 444 → 447 of 512, inside the noise.
The fixed-pool control, 60 steps from the same start, stayed within 442-451 (R-fin is saturated on its
pool). Round 2 kept 4,181 proposals, 207 through the gate, and its RL left the val at 445 / 446 / 442
(steps 0 / 10 / 20, 11726394). Duplicates rose from 26 % to 41 % to 48 % of proposals over rounds 1-3,
and neither trained round moved the val beyond its noise.

**Pre-training side track, P2** (nanochat from scratch, bilingual 65,536-token tokenizer, data:param
ratio 12, one H200, one seed; bpb on the M0 held-out documents, 500 per source, which is
tokenizer-free and so comparable with Qwen; sizes, mixes and the held-out size fixed before any run,
11817898 and 11835552, 9.24 H200-hours):

| Depth (params) | d6 (0.14B) | d8 (0.23B) | d10 (0.34B) | d12 (0.49B) | d14 (0.66B), held out |
|---|---|---|---|---|---|
| Training FLOPs | 1.04e17 | 2.79e17 | 6.79e17 | 1.48e18 | 3.03e18 |
| Val bpb | 1.1351 | 1.0650 | 1.0051 | 0.9575 | **0.9201** (fit on d6-d12 predicts 0.9137) |
| CORE | 0.073 | 0.105 | 0.115 | 0.160 | 0.166 |

`log bpb = a + b log C` fits d6-d12 with b = −0.064 (largest residual 0.0009 in log) and misses d14
by +0.70 %; each of the eight held-out sources also lands above its prediction (+0.51 % to +1.01 %).
The local slope flattens with size (−0.065, −0.065, −0.062, −0.056), which a power law without a
constant term cannot follow, so its extrapolation is optimistic.

| d12, domain share of training bytes | 0 % | 5 % | 10 % | 20 % | Qwen3.5-0.8B-Base | Qwen3.5-4B-Base |
|---|---|---|---|---|---|---|
| Domain bpb (fin, law, med × en, zh; mean of 6) | 1.0479 | 0.9934 | 0.9777 | **0.9558** | 0.8438 | 0.7150 |
| General bpb (en, zh; mean of 2) | 1.0476 | 1.0444 | 1.0452 | 1.0442 | 0.9193 | 0.7755 |
| CORE | 0.153 | 0.160 | 0.148 | 0.144 | — | — |

Up to 20 % domain data the domain bpb keeps falling, most for the first 5 % (Chinese domain text
−0.123 at 20 %, English −0.062), and the general bpb stays within 0.0034. The 0 / 10 / 20 % builds
are 9 GB and re-read their last 5-9 % of steps; the 5 % build is 14 GB and does not (F#125). CORE
moves by at most 0.017 with 500 items a task, one seed, and is not read as a trend.

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
