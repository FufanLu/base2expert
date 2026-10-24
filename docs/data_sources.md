# Data sources

Candidate datasets with their licenses as reported by the Hugging Face API
(`cardData.license` and `gated`), checked 2026-10-24. "—" means the card
declares no license; such sets are used only for research and are flagged in
the manifest. Sets marked *not found* returned 401 (the repo id is wrong or
private) and still need their real location.

## Mid-training corpora

| Dataset | Use | License | Gated |
|---|---|---|---|
| BAAI/IndustryCorpus2_finance_economics | finance text, zh+en, quality tiers | Apache-2.0 | no |
| BAAI/IndustryCorpus2_law_judiciary | law text, zh+en | Apache-2.0 | no |
| BAAI/IndustryCorpus2_medicine_health_psychology_traditional_chinese_medicine | medical text, zh+en | Apache-2.0 | no |
| HuggingFaceFW/fineweb-edu | general replay (en) | ODC-BY | no |
| opencsg/Fineweb-Edu-Chinese-V2.1 | general replay (zh) | Apache-2.0 | no |
| open-thoughts/OpenThoughts3-1.2M | long chain-of-thought anneal | Apache-2.0 | no |

## SFT / RL (with verifiers)

| Dataset | Domain | Answer type | License |
|---|---|---|---|
| ibm-research/finqa | finance | numeric | CC-BY-4.0 |
| next-tat/TAT-QA | finance | numeric / span | CC-BY-4.0 |
| ConvFinQA | finance | numeric (multi-turn) | *not found* on the Hub under TheFinAI/ConvFinQA; original is on GitHub |
| Fin-R1 data | finance | mixed | *not found* under SUFE-AIFLM-Lab/Fin-R1-Data |
| DianJin/DianJin-R1-Data | finance (zh) | mixed | MIT |
| china-ai-law-challenge/cail2018 | law (zh) | charges, articles, prison term | unknown |
| ShengbinYue/DISC-Law-SFT | law (zh) | SFT pairs | Apache-2.0 |
| GBaker/MedQA-USMLE-4-options | medicine (en) | multiple choice | CC-BY-4.0 |
| openlifescienceai/medmcqa | medicine (en) | multiple choice | Apache-2.0 |
| fzkuji/CMExam | medicine (zh) | multiple choice | Apache-2.0 |
| FreedomIntelligence/medical-o1-verifiable-problem | medicine | verifiable open answer | Apache-2.0 |
| FreedomIntelligence/medical-o1-reasoning-SFT | medicine | CoT SFT | Apache-2.0 |
| ScaleAI/RaR-Medicine | medicine | rubric | — |
| allenai/IF_multi_constraints_upto5 | instruction following | constraint checkers | ODC-BY |
| allenai/tulu-3-sft-mixture | general SFT | — | ODC-BY |
| HuggingFaceTB/smoltalk2 | general SFT | — | — |

## Preference, reward-model and eval sets

| Dataset | Use | License |
|---|---|---|
| nvidia/HelpSteer3 | preference pairs | CC-BY-4.0 |
| allenai/reward-bench-2 | reward-model eval | ODC-BY |
| google/IFEval | eval | Apache-2.0 |
| allenai/IFBench_test | eval | — |
