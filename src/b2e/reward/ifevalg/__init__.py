"""Instruction-following verifiers, vendored from AllenAI's open-instruct.

Source: https://github.com/allenai/open-instruct, open_instruct/IFEvalG/
(instructions.py, instructions_registry.py, instructions_util.py) at commit
3bb265641634d8ebe274311271a0b0dfbdcdd81f, Apache License 2.0 (file headers
kept). The only change is the import path (open_instruct.IFEvalG ->
b2e.reward.ifevalg). They cover IFEval's 25 constraint types plus the
IFBench-Train ones used by allenai/IF_multi_constraints_upto5, so training
rewards and data filtering check constraints with the same code the data
was built for. b2e.reward.ifcheck is the entry point.

Needs nltk (with the punkt / punkt_tab data), langdetect, immutabledict and
absl-py; on the cluster the last two come from $B2E_ROOT/envs/addons-r1
(cluster/setup_addons.sbatch).
"""
