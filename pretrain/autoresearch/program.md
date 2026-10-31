# autoresearch: you are the researcher

Adapted from Karpathy's `program.md` (github.com/karpathy/autoresearch, MIT) for a loop in which you only
propose changes; a script applies them, runs the training and keeps or discards them.

## The setup

- `prepare.py` is fixed: the data (ClimbMix shards), the tokenizer (BPE, vocabulary 8192), the context
  length, the evaluation (`evaluate_bpb`) and the **time budget: 5 minutes of training**. You never see or
  change it.
- `train.py` is the only file that changes: the GPT model, the optimizer (Muon for matrices, AdamW for the
  rest) and the training loop. Everything in it is fair game: architecture, hyperparameters, optimizer,
  schedules, batch size, model size.
- The metric is **val_bpb** (validation bits per byte), lower is better. It does not depend on the
  vocabulary, so architectural changes compare fairly.
- One NVIDIA H200. The GPU is shared with the language model you are running on, so keep peak memory below
  about 60 GB; an out-of-memory crash wastes a round.
- No new packages: only what `train.py` already imports, plus the Python standard library.

## Each round

You see the current `train.py` (the best version so far) and `results.tsv`: one row per training run
(commit, val_bpb, peak memory in GB, status, description). Statuses: `keep` (the baseline's first run, or a
kept change), `noise` (the unchanged baseline trained again, to measure run-to-run noise), `discard`,
`crash`, `invalid` (the edit could not be applied); and for a change whose first run looked better,
`screen` (that first run), `confirm` (the change trained again), `recheck` (the best version trained again
alongside it), then one decision row, `kept` or `rejected`, with the change's mean. Propose **one**
experiment: one idea, so that its effect can be read from its runs.

The script then commits your change and trains it. Two runs of the same code differ (the budget is
wall-clock, so the number of steps varies), so a change is kept only if it beats the best by more than
that noise: when its first run is clearly better, it is trained twice more, alternating with the best
version, and kept only if its mean beats the best's by two standard errors. Otherwise the file goes back
to the best version. A crash gets one repair attempt. Small effects cannot be told from noise; prefer
ideas that could move val_bpb by a few thousandths.

## How to answer

First a line

    DESCRIPTION: <what you change and why, one line>

then one or more blocks, each replacing a piece of the current `train.py`:

    <<<<<<< SEARCH
    <lines copied exactly from the current train.py>
    =======
    <the new lines>
    >>>>>>> REPLACE

Each SEARCH text must appear exactly once in the file, character for character, including indentation.
Include enough surrounding lines to make it unique. Blocks apply in order.

## How to choose

- Read `results.tsv` before deciding. Do not repeat an experiment that was already discarded, unless you
  change it in a way that answers why it failed.
- The time budget is fixed, so a change that makes each step faster lets the model see more tokens; a
  bigger model sees fewer tokens. Think about that trade before changing size.
- Simpler is better. A tiny gain that adds a lot of complicated code is not worth keeping; removing code
  with an equal result is a win.
- Good directions: learning rates and their schedule (warm-up, warm-down, final fraction), batch size,
  depth and width, attention window pattern, normalization and initialization details, optimizer settings.
  If small tweaks stop helping, try a structural idea.
