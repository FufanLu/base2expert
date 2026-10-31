"""The research loop of Karpathy's autoresearch, driven by a local model (P2.5).

autoresearch fixes the data, tokenizer, time budget (5 min of training) and
the metric (val_bpb) in prepare.py; only train.py may change. Its own loop is
run by a coding agent that edits files and runs shell commands. Here the loop
itself is code and the model only proposes: every round

  1. the proposer gets program.md, the current train.py and results.tsv, and
     answers with a DESCRIPTION line and SEARCH/REPLACE blocks (edit.py);
  2. the edit is applied and committed on the run's branch;
  3. train.py runs (10-minute hard timeout) and its summary is parsed;
  4. val_bpb lower than the best so far by more than --min-gain: keep the
     commit; otherwise (worse, equal, crash, invalid edit) reset to the best
     commit;
  5. a row goes to results.tsv and the full exchange to rounds/NNN.json.

A crash gets one repair attempt with the end of the log. The proposer is
either the model (--proposer llm, an OpenAI-compatible endpoint, e.g. vLLM
serving Qwen3.6-35B-A3B on the same GPU) or the baseline (--proposer random):
one hyperparameter constant perturbed at random, under the same keep / reset
rule, so the comparison isolates what the model's ideas are worth. It draws
without replacement (a change already in results.tsv is not tried again),
because the model is told not to repeat discarded experiments (F#36).

The time budget is wall-clock, so the number of steps, and val_bpb, vary from
run to run of the same code: the first stretch's accidental repeats differed
by up to 0.0012. --noise-runs trains the unchanged baseline a few more times
(status `noise`) to measure that spread, and --min-gain keeps a change only
if it beats the best by more than the spread.

The spread itself differs from node to node (0.00025 on one, 0.0039 on
another: 11360201, 11378056), so one fixed margin is too strict on a quiet
node and too loose on a noisy one, and a single run that beats the best can
be luck. --confirm K replaces step 4: the best is the mean of all its runs; a
candidate whose first run beats it by --screen-sd run-to-run standard
deviations (status `screen`) is trained K more times, alternating with the
best (`confirm`, `recheck`), and is kept only if its mean beats the mean of
those best runs by two standard errors of the difference, and by at least
--min-gain (`kept`, else `rejected`; that row carries the candidate's mean).
The standard deviation is pooled over every commit trained more than once in
this run. runs.jsonl keeps every run's full summary (steps, MFU, seconds).

    python -P pretrain/autoresearch/agent.py --repo $AR/repo --python $NC_ENV/bin/python \\
        --proposer llm --llm-url http://127.0.0.1:8000 --llm-model qwen3.6 --rounds 15
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit import EditError, apply, parse  # noqa: E402

HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
SUMMARY_KEYS = ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent", "total_tokens_M",
                "num_steps", "num_params_M", "depth")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def summary(log: str) -> dict:
    out = {}
    for k in SUMMARY_KEYS:
        m = re.search(rf"^{k}:\s+(\S+)", log, re.M)
        if m:
            out[k] = float(m.group(1))
    return out


def run_train(repo: Path, python: str, timeout_s: int) -> tuple[dict, str]:
    log = repo / "run.log"
    try:
        with log.open("w") as f:
            subprocess.run([python, "train.py"], cwd=repo, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {}, f"timeout after {timeout_s} s"
    text = log.read_text(errors="replace")
    return summary(text), "\n".join(text.splitlines()[-50:])


# --- proposers --------------------------------------------------------------

def llm_propose(args, program: str, source: str, results: str, note: str) -> str:
    import httpx

    user = (f"Current train.py:\n```python\n{source}\n```\n\nresults.tsv so far:\n```\n{results}```\n\n{note}"
            "Propose ONE experiment. Answer with a line `DESCRIPTION: ...` and SEARCH/REPLACE blocks against the "
            "current train.py, as program.md describes.")
    r = httpx.post(f"{args.llm_url}/v1/chat/completions", timeout=900, json={
        "model": args.llm_model, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "messages": [{"role": "system", "content": program}, {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": True}})
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # without --reasoning-parser the reasoning stays in content; with one, vLLM names the field
    # reasoning_content or (newer) reasoning. edit.parse reads only what follows the last </think>.
    return (msg.get("reasoning_content") or msg.get("reasoning") or "") + "</think>" + (msg.get("content") or "")


# constant name -> candidate multipliers or values, for the random baseline
RANDOM_SPACE = {
    "MATRIX_LR": [0.5, 0.7, 1.4, 2.0], "EMBEDDING_LR": [0.5, 0.7, 1.4, 2.0], "UNEMBEDDING_LR": [0.5, 0.7, 1.4, 2.0],
    "SCALAR_LR": [0.5, 2.0], "WEIGHT_DECAY": [0.0, 0.5, 2.0], "WARMDOWN_RATIO": ["0.3", "0.4", "0.6", "0.7"],
    "WARMUP_RATIO": ["0.0", "0.02", "0.05"], "DEPTH": ["6", "7", "9", "10"], "TOTAL_BATCH_SIZE": ["2**18", "2**20"],
    "ASPECT_RATIO": ["48", "80", "96"],
}


def _num(text: str) -> float:
    """Value of a constant written as a number or as a power such as 2**19."""
    base, _, exp = text.partition("**")
    return float(base) ** float(exp) if exp else float(base)


def random_propose(rng: random.Random, source: str, tried: set[str]) -> str | None:
    """One untried change: a constant drawn uniformly, then one of its untried values; None when all are tried."""
    options: dict[str, list[tuple[str, str]]] = {}
    for name in sorted(RANDOM_SPACE):
        m = re.search(rf"^({name}\s*=\s*)([^#\n]+?)(\s*(#.*)?)$", source, re.M)
        if not m:
            continue
        old = m.group(2).strip()
        for choice in RANDOM_SPACE[name]:
            new = f"{_num(old) * choice:.6g}" if isinstance(choice, float) else choice
            desc = f"(random) {name} {old} -> {new}"
            if new != old and desc not in tried:
                block = f"<<<<<<< SEARCH\n{m.group(0)}\n=======\n{m.group(1)}{new}{m.group(3)}\n>>>>>>> REPLACE\n"
                options.setdefault(name, []).append((desc, block))
    if not options:
        return None
    desc, block = rng.choice(options[rng.choice(sorted(options))])
    return f"DESCRIPTION: {desc}\n{block}"


# --- the keep rule of --confirm ---------------------------------------------

# results.tsv statuses that are one training run each; `kept` / `rejected` rows
# are --confirm's decisions and carry the candidate's mean.
RUN_STATUSES = ("keep", "noise", "discard", "screen", "confirm", "recheck")
# statuses that end a round (or are the baseline's runs), so resuming counts rounds
ROUND_STATUSES = ("keep", "noise", "discard", "crash", "invalid", "kept", "rejected")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("inf")


def runs_by_commit(tsv_text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in tsv_text.splitlines()[1:]:
        f = row.split("\t")
        if f[3] in RUN_STATUSES and float(f[1]) > 0:
            out.setdefault(f[0], []).append(float(f[1]))
    return out


def pooled_sd(groups: dict[str, list[float]]) -> float | None:
    """Run-to-run standard deviation of val_bpb, pooled over the commits trained more than once."""
    ss = dof = 0.0
    for v in groups.values():
        if len(v) > 1:
            m = mean(v)
            ss += sum((x - m) ** 2 for x in v)
            dof += len(v) - 1
    return math.sqrt(ss / dof) if dof else None


def keep_margin(sd: float | None, n_cand: int, n_best: int, min_gain: float) -> float:
    """How far below the best's mean a candidate's mean must be: two standard errors, and min_gain at least."""
    return max(min_gain, 2 * sd * math.sqrt(1 / n_cand + 1 / n_best)) if sd else min_gain


# --- loop -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="autoresearch checkout on the run's branch")
    ap.add_argument("--python", required=True, help="interpreter of the env that runs train.py")
    ap.add_argument("--program", type=Path, default=Path(__file__).with_name("program.md"))
    ap.add_argument("--proposer", choices=["llm", "random"], required=True)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--llm-model", default="qwen3.6")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--hours", type=float, default=0, help="stop starting new rounds after this long (0 = no limit)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-runs", type=int, default=0, help="extra runs of the unchanged baseline (new run only)")
    ap.add_argument("--min-gain", type=float, default=0.0, help="keep only if val_bpb < best - min_gain")
    ap.add_argument("--confirm", type=int, default=0, help="re-runs of a candidate that passes the screen (0: one run)")
    ap.add_argument("--screen-sd", type=float, default=1.0,
                    help="with --confirm: standard deviations below the best's mean the first run must be")
    args = ap.parse_args()

    repo, rng, t0 = args.repo, random.Random(args.seed), time.time()
    tsv, rounds_dir = repo / "results.tsv", repo / "rounds"
    rounds_dir.mkdir(exist_ok=True)
    if not tsv.exists():
        tsv.write_text(HEADER)
    program = args.program.read_text()
    done = sum(1 for r in tsv.read_text().splitlines()[1:] if r.split("\t")[3] in ROUND_STATUSES)

    def record(commit: str, s: dict, status: str, desc: str) -> None:
        with tsv.open("a") as f:
            f.write(f"{commit}\t{s.get('val_bpb', 0.0):.6f}\t{s.get('peak_vram_mb', 0.0) / 1024:.1f}\t{status}\t"
                    f"{desc.replace(chr(9), ' ')}\n")
        with (repo / "runs.jsonl").open("a") as f:
            f.write(json.dumps({"commit": commit, "status": status, "description": desc, **s}) + "\n")

    def best_mean(commit: str) -> float:
        return mean(runs_by_commit(tsv.read_text()).get(commit, []))

    best_commit = git(repo, "rev-parse", "--short=7", "HEAD")
    best = best_mean(best_commit) if args.confirm else min(
        (float(r.split("\t")[1]) for r in tsv.read_text().splitlines()[1:] if r.split("\t")[3] == "keep"),
        default=float("inf"))
    if done == 0:
        s, tail = run_train(repo, args.python, args.timeout)
        if "val_bpb" not in s:
            raise SystemExit(f"baseline run failed:\n{tail}")
        record(best_commit, s, "keep", "baseline")
        best, done = s["val_bpb"], 1
        print(f"baseline val_bpb {best:.6f}", flush=True)
        for i in range(args.noise_runs):
            s, _ = run_train(repo, args.python, args.timeout)
            record(best_commit, s, "noise", f"baseline repeat {i + 1}")
            done += 1
            print(f"baseline repeat {i + 1}: val_bpb {s.get('val_bpb', 0):.6f}", flush=True)
        if args.confirm:
            best = best_mean(best_commit)
    tried = {r.split("\t")[4] for r in tsv.read_text().splitlines()[1:]}

    note = ""
    while done < args.rounds and not (args.hours and time.time() - t0 > args.hours * 3600):
        source = (repo / "train.py").read_text()
        exchange: dict = {"round": done, "proposer": args.proposer, "best_before": best}
        answer = (llm_propose(args, program, source, tsv.read_text(), note) if args.proposer == "llm"
                  else random_propose(rng, source, tried))
        if answer is None:
            print("random search space exhausted", flush=True)
            break
        exchange["answer"] = answer
        try:
            desc, blocks = parse(answer)
            new_source = apply(source, blocks)
        except EditError as e:
            record(best_commit, {}, "invalid", f"{e}")
            note = f"Your previous answer could not be applied: {e}. Copy SEARCH text exactly from the file.\n\n"
            exchange["status"] = "invalid"
            (rounds_dir / f"{done:03d}.json").write_text(json.dumps(exchange, indent=1))
            done += 1
            continue
        (repo / "train.py").write_text(new_source)
        git(repo, "commit", "-qam", desc or "experiment")
        commit = git(repo, "rev-parse", "--short=7", "HEAD")
        s, tail = run_train(repo, args.python, args.timeout)
        if "val_bpb" not in s and args.proposer == "llm":  # one repair attempt
            fix = llm_propose(args, program, new_source, tsv.read_text(),
                              f"Your last change crashed. End of the log:\n```\n{tail}\n```\nFix the crash with "
                              "SEARCH/REPLACE blocks; keep the idea.\n\n")
            exchange["repair"] = fix
            try:
                _, fb = parse(fix)
                (repo / "train.py").write_text(apply(new_source, fb))
                git(repo, "commit", "-qam", f"fix: {desc}")
                commit = git(repo, "rev-parse", "--short=7", "HEAD")
                s, tail = run_train(repo, args.python, args.timeout)
            except EditError:
                pass
        row_desc = desc
        sd = pooled_sd(runs_by_commit(tsv.read_text()))
        if "val_bpb" not in s:
            status = "crash"
        elif not args.confirm:
            status = "keep" if s["val_bpb"] < best - args.min_gain else "discard"
        elif s["val_bpb"] >= best - args.screen_sd * (sd or 0.0):
            status = "discard"
        else:  # the screen passed: train the candidate and the best alternately
            record(commit, s, "screen", desc)
            cand, base = [s["val_bpb"]], []
            for _ in range(args.confirm):
                for c, vals, st, d in ((best_commit, base, "recheck", f"best again, alongside: {desc}"),
                                       (commit, cand, "confirm", desc)):
                    git(repo, "checkout", "-q", c, "--", "train.py")
                    r, tail = run_train(repo, args.python, args.timeout)
                    record(c, r, st, d)
                    if "val_bpb" in r:
                        vals.append(r["val_bpb"])
            sd = pooled_sd(runs_by_commit(tsv.read_text()))
            margin = keep_margin(sd, len(cand), len(base), args.min_gain)
            complete = len(cand) == args.confirm + 1 and len(base) == args.confirm
            status = "kept" if complete and mean(cand) < mean(base) - margin else "rejected"
            row_desc = (f"{desc} | mean of {len(cand)} runs vs {mean(base):.6f} for the best over {len(base)} runs "
                        f"alongside; margin {margin:.6f}")
            exchange["confirm"] = {"candidate": cand, "best": base, "sd": sd, "margin": margin}
            s = {"val_bpb": mean(cand), "peak_vram_mb": s.get("peak_vram_mb", 0.0)}
        record(commit, s, status, row_desc)
        tried.add(desc)
        exchange.update({"description": desc, "commit": commit, "summary": s, "status": status,
                         "diff": git(repo, "diff", f"{best_commit}..{commit}", "--", "train.py")})
        if status in ("keep", "kept"):
            best_commit = commit
        else:
            git(repo, "reset", "-q", "--hard", best_commit)
        if args.confirm:
            best = best_mean(best_commit)
        elif status == "keep":
            best = s["val_bpb"]
        note = f"The last experiment ({desc}) ended as {status}" + (f": {tail[-800:]}" if status == "crash" else ".")
        if status in ("kept", "rejected"):
            note += f" Its first run beat the best, so it was trained again alternating with the best: {row_desc}."
        note += "\n\n"
        (rounds_dir / f"{done:03d}.json").write_text(json.dumps(exchange, indent=1))
        print(f"round {done}: {status} {s.get('val_bpb', 0):.6f} (best {best:.6f}) {desc}", flush=True)
        done += 1


if __name__ == "__main__":
    main()
