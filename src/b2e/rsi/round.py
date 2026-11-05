"""The data steps of a self-improvement round between its GPU jobs (P7.5; docs/runs.md, 11-04).

The pilot (11594794-5, then the gate at 32,768 tokens: 11612731, 11612734, 11626562) fixed
the order a round runs in: the current model's pass rate first (b2e.data.synth, k 8), then
the validity gate (b2e.rsi.gate) on the mixed problems only. Usable is valid and mixed, two
conditions on the same problem, so the order does not change which problems are usable; it
changes what is paid, the gate being the dearer step (4.2 H200-hours for the pilot's 3,564
problems, of which 527 were mixed).

  halves   split a problems file's ids into n lists, for b2e.data.synth --only-ids on n GPUs
  join     concatenate shard outputs (pass-rate halves, gate shards) into one file, refusing
           a repeated id
  mixed    the problems right and wrong among the finished samples (rl_pool --finished), the
           ones the gate is run on
  pool     the round's RL data: the usable problems of one or more gated files (each with
           its pass rates when the gate saw more than the mixed), deduplicated by propose's
           rule in the order given, ids prefixed with their file's tag so they stay unique;
           rl_train.txt and rl_val.txt for cluster/rl.sbatch

    python -P -m b2e.rsi.round halves --problems $R/problems.parquet --n 2 --out $R/ids
    python -P -m b2e.rsi.round join --out $R/rfin_k8_passrate.parquet $R/rfin_k8_h0_passrate.parquet \\
        $R/rfin_k8_h1_passrate.parquet
    python -P -m b2e.rsi.round mixed --problems $R/problems.parquet --passrate $R/rfin_k8_passrate.parquet \\
        --out $R/mixed.parquet
    python -P -m b2e.rsi.round pool --out $B2E_ROOT/data/processed/rsi_r1 \\
        --val $B2E_ROOT/data/processed/finance/finqa_dev.parquet $B2E_ROOT/data/processed/finance/tatqa_dev.parquet \\
        --sources r1=$R/gate32k/problems_gated.parquet r0n=$P/gate32k/problems_gated.parquet,$P/rfin_k8_passrate.parquet
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from b2e.data.rl_pool import select
from b2e.rsi.docs import split_prompt
from b2e.rsi.propose import Taken, duplicate, shingles


def halves(ids: list[str], n: int) -> list[list[str]]:
    """ids i, i + n, ... for each of n lists."""
    return [ids[i::n] for i in range(n)]


def join(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    if "id" in df and df["id"].astype(str).duplicated().any():
        raise ValueError(f"ids in more than one shard: {df.loc[df['id'].astype(str).duplicated(), 'id'].head().tolist()}")
    return df


def dedup(rows: list[dict]) -> tuple[list[dict], int]:
    """rows without the problems that repeat an earlier one (propose.duplicate), and how many were dropped."""
    taken: Taken = {}
    out: list[dict] = []
    for r in rows:
        _, question = split_prompt(r["prompt"][0]["content"])
        value = float(json.loads(r["reward_model"]["ground_truth"])["value"])
        sh = shingles(question)
        past = taken.setdefault(r["extra_info"]["doc_id"], [])
        if duplicate(sh, value, past):
            continue
        past.append((sh, value))
        out.append(r)
    return out, len(rows) - len(out)


def usable(gated: pd.DataFrame, rates: pd.DataFrame | None) -> pd.DataFrame:
    """The gate's valid problems that are also mixed among finished samples; all of them when the
    gate saw only mixed problems (rates None)."""
    return gated if rates is None else select(gated, rates, 0.1, 0.9, finished=True)


def tagged(rows: list[dict], tag: str) -> list[dict]:
    return [{**r, "extra_info": {**r["extra_info"], "id": f"{tag}.{r['extra_info']['id']}"}} for r in rows]


def parse_source(spec: str) -> tuple[str, Path, Path | None]:
    tag, _, files = spec.partition("=")
    if not tag or not files:
        raise ValueError(f"{spec!r}: want tag=gated.parquet[,passrate.parquet]")
    gated, _, rates = files.partition(",")
    return tag, Path(gated), Path(rates) if rates else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("halves")
    h.add_argument("--problems", type=Path, required=True)
    h.add_argument("--n", type=int, default=2)
    h.add_argument("--out", type=Path, required=True, help="directory for ids_<i>of<n>.txt")
    j = sub.add_parser("join")
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("shards", nargs="+", type=Path)
    m = sub.add_parser("mixed")
    m.add_argument("--problems", type=Path, required=True)
    m.add_argument("--passrate", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pool")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val", nargs="+", required=True, help="validation parquet files, listed as they are")
    # named, not positional: after --val a positional list is taken as more val files (F#48)
    p.add_argument("--sources", nargs="+", required=True,
                   help="tag=gated.parquet[,passrate.parquet], earlier ones kept first")
    args = ap.parse_args(argv)

    if args.cmd == "halves":
        ids = [str(e["id"]) for e in pd.read_parquet(args.problems)["extra_info"]]
        args.out.mkdir(parents=True, exist_ok=True)
        for i, part in enumerate(halves(ids, args.n)):
            (args.out / f"ids_{i}of{args.n}.txt").write_text("\n".join(part) + "\n")
        print(json.dumps({"problems": len(ids), "n": args.n}))
    elif args.cmd == "mixed":
        problems, rates = pd.read_parquet(args.problems), pd.read_parquet(args.passrate)
        df = select(problems, rates, 0.1, 0.9, finished=True)
        df.to_parquet(args.out, index=False)
        print(json.dumps({"problems": len(problems), "scored": len(rates), "mixed": len(df)}))
    elif args.cmd == "join":
        df = join([pd.read_parquet(f) for f in args.shards])
        df.to_parquet(args.out, index=False)
        print(json.dumps({"rows": len(df), "shards": len(args.shards)}))
    else:
        rows: list[dict] = []
        summary: dict = {"sources": {}}
        for spec in args.sources:
            tag, gated, rates = parse_source(spec)
            g = pd.read_parquet(gated)
            u = usable(g, pd.read_parquet(rates) if rates is not None else None)
            rows += tagged(u.to_dict("records"), tag)
            summary["sources"][tag] = {"gated": str(gated), "passrate": str(rates) if rates else None,
                                       "valid": len(g), "usable": len(u)}
        rows, dropped = dedup(rows)
        if len({r["extra_info"]["id"] for r in rows}) != len(rows):
            raise ValueError("an id repeats within one source")
        args.out.mkdir(parents=True, exist_ok=True)
        dst = args.out / "rsi_train.parquet"
        pd.DataFrame(rows).to_parquet(dst, index=False)
        (args.out / "rl_train.txt").write_text(f"{dst.resolve()}\n")
        (args.out / "rl_val.txt").write_text("\n".join(str(Path(v).resolve()) for v in args.val) + "\n")
        by_tag = Counter(r["extra_info"]["id"].split(".", 1)[0] for r in rows)
        summary.update(duplicates_dropped=dropped, problems=len(rows), by_tag=dict(by_tag))
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
