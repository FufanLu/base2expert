"""Clean one corpus with datatrove, the pipeline Hugging Face built FineWeb with.

Steps, in order:
  1. read parquet (text + the source's own quality signals as metadata)
  2. length filter, and a minimum quality_score when the source provides one
     (BAAI IndustryCorpus2 does)
  3. PII scrub (b2e.data.pii: e-mail, IP, phone, checksum-valid ID and card numbers)
  4. 13-gram decontamination against every eval split (b2e.data.decontam)
  5. MinHash near-duplicate removal, datatrove's four stages:
     signatures -> buckets -> clusters -> filter
  6. write parquet

datatrove shingles MinHash text by words from a per-language word tokenizer.
For Chinese ('zho') datatrove 0.10.1 uses a whitespace tokenizer
(tools/datatrove_probe.py, job 11347236): a 35-character sentence comes out
as 2 "words" (the sentence and its full stop), so 5-word shingles barely
exist and Chinese near-duplicates would pass silently. The signature step
here uses the same units as decontamination instead, English words and
single CJK characters, so zh and en shards are shingled alike.

    python -P -m b2e.data.clean_pipeline index --eval-texts eval_texts.jsonl --out eval_13gram.npy
    python -P -m b2e.data.clean_pipeline clean --input DIR --glob 'chinese/high/*.parquet' --name fin_zh \\
        --index eval_13gram.npy --work WORK --out OUT --workers 16
"""

from __future__ import annotations

import argparse
import json
import os

from datatrove.data import Document
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.formatters.base import BaseFormatter

from b2e.data.decontam import FrozenNgramIndex, units
from b2e.data.pii import scrub


class LengthQualityFilter(BaseFilter):
    name = "📏 length / quality"

    def __init__(self, min_chars: int = 200, max_chars: int = 200_000, min_quality: float | None = None,
                 exclusion_writer=None):
        super().__init__(exclusion_writer)
        self.min_chars, self.max_chars, self.min_quality = min_chars, max_chars, min_quality

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        n = len(doc.text)
        if n < self.min_chars:
            return False, "too_short"
        if n > self.max_chars:
            return False, "too_long"
        q = doc.metadata.get("quality_score")
        if self.min_quality is not None and q is not None and float(q) < self.min_quality:
            return False, "low_quality"
        return True


class ScrubPII(BaseFormatter):
    name = "🔒 PII scrub (zh/en)"

    def format(self, text: str) -> str:
        out, counts = scrub(text)
        for kind, c in counts.items():
            if c:
                self.stat_update(f"pii_{kind}", value=c)
        return out


class Decontaminate(BaseFilter):
    name = "🧪 13-gram decontamination"

    def __init__(self, index_path: str, n: int = 13, exclusion_writer=None):
        super().__init__(exclusion_writer)
        self.index_path, self.n = index_path, n
        self._index: FrozenNgramIndex | None = None

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if self._index is None:  # loaded in the worker process, shared via mmap
            self._index = FrozenNgramIndex(self.index_path, self.n)
        return True if self._index.hits(doc.text) == 0 else (False, "eval_overlap")


class UnitWordTokenizer:
    """datatrove word-tokenizer interface over b2e.data.decontam.units."""

    def word_tokenize(self, text: str) -> list[str]:
        return units(text)


class MixedLangMinhashSignature(MinhashDedupSignature):
    """MinHash signatures over English words and single CJK characters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_tokenizer = UnitWordTokenizer()


def build_index(eval_texts: str, out: str) -> None:
    def texts():
        with open(eval_texts) as f:
            for line in f:
                yield json.loads(line)["text"]

    n = FrozenNgramIndex.build(texts(), out)
    print(json.dumps({"index": out, "ngrams": n, "bytes": os.path.getsize(out)}))


def clean(args) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup.minhash import (
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter
    from datatrove.utils.hashing import HashConfig

    w = args.work
    mh = MinhashConfig(hash_config=HashConfig(precision=64), num_buckets=14, hashes_per_bucket=8, n_grams=5)

    def reader():
        return ParquetReader(args.input, glob_pattern=args.glob, text_key="text", id_key=args.id_key,
                             limit=args.limit, default_metadata={"source": args.name})

    stage1 = LocalPipelineExecutor(
        pipeline=[
            reader(),
            LengthQualityFilter(args.min_chars, args.max_chars, args.min_quality,
                                exclusion_writer=ParquetWriter(f"{w}/removed/length_quality")),
            ScrubPII(),
            Decontaminate(args.index, exclusion_writer=ParquetWriter(f"{w}/removed/decontam")),
            ParquetWriter(f"{w}/filtered"),
        ],
        tasks=args.workers, workers=args.workers, logging_dir=f"{w}/logs/filter",
    )
    stage2 = LocalPipelineExecutor(
        pipeline=[ParquetReader(f"{w}/filtered", text_key="text"),
                  MixedLangMinhashSignature(output_folder=f"{w}/minhash/signatures", config=mh)],
        tasks=args.workers, workers=args.workers, logging_dir=f"{w}/logs/signatures", depends=stage1,
    )
    stage3 = LocalPipelineExecutor(
        pipeline=[MinhashDedupBuckets(input_folder=f"{w}/minhash/signatures",
                                      output_folder=f"{w}/minhash/buckets", config=mh)],
        tasks=mh.num_buckets, workers=min(args.workers, mh.num_buckets), logging_dir=f"{w}/logs/buckets",
        depends=stage2,
    )
    stage4 = LocalPipelineExecutor(
        pipeline=[MinhashDedupCluster(input_folder=f"{w}/minhash/buckets",
                                      output_folder=f"{w}/minhash/remove_ids", config=mh)],
        tasks=1, logging_dir=f"{w}/logs/clusters", depends=stage3,
    )
    stage5 = LocalPipelineExecutor(
        pipeline=[ParquetReader(f"{w}/filtered", text_key="text"),
                  MinhashDedupFilter(input_folder=f"{w}/minhash/remove_ids",
                                     exclusion_writer=ParquetWriter(f"{w}/removed/minhash")),
                  ParquetWriter(args.out)],
        tasks=args.workers, workers=args.workers, logging_dir=f"{w}/logs/dedup_filter", depends=stage4,
    )
    stage5.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--eval-texts", required=True)
    a.add_argument("--out", required=True)
    c = sub.add_parser("clean")
    c.add_argument("--input", required=True)
    c.add_argument("--glob", default="*.parquet")
    c.add_argument("--name", required=True)
    c.add_argument("--index", required=True)
    c.add_argument("--work", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--min-chars", type=int, default=200)
    c.add_argument("--max-chars", type=int, default=200_000)
    c.add_argument("--min-quality", type=float, default=None)
    c.add_argument("--id-key", default="_id")
    c.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    if args.cmd == "index":
        build_index(args.eval_texts, args.out)
    else:
        clean(args)


if __name__ == "__main__":
    main()
