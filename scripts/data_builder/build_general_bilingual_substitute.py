"""Build the general_bilingual component substitute from FineWeb-Edu slices.

The original component came from the user-supplied pretrain_t2t.jsonl (10GB,
documented in README as a separate download). That file is not available on
this machine and no public mirror was found, so this rebuild substitutes the
same role: whole bilingual general-web documents, EN+ZH interleaved, whole
records only, <=4096 project tokens. Marked as a substitution in the report;
re-run against a restored pretrain_t2t.jsonl for a byte-faithful mix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401  # before transformers on Windows
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_builder.collect_coder_pretrain_12b import clean_text

SOURCE = "general_bilingual"
TARGET_TOKENS = 330_000_000
MAX_TOKENS = 4096


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-raw", type=Path, nargs="*", default=[
        ROOT / "dataset/coder_pretrain_12b/natural_en/raw/fineweb-edu-dedup/train-00000-of-00234.parquet",
        ROOT / "dataset/coder_pretrain_12b/natural_en/raw/fineweb-edu-dedup/train-00037-of-00234.parquet",
    ])
    parser.add_argument("--zh-raw", type=Path, nargs="*", default=[
        ROOT / "dataset/coder_pretrain_12b/natural_zh/raw/IndustryCorpus/00000.parquet",
        ROOT / "dataset/coder_pretrain_12b/natural_zh/raw/IndustryCorpus/00037.parquet",
    ])
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/pretrain_general_bilingual_substitute_330m.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "dataset/pretrain_general_bilingual_substitute_330m.report.json")
    return parser.parse_args()


def iter_rows(paths):
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                yield path, row


def pick(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("output/report already exists")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    backend = tokenizer.backend_tokenizer
    stats = Counter()
    seen = set()
    streams = {
        "en": iter_rows(args.en_raw),
        "zh": iter_rows(args.zh_raw),
    }
    budgets = {"en": TARGET_TOKENS * 55 // 100, "zh": TARGET_TOKENS * 45 // 100}
    tokens = {"en": 0, "zh": 0}
    rows = 0
    licenses = Counter()
    try:
        with args.output.open("xb", buffering=8 * 1024 * 1024) as destination:
            active = True
            while active:
                active = False
                for language in ("en", "zh"):
                    if tokens[language] >= budgets[language] - MAX_TOKENS:
                        continue
                    active = True
                    text = None
                    while True:
                        try:
                            path, row = next(streams[language])
                        except StopIteration:
                            stats[f"{language}:exhausted"] += 1
                            break
                        stats["scanned"] += 1
                        candidate = pick(row, "text", "content")
                        if not clean_text(candidate):
                            stats[f"{language}:text_quality"] += 1
                            continue
                        fingerprint = hashlib.blake2b(candidate.encode("utf-8"), digest_size=16).digest()
                        if fingerprint in seen:
                            stats[f"{language}:duplicate_text"] += 1
                            continue
                        token_count = len(backend.encode(candidate.strip(), add_special_tokens=False).ids) + 2
                        if not 64 <= token_count <= MAX_TOKENS:
                            stats[f"{language}:outside_token_range"] += 1
                            continue
                        if tokens[language] + token_count > budgets[language]:
                            stats[f"{language}:over_budget"] += 1
                            continue
                        text = candidate.strip()
                        record = {
                            "text": text,
                            "source": SOURCE,
                            "source_id": str(pick(row, "id", "docid") or f"{path.name}:{stats['scanned']}"),
                            "license": "ODC-By-1.0" if language == "en" else "Apache-2.0; review upstream source terms",
                            "url": str(pick(row, "url")),
                            "token_count": token_count,
                            "language": "en" if language == "en" else "zh",
                            "substitution_for": "pretrain_t2t.jsonl",
                        }
                        destination.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                        seen.add(fingerprint)
                        tokens[language] += token_count
                        rows += 1
                        licenses[record["license"]] += 1
                        break
                if rows % 50_000 < 2:
                    print(f"rows={rows:,} en={tokens['en']:,} zh={tokens['zh']:,}", flush=True)
    except BaseException:
        if args.output.exists() and rows == 0:
            args.output.unlink()
        raise

    total = tokens["en"] + tokens["zh"]
    report = {
        "name": "general_bilingual_substitute",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "substitution_for": "pretrain_t2t.jsonl (original general_bilingual source; not available locally, no public mirror found)",
        "output": str(args.output), "rows": rows, "tokens": total,
        "tokens_en": tokens["en"], "tokens_zh": tokens["zh"],
        "target_tokens": TARGET_TOKENS,
        "selection_stats": dict(sorted(stats.items())),
        "license_row_counts": dict(licenses),
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "whole_records_only": True, "text_split": False, "text_truncated": False,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest() if args.output.stat().st_size < 2**31 else "too-large-for-in-memory-hash",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
