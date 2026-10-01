"""Assemble the final 12B-token, DeepSeek-Coder-V2-style corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import orjson
from datasets import load_dataset  # noqa: F401  # before transformers on Windows
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_builder.coder_pretrain_common import (
    BASE, CONTINUED, CONTINUED_SOURCES, MAX_TOKENS, SEED,
    benchmark_ngrams, classify_base, classify_continued,
    has_benchmark_overlap,
)
from scripts.data_builder.collect_coder_pretrain_12b import OUT as COLLECTED
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler


OUTPUT = ROOT / "dataset/pretrain_coder_12b.jsonl"
REPORT = ROOT / "dataset/pretrain_coder_12b.report.json"
TOKENIZER = ROOT / "model"
TARGETS = {
    "code": 7_200_000_000,
    "math": 1_200_000_000,
    "natural": 3_600_000_000,
}
SHUFFLE_SEED = SEED + 12_000_000_000
SHUFFLE_CHUNK_ROWS = 100_000
SHUFFLE_CHUNK_BYTES = 512 * 1024 ** 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reshuffle-existing", action="store_true",
        help=(
            "Atomically reshuffle the existing pretrain_coder_12b.jsonl "
            "without rebuilding or re-encoding its records."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--shuffle-chunk-rows", type=int, default=SHUFFLE_CHUNK_ROWS)
    parser.add_argument(
        "--shuffle-chunk-mb", type=int,
        default=SHUFFLE_CHUNK_BYTES // 1024 ** 2,
        help="Approximate in-memory cap for each external-sort run.",
    )
    parser.add_argument("--audit-windows", type=int, default=64)
    parser.add_argument("--audit-window-rows", type=int, default=1024)
    return parser.parse_args()


def audit_contiguous_mixing(path, *, windows=64, window_rows=1024):
    """Sample deterministic byte-spaced windows and reject block ordering.

    Training's bounded cache consumes contiguous byte ranges.  A correct
    global row shuffle should therefore expose every sampled local window to
    all three mixture categories and several independent sources.
    """
    path = Path(path)
    windows = max(1, int(windows))
    window_rows = max(1, int(window_rows))
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Cannot audit empty dataset: {path}")
    expected = {name: tokens / sum(TARGETS.values()) for name, tokens in TARGETS.items()}
    maximum_error = Counter()
    minimum_sources = None
    sampled_rows = 0
    with path.open("rb") as stream:
        for number in range(windows):
            # Never sample at EOF. Discard the partial first line after seek.
            offset = size * number // windows
            stream.seek(offset)
            if offset:
                stream.readline()
            category_tokens = Counter()
            sources = set()
            rows = 0
            for _ in range(window_rows):
                line = stream.readline()
                if not line:
                    break
                row = orjson.loads(line)
                category = row.get("mix_category")
                if category not in TARGETS:
                    raise ValueError(
                        f"Unknown mix_category {category!r} near byte {offset}"
                    )
                category_tokens[category] += int(row.get("token_count") or 0)
                sources.add(str(row.get("source") or ""))
                rows += 1
            total = sum(category_tokens.values())
            if rows == 0 or total == 0:
                raise ValueError(f"Empty audit window near byte {offset}")
            sampled_rows += rows
            minimum_sources = (
                len(sources) if minimum_sources is None
                else min(minimum_sources, len(sources))
            )
            for category, target in expected.items():
                error = abs(category_tokens[category] / total - target)
                maximum_error[category] = max(maximum_error[category], error)

    # These are intentionally loose relative to the expected sampling noise;
    # they catch source/category blocks without rejecting an honest shuffle.
    limits = {"code": 0.12, "math": 0.08, "natural": 0.12}
    failures = [
        f"{name} local token-fraction error {maximum_error[name]:.1%} > {limits[name]:.1%}"
        for name in TARGETS if maximum_error[name] > limits[name]
    ]
    if (minimum_sources or 0) < 4:
        failures.append(f"only {minimum_sources or 0} sources in the least diverse window")
    result = {
        "method": "deterministic byte-spaced contiguous-window audit",
        "windows": windows,
        "rows_per_window": window_rows,
        "sampled_rows": sampled_rows,
        "minimum_unique_sources_per_window": minimum_sources or 0,
        "maximum_absolute_token_fraction_error": {
            name: maximum_error[name] for name in TARGETS
        },
        "limits": limits,
        "passed": not failures,
    }
    if failures:
        raise RuntimeError("Dataset is not sufficiently mixed: " + "; ".join(failures))
    return result


def reshuffle_existing(path, *, seed, chunk_rows, chunk_bytes,
                       audit_windows=64, audit_window_rows=1024):
    """Globally reshuffle a JSONL corpus, preserving the original on failure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stage = path.with_name(f".{path.name}.reshuffled.{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    shuffler = ExternalRandomShuffler(
        stage, seed, chunk_rows, deduplicate=False, chunk_bytes=chunk_bytes,
    )
    input_rows = input_bytes = 0
    next_report = 1024 ** 3
    try:
        with path.open("rb") as source:
            for line in source:
                if not line.strip():
                    continue
                shuffler.add_serialized(line)
                input_rows += 1
                input_bytes += len(line)
                if input_bytes >= next_report:
                    print(
                        f"[Coder Shuffle] indexed {input_rows:,} rows, "
                        f"{input_bytes / 1024 ** 3:.1f} GiB",
                        flush=True,
                    )
                    next_report += 1024 ** 3
        output = shuffler.finish(progress_label="Coder Shuffle")
        if output["rows"] != input_rows or output["bytes"] != input_bytes:
            raise RuntimeError(
                "Shuffle integrity mismatch: "
                f"input=({input_rows}, {input_bytes}), "
                f"output=({output['rows']}, {output['bytes']})"
            )
        mixing = audit_contiguous_mixing(
            stage, windows=audit_windows, window_rows=audit_window_rows,
        )
        os.replace(stage, path)
    except BaseException:
        shuffler.temporary_dir.cleanup()
        if stage.exists():
            stage.unlink()
        raise
    return {**output, "seed": seed, "mixing_audit": mixing}


def collected_files(component):
    directory = COLLECTED / component / "shards"
    return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []


def classify_collected(row):
    source = str(row.get("source") or "")
    if source.endswith(":code"):
        return "code"
    if source.endswith(":math"):
        return "math"
    if source.endswith(":natural_en") or source.endswith(":natural_zh"):
        return "natural"
    return None


def build(args):
    required = [BASE, CONTINUED]
    missing = [str(path) for path in required if not path.is_file()]
    for component in ("code", "math", "natural_en", "natural_zh"):
        if not collected_files(component):
            missing.append(str(COLLECTED / component / "shards"))
    if missing:
        raise FileNotFoundError("Missing completed inputs: " + ", ".join(missing))
    for path in (OUTPUT, REPORT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True).backend_tokenizer
    benchmark_set, benchmark_sources = benchmark_ngrams()
    stats = {category: Counter() for category in TARGETS}
    rejected = Counter()
    licenses = Counter()
    input_stats = {}
    seen_texts = set()
    shuffler = ExternalRandomShuffler(
        OUTPUT, args.shuffle_seed, args.shuffle_chunk_rows,
        chunk_bytes=args.shuffle_chunk_mb * 1024 ** 2,
    )

    def full(category):
        return stats[category]["tokens"] >= TARGETS[category] - MAX_TOKENS

    def accept(row, category, input_name):
        if category is None:
            rejected[f"{input_name}:unclassified"] += 1
            return False
        if full(category):
            rejected[f"{input_name}:category_full"] += 1
            return False
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            rejected[f"{input_name}:invalid_text"] += 1
            return False
        fingerprint = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if fingerprint in seen_texts:
            rejected[f"{input_name}:duplicate_text"] += 1
            return False
        if has_benchmark_overlap(text, benchmark_set):
            rejected[f"{input_name}:benchmark_overlap"] += 1
            return False
        token_count = row.get("token_count")
        if not isinstance(token_count, int):
            token_count = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
        if not 1 <= token_count <= MAX_TOKENS:
            rejected[f"{input_name}:outside_token_range"] += 1
            return False
        if stats[category]["tokens"] + token_count > TARGETS[category]:
            rejected[f"{input_name}:over_budget"] += 1
            return False
        output_row = dict(row)
        output_row["token_count"] = token_count
        output_row["mix_category"] = category
        output_row["mixture"] = "deepseek-coder-v2-60-code-10-math-30-natural"
        if not shuffler.add(output_row):
            rejected[f"{input_name}:duplicate_record"] += 1
            return False
        seen_texts.add(fingerprint)
        stats[category]["rows"] += 1
        stats[category]["tokens"] += token_count
        stats[category]["bytes"] += len(text.encode("utf-8"))
        licenses[str(row.get("license") or "unspecified")] += 1
        return True

    def consume(path, input_name, classifier, allowed_sources=None):
        before = {name: stats[name]["tokens"] for name in TARGETS}
        scanned = accepted = 0
        with Path(path).open("rb") as stream:
            for line in stream:
                scanned += 1
                row = orjson.loads(line)
                if allowed_sources is not None and row.get("source") not in allowed_sources:
                    continue
                if accept(row, classifier(row), input_name):
                    accepted += 1
                if all(full(name) for name in TARGETS):
                    break
        input_stats[input_name] = {
            "path": str(path), "scanned_rows": scanned, "accepted_rows": accepted,
            "accepted_tokens": {
                name: stats[name]["tokens"] - before[name] for name in TARGETS
            },
        }
        print(f"{input_name}: accepted={accepted:,}; "
              + ", ".join(f"{name}={stats[name]['tokens']:,}" for name in TARGETS),
              flush=True)

    try:
        consume(BASE, "base_codespecialist", classify_base)
        consume(CONTINUED, "continued_novel", classify_continued, CONTINUED_SOURCES)
        for component in ("code", "math", "natural_en", "natural_zh"):
            for index, path in enumerate(collected_files(component)):
                category = "natural" if component.startswith("natural_") else component
                if full(category):
                    break
                consume(path, f"collected_{component}_{index:04d}", classify_collected)

        short = {name: TARGETS[name] - stats[name]["tokens"]
                 for name in TARGETS if not full(name)}
        if short:
            raise RuntimeError(f"Collected headroom cannot fill final targets: {short}")
        shuffle_result = shuffler.finish(progress_label="Coder Build Shuffle")
    except BaseException:
        shuffler.temporary_dir.cleanup()
        raise

    mixing_audit = audit_contiguous_mixing(
        OUTPUT, windows=args.audit_windows,
        window_rows=args.audit_window_rows,
    )

    total_tokens = sum(value["tokens"] for value in stats.values())
    report = {
        "name": "instinct_coder_pretrain_12b",
        "created_for": "Instinct V1 MoE: 678,726,144 total / 106,203,648 active parameters",
        "reference": {
            "paper": "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence",
            "arxiv": "2406.11931",
            "mixture": {"source_code": 0.60, "math": 0.10, "natural_language": 0.30},
        },
        "output": str(OUTPUT),
        "output_sha256": shuffle_result["sha256"],
        "rows": sum(value["rows"] for value in stats.values()),
        "tokens": total_tokens,
        "unique_tokens_per_active_parameter": total_tokens / 106_203_648,
        "unique_tokens_per_total_parameter": total_tokens / 678_726_144,
        "targets": TARGETS,
        "components": {name: dict(value) for name, value in stats.items()},
        "component_token_fraction": {
            name: value["tokens"] / total_tokens for name, value in stats.items()
        },
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "whole_records_only": True,
        "text_split": False,
        "text_truncated": False,
        "global_exact_text_deduplication": True,
        "shuffle_seed": args.shuffle_seed,
        "shuffle": {
            "method": "128-bit random-key external merge sort",
            "seed": args.shuffle_seed,
            "chunk_rows": args.shuffle_chunk_rows,
            "chunk_bytes": args.shuffle_chunk_mb * 1024 ** 2,
            "temporary_chunks": shuffle_result["temporary_chunks"],
            "mixing_audit": mixing_audit,
        },
        "inputs": input_stats,
        "rejected": dict(rejected),
        "benchmark_screening": {
            "method": "reject any record sharing a 13-word n-gram with held-out benchmark text",
            "sources": benchmark_sources,
        },
        "license_row_counts": dict(licenses),
        "checks": [
            "all output texts are globally exact-deduplicated",
            "all records contain at most 4096 project-tokenizer tokens including BOS/EOS",
            "all records are complete and were neither split nor truncated",
            "all rows were screened against HumanEval, sanitized MBPP-test, and GSM8K-test",
            "final token mixture is 60% code, 10% math, and 30% natural language",
        ],
        "limitations": [
            "13-word n-gram screening cannot prove absence of semantic benchmark contamination.",
            "Open-source code retains heterogeneous per-file licenses; redistribution requires upstream license review.",
            "Some inputs carry attribution/share-alike obligations documented in each row.",
            "Source-code rows are file-oriented rather than dependency-ordered repository sequences.",
            "FIM is intentionally left to the training-time augmentation pipeline.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    args = parse_args()
    if args.shuffle_chunk_rows < 1 or args.shuffle_chunk_mb < 1:
        raise ValueError("shuffle chunk rows/MB must both be positive")
    if args.audit_windows < 1 or args.audit_window_rows < 1:
        raise ValueError("audit windows/window rows must both be positive")
    if not args.reshuffle_existing:
        build(args)
        return

    result = reshuffle_existing(
        OUTPUT, seed=args.shuffle_seed,
        chunk_rows=args.shuffle_chunk_rows,
        chunk_bytes=args.shuffle_chunk_mb * 1024 ** 2,
        audit_windows=args.audit_windows,
        audit_window_rows=args.audit_window_rows,
    )
    if REPORT.is_file():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report["output_sha256"] = result["sha256"]
        report["shuffle_seed"] = args.shuffle_seed
        report["shuffle"] = {
            "method": "128-bit random-key external merge sort",
            "seed": args.shuffle_seed,
            "chunk_rows": args.shuffle_chunk_rows,
            "chunk_bytes": args.shuffle_chunk_mb * 1024 ** 2,
            "temporary_chunks": result["temporary_chunks"],
            "mixing_audit": result["mixing_audit"],
            "reshuffled_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        temporary_report = REPORT.with_suffix(REPORT.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, REPORT)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
