"""Build a size-bounded, quality-filtered Codeforces pretraining extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer


COMMON_LANGUAGES = (
    "C++",
    "GNU C",
    "Python",
    "PyPy",
    "Java",
    "Kotlin",
    "Go",
    "Rust",
    "C#",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced, deduplicated Codeforces pretraining mix."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("dataset/pretrain_t2t_mini.jsonl"),
        help="Base corpus; copied verbatim and never modified.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("dataset/codeforces-submissions/data"),
    )
    parser.add_argument(
        "--selected-parquet",
        type=Path,
        default=Path(
            "dataset/codeforces-submissions/selected_accepted/"
            "train-00000-of-00001.parquet"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("dataset/pretrain_t2t_mini_codeforces_600mb.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/pretrain_t2t_mini_codeforces_600mb.report.json"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--target-code-bytes", type=int, default=600_000_000)
    parser.add_argument(
        "--balanced-bytes-per-problem",
        type=int,
        default=60_000,
        help="First-pass per-problem budget before a second pass fills any deficit.",
    )
    parser.add_argument("--max-samples-per-problem", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args()


def copy_and_validate_base(
    source: Path, destination, output_digest, base_digest
) -> tuple[int, int]:
    rows = 0
    byte_count = 0
    ended_with_newline = True
    with source.open("rb") as handle:
        for rows, line in enumerate(handle, start=1):
            record = orjson.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Invalid text record at {source}:{rows}")
            destination.write(line)
            output_digest.update(line)
            base_digest.update(line)
            byte_count += len(line)
            ended_with_newline = line.endswith(b"\n")
    if byte_count and not ended_with_newline:
        destination.write(b"\n")
        output_digest.update(b"\n")
        base_digest.update(b"\n")
        byte_count += 1
    return rows, byte_count


def language_is_supported(language: str | None) -> bool:
    return bool(language) and any(name in language for name in COMMON_LANGUAGES)


def main() -> None:
    args = parse_args()
    started = time.time()
    input_path = args.input_jsonl.resolve()
    output_path = args.output_jsonl.resolve()
    report_path = args.report.resolve()
    selected_path = args.selected_parquet.resolve()
    parquet_files = sorted(args.parquet_dir.resolve().glob("*.parquet"))

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found in {args.parquet_dir}")
    if output_path == input_path:
        raise ValueError("Output must differ from the base corpus")
    if report_path.exists() or output_path.exists():
        raise FileExistsError("Output/report already exists; refusing to duplicate the mix")
    if not (0 < args.min_tokens <= args.max_tokens):
        raise ValueError("Require 0 < min_tokens <= max_tokens")
    if args.target_code_bytes <= 0 or args.balanced_bytes_per_problem <= 0:
        raise ValueError("Byte budgets must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path.resolve(), trust_remote_code=True
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(temporary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_digest = hashlib.sha256()
    base_digest = hashlib.sha256()
    seen_source_hashes: set[bytes] = set()
    problem_bytes: defaultdict[str, int] = defaultdict(int)
    problem_samples: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    code_bytes = 0
    base_rows = 0
    base_bytes = 0

    def encoded_line_if_eligible(row: dict) -> bytes | None:
        nonlocal code_bytes
        source = row.get("source")
        language = row.get("programmingLanguage")
        problem_id = row.get("problem_id")
        if not source or not source.strip() or not problem_id:
            stats["rejected_empty"] += 1
            return None
        if not language_is_supported(language):
            stats["rejected_language"] += 1
            return None
        source_hash = hashlib.blake2b(source.encode("utf-8"), digest_size=16).digest()
        if source_hash in seen_source_hashes:
            stats["rejected_duplicate"] += 1
            return None
        token_count = len(
            tokenizer.backend_tokenizer.encode(source, add_special_tokens=False).ids
        )
        if token_count < args.min_tokens:
            stats["rejected_too_short"] += 1
            return None
        if token_count > args.max_tokens:
            stats["rejected_too_long"] += 1
            return None
        line = orjson.dumps({"text": source}, option=orjson.OPT_APPEND_NEWLINE)
        if code_bytes + len(line) > args.target_code_bytes:
            stats["rejected_over_budget"] += 1
            return None
        seen_source_hashes.add(source_hash)
        return line

    def keep_row(row: dict, destination, origin: str) -> bool:
        nonlocal code_bytes
        problem_id = row["problem_id"]
        if problem_samples[problem_id] >= args.max_samples_per_problem:
            stats["rejected_problem_cap"] += 1
            return False
        line = encoded_line_if_eligible(row)
        if line is None:
            return False
        destination.write(line)
        output_digest.update(line)
        code_bytes += len(line)
        problem_bytes[problem_id] += len(line)
        problem_samples[problem_id] += 1
        language_counts[row["programmingLanguage"]] += 1
        stats[f"kept_{origin}"] += 1
        return True

    columns = ["source", "programmingLanguage", "problem_id"]

    try:
        with temporary_path.open("xb", buffering=8 * 1024 * 1024) as destination:
            base_rows, base_bytes = copy_and_validate_base(
                input_path, destination, output_digest, base_digest
            )

            # Highest-confidence seed: submissions re-executed against public tests.
            selected = pq.ParquetFile(selected_path)
            for batch in selected.iter_batches(batch_size=args.batch_size, columns=columns):
                for row in batch.to_pylist():
                    keep_row(row, destination, "selected_accepted")

            # Balanced pass: give every problem comparable representation.
            raw_columns = columns + ["verdict", "testset"]
            for shard_index, parquet_path in enumerate(parquet_files, start=1):
                parquet = pq.ParquetFile(parquet_path)
                for batch in parquet.iter_batches(
                    batch_size=args.batch_size, columns=raw_columns
                ):
                    mask = pc.and_(
                        pc.equal(batch.column("verdict"), "OK"),
                        pc.equal(batch.column("testset"), "TESTS"),
                    )
                    filtered = pc.filter(batch, mask)
                    for row in filtered.select(columns).to_pylist():
                        problem_id = row["problem_id"]
                        if problem_bytes[problem_id] >= args.balanced_bytes_per_problem:
                            stats["deferred_balanced_budget"] += 1
                            continue
                        keep_row(row, destination, "balanced_raw")
                print(
                    f"balanced pass {shard_index}/{len(parquet_files)}; "
                    f"code={code_bytes / 1_000_000:.1f} MB",
                    flush=True,
                )

            # Fill a small remaining deficit while retaining the per-problem cap.
            if code_bytes < args.target_code_bytes:
                for shard_index, parquet_path in enumerate(parquet_files, start=1):
                    parquet = pq.ParquetFile(parquet_path)
                    for batch in parquet.iter_batches(
                        batch_size=args.batch_size, columns=raw_columns
                    ):
                        mask = pc.and_(
                            pc.equal(batch.column("verdict"), "OK"),
                            pc.equal(batch.column("testset"), "TESTS"),
                        )
                        filtered = pc.filter(batch, mask)
                        for row in filtered.select(columns).to_pylist():
                            keep_row(row, destination, "fill_raw")
                            if args.target_code_bytes - code_bytes < 128:
                                break
                        if args.target_code_bytes - code_bytes < 128:
                            break
                    print(
                        f"fill pass {shard_index}/{len(parquet_files)}; "
                        f"code={code_bytes / 1_000_000:.1f} MB",
                        flush=True,
                    )
                    if args.target_code_bytes - code_bytes < 128:
                        break

            destination.flush()
            os.fsync(destination.fileno())

        output_bytes = temporary_path.stat().st_size
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    report = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "selected_accepted": str(selected_path),
        "parquet_dir": str(args.parquet_dir.resolve()),
        "quality_filter": {
            "selected_accepted_first": True,
            "raw_verdict": "OK",
            "raw_testset": "TESTS",
            "languages": list(COMMON_LANGUAGES),
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "exact_source_deduplication": True,
            "balanced_bytes_per_problem": args.balanced_bytes_per_problem,
            "max_samples_per_problem": args.max_samples_per_problem,
        },
        "target_code_bytes": args.target_code_bytes,
        "code_jsonl_bytes": code_bytes,
        "base_rows": base_rows,
        "base_bytes": base_bytes,
        "base_sha256": base_digest.hexdigest(),
        "code_rows": sum(value for key, value in stats.items() if key.startswith("kept_")),
        "output_rows": base_rows
        + sum(value for key, value in stats.items() if key.startswith("kept_")),
        "output_bytes": output_bytes,
        "represented_problems": len(problem_samples),
        "selection_stats": dict(sorted(stats.items())),
        "language_counts": dict(language_counts.most_common()),
        "output_sha256": output_digest.hexdigest(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
