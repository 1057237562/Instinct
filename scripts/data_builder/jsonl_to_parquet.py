"""Convert a pretrain JSONL corpus to Parquet, mirroring dataset_compiler rules.

Used when the Rust dataset_compiler toolchain is unavailable. Schema contract
(same as dataset_compiler/README.md and dataset.md):

- ``text`` stays a string column; other scalar keys are preserved as-is,
- nested objects/arrays (e.g. ``metadata``) become JSON text,
- keys whose types conflict across rows degrade to JSON text and are listed
  as warnings in the report,
- zstd compression; row counts and sampled values are verified against the
  source JSONL afterwards.

Two passes over the source: one full schema scan, then the write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

INT = "int"
FLOAT = "float"
STR = "str"
JSON_TEXT = "json"


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def describe(value) -> str:
    if value is None:
        return "null"
    if is_int(value):
        return INT
    if isinstance(value, float):
        return FLOAT
    if isinstance(value, str):
        return STR
    return JSON_TEXT


def merge_type(current: str | None, incoming: str) -> str:
    if incoming == "null":
        return current or "null"
    if current is None or current == "null":
        return incoming
    if current == incoming:
        return current
    if {current, incoming} <= {INT, FLOAT}:
        return FLOAT
    return JSON_TEXT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--verify-rows", type=int, default=2000)
    return parser.parse_args()


def scan_schema(path: Path):
    types: dict[str, str] = {}
    rows = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += 1
            row = orjson.loads(line)
            for key, value in row.items():
                types[key] = merge_type(types.get(key), describe(value))
    return types, rows, digest.hexdigest()


def json_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return orjson.dumps(value).decode("utf-8")


def arrow_type(kind: str):
    return {INT: pa.int64(), FLOAT: pa.float64(), STR: pa.string(),
            JSON_TEXT: pa.string(), "null": pa.string()}[kind]


def convert(input_path: Path, output_path: Path, types: dict, batch_rows: int):
    if "text" not in types:
        raise ValueError("corpus must contain a text column")
    ordered = ["text"] + sorted(key for key in types if key != "text")
    schema = pa.schema([pa.field(key, arrow_type(types[key])) for key in ordered])
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0
    started = time.time()
    batch = {key: [] for key in ordered}

    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        def flush():
            nonlocal written, batch
            if not batch["text"]:
                return
            table = pa.table(
                {key: pa.array(values, type=schema.field(key).type)
                 for key, values in batch.items()},
                schema=schema)
            writer.write_table(table)
            written += table.num_rows
            if written % 1_000_000 < batch_rows:
                print(f"[parquet] wrote {written:,} rows", flush=True)
            batch = {key: [] for key in ordered}

        with input_path.open("rb") as handle:
            for line in handle:
                row = orjson.loads(line)
                for key in ordered:
                    value = row.get(key)
                    kind = types[key]
                    if kind == JSON_TEXT:
                        batch[key].append(json_text(value))
                    elif kind == INT:
                        batch[key].append(int(value) if is_int(value) else None)
                    elif kind == FLOAT:
                        batch[key].append(
                            float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
                    else:
                        batch[key].append(None if value is None else str(value))
                if len(batch["text"]) >= batch_rows:
                    flush()
        flush()
    output_path.unlink(missing_ok=True)
    temporary.rename(output_path)
    return written, ordered, time.time() - started


def verify(input_path: Path, output_path: Path, types: dict, source_rows: int, sample_rows: int) -> dict:
    parquet = pq.ParquetFile(output_path)
    total = parquet.metadata.num_rows
    if total != source_rows:
        return {"parquet_rows": total, "source_rows": source_rows, "passed": False,
                "reason": "row count mismatch"}
    rng = random.Random(20261005)
    wanted = set(rng.sample(range(source_rows), min(sample_rows, source_rows)))
    expected = {}
    with input_path.open("rb") as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                expected[index] = orjson.loads(line)
                if len(expected) == len(wanted):
                    break
    checked = mismatches = 0
    index = 0
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if index in wanted:
                exp = expected[index]
                for key, value in exp.items():
                    got = row.get(key)
                    kind = types[key]
                    if kind == JSON_TEXT:
                        same = (orjson.loads(got) if got is not None else None) == value
                    elif kind == FLOAT:
                        same = (got is None and value is None) or (
                            isinstance(value, (int, float)) and not isinstance(value, bool)
                            and got is not None and float(value) == got)
                    elif kind == INT:
                        same = got == (int(value) if is_int(value) else None)
                    else:
                        same = (got is None and value is None) or str(value) == got
                    if not same:
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"[verify] mismatch row {index} key {key}: {value!r} != {got!r}", flush=True)
                checked += 1
            index += 1
        if checked == len(wanted):
            break
    return {"parquet_rows": total, "source_rows": source_rows,
            "sampled_rows": checked, "value_mismatches": mismatches,
            "passed": mismatches == 0}


def main() -> None:
    args = parse_args()
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("output/report already exists")
    print(f"[parquet] scanning schema of {args.input}", flush=True)
    types, rows, source_sha = scan_schema(args.input)
    warnings = [f"{key}: mixed/nested values stored as JSON text"
                for key, kind in types.items() if kind == JSON_TEXT]
    print(f"[parquet] rows={rows:,}, columns={list(types)}, json-text={len(warnings)}", flush=True)
    written, ordered, elapsed = convert(args.input, args.output, types, args.batch_rows)
    if written != rows:
        raise RuntimeError(f"row mismatch: wrote {written}, source {rows}")
    print("[parquet] verifying sampled rows", flush=True)
    verification = verify(args.input, args.output, types, rows, args.verify_rows)
    if not verification["passed"]:
        raise RuntimeError(f"verification failed: {verification}")
    digest = hashlib.sha256()
    with args.output.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report = {
        "name": "jsonl_to_parquet (dataset_compiler-compatible rules)",
        "source": str(args.input), "source_rows": rows, "source_sha256": source_sha,
        "output": str(args.output), "output_rows": written,
        "columns": {key: types[key] for key in ordered},
        "compression": "zstd",
        "warnings": warnings,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "conversion_seconds": round(elapsed, 1),
        "verification": verification,
        "note": "text + scalar keys preserved; nested objects and type conflicts as JSON text (same contract as dataset_compiler)",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "columns"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
