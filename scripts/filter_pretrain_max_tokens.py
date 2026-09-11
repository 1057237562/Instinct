"""Atomically remove overlength records from a pretraining JSONL corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import orjson
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("dataset/pretrain_specialist.jsonl")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-chars", type=int, default=2_000_000)
    parser.add_argument("--batch-rows", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (args.output or args.input).resolve()
    report_path = (
        args.report.resolve()
        if args.report
        else output_path.with_name(output_path.stem + ".report.json")
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.max_tokens <= 0 or args.batch_chars <= 0 or args.batch_rows <= 0:
        raise ValueError("Limits must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path.resolve(), trust_remote_code=True
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    backend = tokenizer.backend_tokenizer
    tokenizer_spec = orjson.loads(backend.to_str())
    if tokenizer_spec.get("pre_tokenizer", {}).get("type") != "ByteLevel":
        raise ValueError("The proven UTF-8 byte fast path requires ByteLevel tokenization")

    temporary_path = output_path.with_name(output_path.name + ".max_tokens.tmp")
    if temporary_path.exists():
        raise FileExistsError(temporary_path)
    if output_path != input_path and output_path.exists():
        raise FileExistsError(output_path)

    old_report = None
    if report_path.is_file():
        old_report = json.loads(report_path.read_text(encoding="utf-8"))

    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    input_rows = 0
    output_rows = 0
    rejected_rows = 0
    input_bytes = 0
    output_bytes = 0
    exactly_tokenized_rows = 0
    byte_proven_rows = 0
    largest_rejected_tokens = 0
    batch: list[tuple[bytes, str | None]] = []
    batch_chars = 0

    def flush(destination) -> None:
        nonlocal batch, batch_chars, output_rows, rejected_rows
        nonlocal output_bytes, exactly_tokenized_rows, byte_proven_rows
        nonlocal largest_rejected_tokens
        candidate_texts = [text for _, text in batch if text is not None]
        encodings = backend.encode_batch(candidate_texts, add_special_tokens=False)
        encoding_iter = iter(encodings)
        for line, candidate_text in batch:
            if candidate_text is None:
                keep = True
                byte_proven_rows += 1
            else:
                token_count = len(next(encoding_iter).ids)
                exactly_tokenized_rows += 1
                keep = token_count <= args.max_tokens
                if not keep:
                    largest_rejected_tokens = max(largest_rejected_tokens, token_count)
            if keep:
                destination.write(line)
                output_digest.update(line)
                output_rows += 1
                output_bytes += len(line)
            else:
                rejected_rows += 1
        batch = []
        batch_chars = 0

    try:
        with input_path.open("rb") as source, temporary_path.open(
            "xb", buffering=8 * 1024 * 1024
        ) as destination:
            for input_rows, line in enumerate(source, start=1):
                input_digest.update(line)
                input_bytes += len(line)
                record = orjson.loads(line)
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"Invalid string text field at line {input_rows}")

                text_bytes = len(text.encode("utf-8"))
                # ByteLevel maps each UTF-8 byte to at most one initial symbol;
                # BPE merges symbols, so token_count cannot exceed byte count.
                candidate_text = None if text_bytes <= args.max_tokens else text
                batch.append((line, candidate_text))
                batch_chars += len(text) if candidate_text is not None else 0
                if len(batch) >= args.batch_rows or batch_chars >= args.batch_chars:
                    flush(destination)
                if input_rows % 250_000 == 0:
                    print(
                        f"scanned={input_rows:,} kept={output_rows:,} "
                        f"rejected={rejected_rows:,}",
                        flush=True,
                    )
            flush(destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    report = {
        "name": "pretrain_specialist",
        "output": str(output_path),
        "format": "line-delimited JSON",
        "components": old_report.get("components") if old_report else None,
        "pre_filter": {
            "rows": input_rows,
            "bytes": input_bytes,
            "sha256": input_digest.hexdigest(),
        },
        "max_token_filter": {
            "tokenizer": str(args.tokenizer_path.resolve()),
            "max_tokens": args.max_tokens,
            "kept_rows": output_rows,
            "rejected_rows": rejected_rows,
            "byte_proven_rows": byte_proven_rows,
            "exactly_tokenized_rows": exactly_tokenized_rows,
            "largest_rejected_tokens": largest_rejected_tokens,
        },
        "output_rows": output_rows,
        "output_bytes": output_bytes,
        "output_sha256": output_digest.hexdigest(),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
