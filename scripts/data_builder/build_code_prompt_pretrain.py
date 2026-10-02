"""Build a prompt-rich code pretraining JSONL from locally collected datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import orjson
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bigcode",
        type=Path,
        default=Path(
            "dataset/bigcode-self-oss-instruct-50k/data/"
            "train-00000-of-00001.parquet"
        ),
    )
    parser.add_argument(
        "--magicoder",
        type=Path,
        default=Path(
            "dataset/magicoder-oss-75k/data-oss_instruct-decontaminated.jsonl"
        ),
    )
    parser.add_argument(
        "--vezora",
        type=Path,
        default=Path(
            "dataset/vezora-tested-python-22k/188k-Vezora-PyCode-Alpaca.json"
        ),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/pretrain_code_prompts_extra.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/pretrain_code_prompts_extra.report.json"),
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_bigcode(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["instruction", "response"]):
        for row in batch.to_pylist():
            yield row["instruction"], row["response"]


def iter_magicoder(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            row = orjson.loads(line)
            yield row.get("problem"), row.get("solution")


def iter_vezora(path: Path):
    rows = orjson.loads(path.read_bytes())
    for row in rows:
        instruction = normalize_text(row.get("instruction"))
        input_text = normalize_text(row.get("input"))
        if input_text:
            instruction = f"{instruction}\n\nInput:\n{input_text}"
        yield instruction, row.get("output")


def format_example(instruction: str, response: str) -> str:
    return f"### Instruction\n{instruction}\n\n### Response\n{response}"


def main() -> None:
    args = parse_args()
    started = time.time()
    sources = [
        (
            "bigcode_self_oss_exec_50k",
            args.bigcode.resolve(),
            iter_bigcode,
            "ODC-By",
            "356bb069eee815daa6e23e9a282eeefe1490ad44",
        ),
        (
            "vezora_tested_python_22k",
            args.vezora.resolve(),
            iter_vezora,
            "Apache-2.0",
            "578b29091cce76ec1b464db1bb76cd37c4e9a7bf",
        ),
        (
            "magicoder_oss_75k",
            args.magicoder.resolve(),
            iter_magicoder,
            "MIT",
            "5f839b1f368a76b161028bb9edff055db34022b2",
        ),
    ]
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    for _, path, _, _, _ in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() or report_path.exists() or temporary_path.exists():
        raise FileExistsError("Output, report, or temporary output already exists")
    if not (0 < args.min_tokens <= args.max_tokens):
        raise ValueError("Require 0 < min_tokens <= max_tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path.resolve(), trust_remote_code=True
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    backend = tokenizer.backend_tokenizer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_hashes: set[bytes] = set()
    full_hashes: set[bytes] = set()
    stats: dict[str, Counter] = {name: Counter() for name, *_ in sources}
    digest = hashlib.sha256()
    output_rows = 0
    output_bytes = 0

    try:
        with temporary_path.open("xb", buffering=8 * 1024 * 1024) as destination:
            for name, path, iterator, _, _ in sources:
                for raw_instruction, raw_response in iterator(path):
                    stats[name]["seen"] += 1
                    instruction = normalize_text(raw_instruction)
                    response = normalize_text(raw_response)
                    if not instruction or not response:
                        stats[name]["rejected_empty"] += 1
                        continue

                    prompt_hash = hashlib.blake2b(
                        instruction.casefold().encode("utf-8"), digest_size=16
                    ).digest()
                    if prompt_hash in prompt_hashes:
                        stats[name]["rejected_duplicate_prompt"] += 1
                        continue

                    text = format_example(instruction, response)
                    full_hash = hashlib.blake2b(
                        text.encode("utf-8"), digest_size=16
                    ).digest()
                    if full_hash in full_hashes:
                        stats[name]["rejected_duplicate_record"] += 1
                        continue

                    token_count = len(
                        backend.encode(text, add_special_tokens=False).ids
                    )
                    if token_count < args.min_tokens:
                        stats[name]["rejected_too_short"] += 1
                        continue
                    if token_count > args.max_tokens:
                        stats[name]["rejected_too_long"] += 1
                        continue

                    line = orjson.dumps(
                        {"text": text}, option=orjson.OPT_APPEND_NEWLINE
                    )
                    destination.write(line)
                    digest.update(line)
                    output_rows += 1
                    output_bytes += len(line)
                    prompt_hashes.add(prompt_hash)
                    full_hashes.add(full_hash)
                    stats[name]["kept"] += 1
                print(name, dict(stats[name]), flush=True)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    report = {
        "output": str(output_path),
        "format": "### Instruction\\n...\\n\\n### Response\\n...",
        "tokenizer": str(args.tokenizer_path.resolve()),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "deduplication": "case-insensitive exact prompt plus exact full record",
        "sources": {
            name: {
                "path": str(path),
                "license": license_name,
                "revision": revision,
                **dict(stats[name]),
            }
            for name, path, _, license_name, revision in sources
        },
        "output_rows": output_rows,
        "output_bytes": output_bytes,
        "output_sha256": digest.hexdigest(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
