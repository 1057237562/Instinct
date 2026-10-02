"""Build a math-reasoning extension and merge it with codespecialist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pyarrow.dataset as ds
from transformers import AutoTokenizer


NUMINA_SOURCES = (
    "olympiads_ref",
    "amc_aime",
    "inequalities",
    "number_theory",
    "cn_contest",
    "aops_forum",
    "olympiads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codespecialist",
        type=Path,
        default=Path("dataset/codespecialist.jsonl"),
    )
    parser.add_argument(
        "--numina-dir", type=Path, default=Path("dataset/numinamath-1.5/data")
    )
    parser.add_argument(
        "--verified-math",
        type=Path,
        default=Path("dataset/verified-math-reasoning-3k/data/train.jsonl"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--target-math-bytes", type=int, default=500_000_000)
    parser.add_argument("--balanced-bytes-per-domain", type=int, default=55_000_000)
    parser.add_argument(
        "--math-output",
        type=Path,
        default=Path("dataset/pretrain_math_reasoning_500mb.jsonl"),
    )
    parser.add_argument(
        "--math-report",
        type=Path,
        default=Path("dataset/pretrain_math_reasoning_500mb.report.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/pretrain_specialist.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("dataset/pretrain_specialist.report.json")
    )
    return parser.parse_args()


def normalize(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def format_example(problem: str, solution: str) -> str:
    return f"### Instruction\n{problem}\n\n### Response\n{solution}"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    started = time.time()
    codespecialist = args.codespecialist.resolve()
    numina_dir = args.numina_dir.resolve()
    verified_math = args.verified_math.resolve()
    math_output = args.math_output.resolve()
    math_report_path = args.math_report.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    math_tmp = math_output.with_name(math_output.name + ".tmp")
    output_tmp = output.with_name(output.name + ".tmp")

    if not codespecialist.is_file() or not verified_math.is_file():
        raise FileNotFoundError("Missing codespecialist or verified-math input")
    if not numina_dir.is_dir():
        raise FileNotFoundError(numina_dir)
    for path in (math_output, math_report_path, output, report_path, math_tmp, output_tmp):
        if path.exists():
            raise FileExistsError(path)
    if not (0 < args.min_tokens <= args.max_tokens):
        raise ValueError("Require 0 < min_tokens <= max_tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path.resolve(), trust_remote_code=True
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    backend = tokenizer.backend_tokenizer
    numina = ds.dataset(numina_dir, format="parquet")

    math_output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_prompts: set[bytes] = set()
    domain_bytes: defaultdict[str, int] = defaultdict(int)
    source_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    math_digest = hashlib.sha256()
    math_bytes = 0
    math_rows = 0

    def keep(problem_value, solution_value, source: str, domain: str, destination) -> bool:
        nonlocal math_bytes, math_rows
        problem = normalize(problem_value)
        solution = normalize(solution_value)
        if not problem or not solution:
            stats["rejected_empty"] += 1
            return False
        prompt_hash = hashlib.blake2b(
            problem.casefold().encode("utf-8"), digest_size=16
        ).digest()
        if prompt_hash in seen_prompts:
            stats["rejected_duplicate_prompt"] += 1
            return False
        text = format_example(problem, solution)
        token_count = len(backend.encode(text, add_special_tokens=False).ids)
        if token_count < args.min_tokens:
            stats["rejected_too_short"] += 1
            return False
        if token_count > args.max_tokens:
            stats["rejected_too_long"] += 1
            return False
        line = orjson.dumps({"text": text}, option=orjson.OPT_APPEND_NEWLINE)
        if math_bytes + len(line) > args.target_math_bytes:
            stats["rejected_over_budget"] += 1
            return False
        destination.write(line)
        math_digest.update(line)
        math_bytes += len(line)
        math_rows += 1
        seen_prompts.add(prompt_hash)
        domain_bytes[domain] += len(line)
        source_counts[source] += 1
        domain_counts[domain] += 1
        return True

    columns = ["problem", "solution", "problem_type"]
    base_filter = (
        (ds.field("problem_is_valid") == "Yes")
        & (ds.field("solution_is_valid") == "Yes")
        & (ds.field("synthetic") == False)  # noqa: E712
    )

    try:
        with math_tmp.open("xb", buffering=8 * 1024 * 1024) as destination:
            # Small, independently answer-verified arithmetic foundation.
            with verified_math.open("rb") as handle:
                for line in handle:
                    row = orjson.loads(line)
                    keep(
                        row.get("instruction"),
                        row.get("output"),
                        "verified_math_reasoning_3k",
                        "Arithmetic Foundations",
                        destination,
                    )

            # First pass balances mathematical domains and processes the most
            # curated Numina sources before broader olympiad/forum material.
            for source in NUMINA_SOURCES:
                scanner = numina.scanner(
                    columns=columns,
                    filter=base_filter & (ds.field("source") == source),
                    batch_size=32_768,
                )
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        domain = normalize(row.get("problem_type")) or "Other"
                        if domain_bytes[domain] >= args.balanced_bytes_per_domain:
                            stats["deferred_domain_budget"] += 1
                            continue
                        keep(
                            row.get("problem"),
                            row.get("solution"),
                            source,
                            domain,
                            destination,
                        )
                print(
                    f"balanced {source}: {math_bytes / 1_000_000:.1f} MB",
                    flush=True,
                )

            # Fill the remaining budget in the same source-quality order.
            if args.target_math_bytes - math_bytes >= 128:
                for source in NUMINA_SOURCES:
                    scanner = numina.scanner(
                        columns=columns,
                        filter=base_filter & (ds.field("source") == source),
                        batch_size=32_768,
                    )
                    stop = False
                    for batch in scanner.to_batches():
                        for row in batch.to_pylist():
                            domain = normalize(row.get("problem_type")) or "Other"
                            keep(
                                row.get("problem"),
                                row.get("solution"),
                                source,
                                domain,
                                destination,
                            )
                            if args.target_math_bytes - math_bytes < 128:
                                stop = True
                                break
                        if stop:
                            break
                    print(
                        f"fill {source}: {math_bytes / 1_000_000:.1f} MB",
                        flush=True,
                    )
                    if stop:
                        break
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(math_tmp, math_output)
    except BaseException:
        if math_tmp.exists():
            math_tmp.unlink()
        raise

    math_report = {
        "output": str(math_output),
        "format": "JSONL: {'text': '### Instruction\\n...\\n\\n### Response\\n...'}",
        "target_bytes": args.target_math_bytes,
        "output_bytes": math_bytes,
        "output_rows": math_rows,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "numina_filter": {
            "sources": list(NUMINA_SOURCES),
            "problem_is_valid": "Yes",
            "solution_is_valid": "Yes",
            "synthetic": False,
            "balanced_bytes_per_domain": args.balanced_bytes_per_domain,
        },
        "source_counts": dict(source_counts),
        "domain_counts": dict(domain_counts),
        "domain_bytes": dict(domain_bytes),
        "selection_stats": dict(stats),
        "licenses": {
            "AI-MO/NuminaMath-1.5": "Apache-2.0",
            "HSH-Intelligence/verified-math-reasoning-3k": "Apache-2.0",
        },
        "revisions": {
            "AI-MO/NuminaMath-1.5": "1b05109f9e5c1ad06c0663519502416c30b300f8",
            "HSH-Intelligence/verified-math-reasoning-3k": "3726afcbcf556adbb3df2611a6d0d14b45cc49fa",
        },
        "output_sha256": math_digest.hexdigest(),
    }
    math_report_path.write_text(
        json.dumps(math_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    codespecialist_report = json.loads(
        Path("dataset/codespecialist.report.json").read_text(encoding="utf-8")
    )
    expected_code_hash = codespecialist_report["output_sha256"]
    final_digest = hashlib.sha256()
    try:
        with output_tmp.open("xb", buffering=8 * 1024 * 1024) as destination:
            for source_path, expected_hash in (
                (codespecialist, expected_code_hash),
                (math_output, math_digest.hexdigest()),
            ):
                source_digest = hashlib.sha256()
                last_byte = b""
                with source_path.open("rb") as source_handle:
                    for chunk in iter(
                        lambda: source_handle.read(8 * 1024 * 1024), b""
                    ):
                        destination.write(chunk)
                        final_digest.update(chunk)
                        source_digest.update(chunk)
                        last_byte = chunk[-1:]
                if last_byte != b"\n":
                    raise ValueError(f"Component lacks final newline: {source_path}")
                if source_digest.hexdigest() != expected_hash:
                    raise ValueError(f"Component hash mismatch: {source_path}")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(output_tmp, output)
    except BaseException:
        if output_tmp.exists():
            output_tmp.unlink()
        raise

    final_report = {
        "name": "pretrain_specialist",
        "output": str(output),
        "format": "line-delimited JSON despite the requested .json extension",
        "components": [
            {
                "path": str(codespecialist),
                "rows": codespecialist_report["output_rows"],
                "bytes": codespecialist_report["output_bytes"],
                "sha256": expected_code_hash,
            },
            {
                "path": str(math_output),
                "rows": math_rows,
                "bytes": math_bytes,
                "sha256": math_digest.hexdigest(),
            },
        ],
        "output_rows": codespecialist_report["output_rows"] + math_rows,
        "output_bytes": output.stat().st_size,
        "output_sha256": final_digest.hexdigest(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
