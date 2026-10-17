"""Reconstruct the verified_competitive_submissions component from open-r1 raws.

The original component was carried forward between mix generations by copying
rows out of the previous mix (build_codespecialist_mix.py --verified-submissions
default). That prior mix no longer exists locally, so this script rebuilds the
component in kind from the documented upstream: accepted (verdict OK) Codeforces
submissions paired with their problem statements, chat-rendered, whole-record,
<=4096 tokens. Marked as a reconstruction in the report.
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

PROBLEMS_REV = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
SUBMISSIONS_REV = "downloaded main via hf-mirror on rebuild"
SOURCE = "verified_competitive_submissions"
TARGET_TOKENS = 115_000_000
MAX_TOKENS = 4096
CAP_PER_PROBLEM_LANGUAGE = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems-dir", type=Path, default=ROOT / "dataset/codeforces-problems/data")
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "dataset/codeforces-submissions/data")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/pretrain_verified_submissions_115m.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "dataset/pretrain_verified_submissions_115m.report.json")
    return parser.parse_args()


def problem_prompt(problem: dict) -> str:
    parts = [
        f"Solve this competitive programming problem using Python or C++. "
        f"Read from standard input and write to standard output. "
        f"Return a complete solution.\n\n{problem.get('title', '')}\n\n{problem.get('description', '')}"
    ]
    for field, title in (("input_format", "Input"), ("output_format", "Output")):
        if problem.get(field):
            parts.append(f"\n\n{title}\n{problem[field]}")
    for example in problem.get("examples") or []:
        parts.append(
            f"\n\nExample input\n{example.get('input', '')}\n"
            f"Example output\n{example.get('output', '')}"
        )
    if problem.get("note"):
        parts.append(f"\n\nNote\n{problem['note']}")
    return "".join(parts)


def load_problems(directory: Path):
    """Return (problems indexed by canonical id and alias, test-split ids)."""
    problems = {}
    test_ids = set()
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No problem parquets under {directory}")
    for path in paths:
        is_test = path.name.startswith("test")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if is_test:
                    test_ids.update([row["id"]] + list(row.get("aliases") or []))
                    continue
                problems[row["id"]] = row
                for alias in row.get("aliases") or []:
                    problems.setdefault(alias, row)
    return problems, test_ids


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("output/report already exists")
    problems, test_ids = load_problems(args.problems_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    backend = tokenizer.backend_tokenizer
    stats = Counter()
    seen_code = set()
    per_key = Counter()
    rows = tokens = 0
    submissions = sorted(args.submissions_dir.glob("*.parquet"))
    if not submissions:
        raise FileNotFoundError(f"No submission parquets under {args.submissions_dir}")
    try:
        with args.output.open("xb", buffering=8 * 1024 * 1024) as destination:
            for path in submissions:
                if tokens >= TARGET_TOKENS - MAX_TOKENS:
                    break
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=2048,
                    columns=["problem_id", "source", "programmingLanguage", "verdict", "testset", "submission_id"],
                ):
                    if tokens >= TARGET_TOKENS - MAX_TOKENS:
                        break
                    for row in batch.to_pylist():
                        stats["scanned"] += 1
                        if row["verdict"] != "OK":
                            stats["rejected_verdict"] += 1
                            continue
                        if row["testset"] != "TESTS":
                            stats["rejected_testset"] += 1
                            continue
                        problem = problems.get(row["problem_id"])
                        if problem is None or problem.get("id") in test_ids:
                            stats["rejected_problem_unavailable_or_test"] += 1
                            continue
                        language = row["programmingLanguage"] or ""
                        language = "python" if ("Python" in language or "PyPy" in language) else "cpp" if "C++" in language else None
                        if not language:
                            stats["rejected_language"] += 1
                            continue
                        code = row["source"]
                        if not code or not code.strip():
                            stats["rejected_empty_code"] += 1
                            continue
                        key = (row["problem_id"], language)
                        if per_key[key] >= CAP_PER_PROBLEM_LANGUAGE:
                            stats["rejected_per_problem_cap"] += 1
                            continue
                        code_hash = hashlib.sha256(code.encode()).hexdigest()
                        if code_hash in seen_code:
                            stats["rejected_duplicate_code"] += 1
                            continue
                        prompt = problem_prompt(problem)
                        if prompt in ("<empty>",):
                            stats["rejected_empty_prompt"] += 1
                            continue
                        text = (
                            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                            "<|im_start|>assistant\n" + f"```{language}\n{code}\n```" + "<|im_end|>\n"
                        )
                        token_count = len(backend.encode(text, add_special_tokens=False).ids) + 2
                        if not 64 <= token_count <= MAX_TOKENS:
                            stats["rejected_outside_token_range"] += 1
                            continue
                        if tokens + token_count > TARGET_TOKENS:
                            stats["rejected_over_budget"] += 1
                            continue
                        contest, _, letter = str(row["problem_id"]).partition("/")
                        record = {
                            "text": text,
                            "source": SOURCE,
                            "source_id": f"codeforces:{row['submission_id']}",
                            "license": "ODC-By",
                            "url": f"https://codeforces.com/contest/{contest}/problem/{letter}",
                            "token_count": token_count,
                            "language": language,
                            "rating": problem.get("rating"),
                        }
                        destination.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                        seen_code.add(code_hash)
                        per_key[key] += 1
                        rows += 1
                        tokens += token_count
                print(f"{path.name}: rows={rows:,} tokens={tokens:,}", flush=True)
    except BaseException:
        if args.output.exists() and rows == 0:
            args.output.unlink()
        raise

    report = {
        "name": "verified_competitive_submissions_reconstruction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "problems": "open-r1/codeforces",
            "problems_revision": PROBLEMS_REV,
            "submissions": "open-r1/codeforces-submissions",
            "submissions_revision": SUBMISSIONS_REV,
            "note": "original component was copied between mix generations from a prior mix that is no longer available; rebuilt in kind from upstream accepted submissions",
        },
        "output": str(args.output), "rows": rows, "tokens": tokens,
        "selection_stats": dict(sorted(stats.items())),
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "cap_per_problem_language": CAP_PER_PROBLEM_LANGUAGE,
        "languages": {"python": None, "cpp": None},
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
