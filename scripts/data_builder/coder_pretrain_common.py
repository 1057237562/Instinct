"""Shared provenance, classification, and decontamination helpers for coder corpora."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "dataset/codespecialist.jsonl"
CONTINUED = ROOT / "dataset/pretrain_continue.jsonl"
TOKENIZER = ROOT / "model"
MAX_TOKENS = 4096
SEED = 20260918
BASE_CODE_SOURCES = {
    "open_repository_code_and_docs", "competitive_problem_reasoning",
    "verified_competitive_submissions", "code_instruction", "text_to_sql",
    "exercism_software_tasks",
}
CONTINUED_SOURCES = {
    "python_edu_novel", "open_code_cot_novel", "verified_math_cot_novel",
    "cosmopedia_novel", "verified_math_code",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|\d+", text.lower())


def ngrams(text: str, size: int = 13):
    tokens = words(text)
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def benchmark_ngrams():
    fingerprints = set()
    sources = {}
    humaneval_path = ROOT / "dataset/humaneval/HumanEval.jsonl.gz"
    with gzip.open(humaneval_path, "rt", encoding="utf-8") as stream:
        rows = 0
        for line in stream:
            row = json.loads(line)
            fingerprints.update(ngrams("\n".join(str(value) for value in row.values())))
            rows += 1
    sources["HumanEval"] = {"rows": rows, "path": str(humaneval_path)}

    gsm_path = ROOT / "dataset/gsm8k/test.jsonl"
    with gsm_path.open("r", encoding="utf-8") as stream:
        rows = 0
        for line in stream:
            fingerprints.update(ngrams(str(json.loads(line).get("question") or "")))
            rows += 1
    sources["GSM8K-test"] = {"rows": rows, "path": str(gsm_path)}

    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for row in mbpp:
        fingerprints.update(ngrams("\n".join([
            str(row.get("prompt") or ""), str(row.get("code") or ""),
            "\n".join(row.get("test_list") or []),
        ])))
    sources["MBPP-sanitized-test"] = {
        "rows": len(mbpp), "dataset": "google-research-datasets/mbpp",
        "config": "sanitized",
    }
    return fingerprints, sources


def has_benchmark_overlap(text: str, targets: set[tuple[str, ...]], size: int = 13) -> bool:
    tokens = words(text)
    return any(tuple(tokens[index:index + size]) in targets
               for index in range(len(tokens) - size + 1))


def classify_base(row: dict) -> str | None:
    source = row.get("source")
    if source in BASE_CODE_SOURCES:
        return "code"
    if source == "math_reasoning":
        return "math"
    if source in {"general_bilingual", "academic_technical"}:
        return "natural"
    return None


def classify_continued(row: dict) -> str | None:
    source = row.get("source")
    if source in {"python_edu_novel", "open_code_cot_novel"}:
        return "code"
    if source == "verified_math_cot_novel":
        return "math"
    if source == "cosmopedia_novel":
        return "natural"
    if source == "verified_math_code":
        return "code" if (row.get("metadata") or {}).get("domain") == "code" else "math"
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
