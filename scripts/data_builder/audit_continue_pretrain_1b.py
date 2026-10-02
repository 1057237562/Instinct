"""Independent full-record audit for dataset/pretrain_continue.jsonl."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import warnings

import orjson
from datasets import load_dataset  # noqa: F401  # before transformers/torch on Windows
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.data_builder.collect_continue_pretrain_1b import benchmark_sets, has_overlap, MAX_TOKENS, NEW_TARGET, REPLAY_TARGET
from scripts.data_builder.parallel_utils import batched, default_workers

DATA = ROOT / "dataset/pretrain_continue.jsonl"
MANIFEST = ROOT / "dataset/pretrain_continue.report.json"
OUTPUT = ROOT / "dataset/pretrain_continue.audit.json"
NOVEL = {
    "python_edu_novel", "open_code_cot_novel", "verified_math_cot_novel",
    "cosmopedia_novel", "verified_math_code",
}
REPLAY = {"replay_code", "replay_general", "replay_math", "replay_technical"}
ALLOWED = NOVEL | REPLAY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True).backend_tokenizer
    he_grams, gsm_grams = benchmark_sets()
    benchmark_grams = he_grams | gsm_grams
    counts = {source: Counter() for source in sorted(ALLOWED)}
    seen = set()
    file_sha = hashlib.sha256()
    total_rows = 0
    max_tokens = 0

    def validate_structure(item):
        line_number, line = item
        try:
            row = orjson.loads(line)
        except Exception as error:
            raise ValueError(f"Invalid JSON at line {line_number}") from error
        text, source, stored = row.get("text"), row.get("source"), row.get("token_count")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty text at line {line_number}")
        if source not in ALLOWED:
            raise ValueError(f"Unknown source at line {line_number}: {source!r}")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if source in NOVEL and has_overlap(text, benchmark_grams):
            raise ValueError(f"Benchmark n-gram overlap at line {line_number}")
        if source == "python_edu_novel":
            try:
                ast.parse(text)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Python AST failure at line {line_number}") from error
        elif source == "open_code_cot_novel":
            marker = "### Verified Python solution\n```python\n"
            if marker not in text or not text.endswith("\n```"):
                raise ValueError(f"Missing full Python solution at line {line_number}")
            solution = text.split(marker, 1)[1][:-4]
            try:
                ast.parse(solution)
            except SyntaxError as error:
                raise ValueError(f"Code-CoT solution AST failure at line {line_number}") from error
        return line_number, text, source, stored, digest

    with DATA.open("rb") as handle, ThreadPoolExecutor(
            max_workers=max(1, args.workers), thread_name_prefix="dataset-audit") as executor:
        for batch in batched(enumerate(handle, 1), args.batch_size):
            for _, line in batch:
                file_sha.update(line)
            checked = list(executor.map(validate_structure, batch))
            encoded = tokenizer.encode_batch(
                [item[1] for item in checked], add_special_tokens=False
            )
            for (line_number, _, source, stored, digest), encoding in zip(checked, encoded):
                if digest in seen:
                    raise ValueError(f"Duplicate text at line {line_number}")
                seen.add(digest)
                actual = len(encoding.ids) + 2
                if actual != stored:
                    raise ValueError(
                        f"Token mismatch at line {line_number}: stored={stored}, actual={actual}"
                    )
                if not 64 <= actual <= MAX_TOKENS:
                    raise ValueError(
                        f"Token length outside 64..4096 at line {line_number}: {actual}"
                    )
                counts[source]["rows"] += 1
                counts[source]["tokens"] += actual
                total_rows += 1
                max_tokens = max(max_tokens, actual)
                if total_rows % 100_000 == 0:
                    print(f"audited {total_rows:,} rows", flush=True)

    sha = file_sha.hexdigest()
    novel_tokens = sum(counts[source]["tokens"] for source in NOVEL)
    replay_tokens = sum(counts[source]["tokens"] for source in REPLAY)
    total_tokens = novel_tokens + replay_tokens
    if sha != manifest["output_sha256"]:
        raise AssertionError("Output SHA256 differs from manifest")
    if total_rows != manifest["rows"] or total_tokens != manifest["tokens"]:
        raise AssertionError("Independent totals differ from manifest")
    for source, expected in manifest["components"].items():
        if counts[source]["rows"] != expected["rows"] or counts[source]["tokens"] != expected["tokens"]:
            raise AssertionError(f"Component differs from manifest: {source}")
    if not REPLAY_TARGET - len(REPLAY) * MAX_TOKENS <= replay_tokens <= REPLAY_TARGET:
        raise AssertionError(f"Replay total outside tolerance: {replay_tokens}")

    report = {
        "status": "passed",
        "data": str(DATA),
        "output_sha256": sha,
        "rows": total_rows,
        "tokens": total_tokens,
        "novel_tokens": novel_tokens,
        "replay_tokens": replay_tokens,
        "max_tokens_including_bos_eos": max_tokens,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "components": {source: dict(value) for source, value in counts.items()},
        "checks": [
            "parsed every JSONL row",
            "recomputed every row token_count with the project tokenizer",
            "checked SHA-256 text uniqueness across every row",
            "screened every novel row against HumanEval and GSM8K-test 13-token n-grams",
            "parsed every Python educational file and every code-CoT final solution as Python AST",
            "matched independent totals and output SHA-256 to the assembly manifest",
        ],
        "replay_benchmark_note": "Replay inherits the prior corpus and is not claimed benchmark-clean.",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
