"""Assemble recovered CPT caches into one deterministic, shuffled JSONL file."""
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
from scripts.data_builder.collect_continue_pretrain_1b import benchmark_sets, has_overlap, sha_file, MAX_TOKENS, NEW_TARGET, REPLAY_TARGET, SEED
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler
from scripts.data_builder.parallel_utils import batched, default_workers

CACHE = ROOT / "dataset/continue_pretrain_1b/filtered_cache"
PRIOR = ROOT / "dataset/pretrain_codespecialist.jsonl"
OUTPUT = ROOT / "dataset/pretrain_continue.jsonl"
REPORT = ROOT / "dataset/pretrain_continue.report.json"

NOVEL = (
    "python_edu_novel",
    "open_code_cot_novel",
    "verified_math_cot_novel",
    "cosmopedia_novel",
    "verified_math_code",
)
REPLAY = ("replay_code", "replay_general", "replay_math", "replay_technical")
SOURCES = NOVEL + REPLAY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    missing = [str(CACHE / f"{source}.jsonl") for source in SOURCES
               if not (CACHE / f"{source}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError("Missing completed caches: " + ", ".join(missing))

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True).backend_tokenizer
    he_grams, gsm_grams = benchmark_sets()
    benchmark_grams = he_grams | gsm_grams
    print("Indexing the prior corpus for exact novelty checks", flush=True)
    prior_hashes = set()
    prior_rows = 0
    with PRIOR.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            row = orjson.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Invalid prior text at line {line_number}")
            prior_hashes.add(hashlib.sha256(text.encode("utf-8")).digest())
            prior_rows += 1

    shuffler = ExternalRandomShuffler(OUTPUT, SEED, 16_000)
    seen_texts = set()
    component = {source: Counter() for source in SOURCES}
    rejected = {source: Counter() for source in SOURCES}
    licenses = Counter()
    executor = None
    try:
        executor = ThreadPoolExecutor(
            max_workers=max(1, args.workers), thread_name_prefix="dataset-assemble"
        )
        for source in SOURCES:
            path = CACHE / f"{source}.jsonl"
            with path.open("rb") as handle:
                for batch in batched(enumerate(handle, 1), args.batch_size):
                    parsed = []
                    for line_number, line in batch:
                        try:
                            row = orjson.loads(line)
                        except Exception as error:
                            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                        text = row.get("text")
                        if row.get("source") != source:
                            raise ValueError(f"Source mismatch at {path}:{line_number}")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f"Empty text at {path}:{line_number}")
                        tokens = row.get("token_count")
                        if not isinstance(tokens, int) or not 64 <= tokens <= MAX_TOKENS:
                            raise ValueError(f"Bad token_count at {path}:{line_number}: {tokens!r}")
                        parsed.append((line_number, line, row, text, tokens))
                    actual_counts = executor.map(
                        lambda item: len(tokenizer.encode(
                            item[3], add_special_tokens=False
                        ).ids) + 2,
                        parsed,
                    )
                    for (line_number, line, row, text, tokens), actual_tokens in zip(parsed, actual_counts):
                        if actual_tokens != tokens:
                            raise ValueError(
                                f"Stale token_count at {path}:{line_number}: stored={tokens}, actual={actual_tokens}"
                            )
                        fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
                        if fingerprint in seen_texts:
                            rejected[source]["duplicate_text"] += 1
                            rejected[source]["duplicate_tokens"] += tokens
                            continue
                        if source in NOVEL and fingerprint in prior_hashes:
                            rejected[source]["present_in_prior"] += 1
                            rejected[source]["present_in_prior_tokens"] += tokens
                            continue
                        if source in NOVEL and has_overlap(text, benchmark_grams):
                            rejected[source]["benchmark_overlap"] += 1
                            rejected[source]["benchmark_overlap_tokens"] += tokens
                            continue
                        if source == "python_edu_novel":
                            try:
                                ast.parse(text)
                            except (SyntaxError, ValueError):
                                rejected[source]["python_ast_failure"] += 1
                                rejected[source]["python_ast_failure_tokens"] += tokens
                                continue
                        elif source == "open_code_cot_novel":
                            marker = "### Verified Python solution\n```python\n"
                            if marker not in text or not text.endswith("\n```"):
                                rejected[source]["missing_solution"] += 1
                                rejected[source]["missing_solution_tokens"] += tokens
                                continue
                            solution = text.split(marker, 1)[1][:-4]
                            try:
                                ast.parse(solution)
                            except SyntaxError:
                                rejected[source]["solution_ast_failure"] += 1
                                rejected[source]["solution_ast_failure_tokens"] += tokens
                                continue
                        if source in REPLAY and fingerprint not in prior_hashes:
                            raise ValueError(f"Replay row absent from prior corpus at {path}:{line_number}")
                        seen_texts.add(fingerprint)
                        if not shuffler.add(row):
                            raise ValueError(f"Duplicate serialized row at {path}:{line_number}")
                        component[source]["rows"] += 1
                        component[source]["tokens"] += tokens
                        component[source]["bytes"] += len(line)
                        licenses[str(row.get("license") or "unspecified")] += 1
            print(f"accepted {source}: {component[source]['rows']:,} rows, "
                  f"{component[source]['tokens']:,} tokens", flush=True)

        novel_tokens = sum(component[source]["tokens"] for source in NOVEL)
        replay_tokens = sum(component[source]["tokens"] for source in REPLAY)
        # Collection was explicitly sealed at the existing cache size. Replay
        # remains the requested ~200M whole-record subset; novel data is
        # reported at its actual audited size rather than padded to 800M.
        if not REPLAY_TARGET - len(REPLAY) * MAX_TOKENS <= replay_tokens <= REPLAY_TARGET:
            raise AssertionError(f"Replay total outside tolerance: {replay_tokens:,}")

        print("Merging random-key chunks", flush=True)
        shuffler.finish()
    except BaseException:
        shuffler.temporary_dir.cleanup()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    total = novel_tokens + replay_tokens
    report = {
        "output": str(OUTPUT),
        "output_sha256": sha_file(OUTPUT),
        "rows": sum(value["rows"] for value in component.values()),
        "tokens": total,
        "novel_tokens": novel_tokens,
        "replay_tokens": replay_tokens,
        "model_parameters": 152_406_528,
        "tokens_per_parameter": total / 152_406_528,
        "collection_status": "sealed_existing_data_per_user",
        "original_requested_tokens": 1_000_000_000,
        "actual_sealed_tokens": total,
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "whole_records_only": True,
        "text_truncated": False,
        "shuffle_seed": SEED,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "prior_corpus": {"path": str(PRIOR), "rows": prior_rows,
                         "unique_text_hashes": len(prior_hashes)},
        "components": {name: dict(stats) for name, stats in component.items()},
        "rejected_during_final_assembly": {
            name: dict(stats) for name, stats in rejected.items() if stats
        },
        "component_token_fraction": {
            name: stats["tokens"] / total for name, stats in component.items()
        },
        "license_row_counts": dict(licenses),
        "checks": [
            "all cache rows parsed as JSON",
            "all text values are globally exact-deduplicated",
            "every novel exact text is absent from the prior corpus",
            "every replay exact text is present in the prior corpus",
            "all stored token counts are in the inclusive range 64..4096",
            "no record was split or truncated",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
