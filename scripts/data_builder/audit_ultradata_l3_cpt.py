"""Independently audit every assembled CPT row and sample true token lengths."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import orjson
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def audit(path):
    report_path = path.with_suffix(".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    excluded_ids = set()
    with Path(report["identity_clean"]["review_decisions"]).open("rb") as decisions:
        for line in decisions:
            decision = orjson.loads(line)
            if decision["decision"] == "drop":
                excluded_ids.add(decision["text_id"])
    tokenizer = Tokenizer.from_file(str(ROOT / "model/tokenizer.json"))
    if hashlib.sha256((ROOT / "model/tokenizer.json").read_bytes()).hexdigest() != report["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed after corpus selection")
    signature = path.stat()
    digest, seen_text, seen_exercise = hashlib.sha256(), set(), set()
    counts, categories = Counter(), Counter()
    minimum, maximum = 4096, 0
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            row = orjson.loads(line)
            text, token_count = row["text"], row["token_count"]
            if not isinstance(text, str) or not text.strip() or not isinstance(token_count, int):
                raise ValueError(f"Bad training schema at row {number}")
            if not 128 <= token_count <= 4096:
                raise ValueError(f"Token range violation at row {number}")
            fingerprint = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).digest()
            if fingerprint.hex() in excluded_ids:
                raise ValueError(f"Manually excluded record leaked into artifact at row {number}")
            if fingerprint.hex() != row["text_id"] or fingerprint in seen_text:
                raise ValueError(f"Text fingerprint mismatch/duplicate at row {number}")
            exercise = bytes.fromhex(row["exercise_id"])
            if exercise in seen_exercise:
                raise ValueError(f"Duplicated exercise at row {number}")
            seen_text.add(fingerprint)
            seen_exercise.add(exercise)
            if row["source_revision"] != report["source_revision"] or row["source_row"] < 1:
                raise ValueError(f"Bad provenance at row {number}")
            # Evenly spaced across the complete shuffled corpus, including the first row.
            if number == 1 or number % 1000 == 0:
                actual = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
                if actual != token_count:
                    raise ValueError(f"Stored token count differs from tokenizer at row {number}")
                counts["retokenized_rows"] += 1
            counts["rows"] += 1
            counts["tokens"] += token_count
            categories[row["mix_category"]] += token_count
            minimum, maximum = min(minimum, token_count), max(maximum, token_count)
            if number % 1_000_000 == 0:
                print(f"[audit] {number:,} rows; {counts['tokens']:,} tokens", flush=True)
    if digest.hexdigest() != report["output_sha256"]:
        raise ValueError("Output checksum differs from cleaning report")
    if counts["rows"] != report["rows"] or counts["tokens"] != report["tokens"]:
        raise ValueError("Output totals differ from cleaning report")
    expected = {k: v["tokens"] for k, v in report["components"].items()}
    if dict(categories) != expected:
        raise ValueError("Language totals differ from cleaning report")
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (signature.st_size, signature.st_mtime_ns):
        raise ValueError("Artifact changed during audit")
    result = {"artifact": str(path.resolve()), "output_sha256": digest.hexdigest(),
              "all_rows_schema_and_provenance_checked": True,
              "global_text_and_exercise_uniqueness_checked": True,
              "manual_exclusion_ids_checked": len(excluded_ids),
              "counts": dict(counts), "language_tokens": dict(categories),
              "min_tokens": minimum, "max_tokens": maximum, "passed": True,
              "scope": "All rows: integrity, schema, provenance, uniqueness and stored token bounds. Every 1000th row: independent true tokenizer length. Generated code correctness is not tested."}
    destination = path.with_suffix(".verification.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    audit(parser.parse_args().artifact)
