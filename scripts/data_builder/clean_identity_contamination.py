"""Remove model-identity contamination from a chat/SFT corpus.

The audit in ``audit_identity_contamination.py`` measures contamination; this
script acts on it. Rows whose assistant turns claim another model's identity,
attach the model's own brand to a foreign vendor, or claim a commercial
developer/owner are dropped, and kept rows are copied byte for byte - no text is
rewritten, so no blind string replacement can invent facts.

The input is never modified. The output uses an ``identity_clean`` suffix and a
sidecar lists every dropped row with its reason and evidence, so the decision can
be reviewed or reversed.

Target identity: Instinct, developed by L1bra, affiliated with no commercial
organization.

Run from the repository root::

    python scripts/data_builder/clean_identity_contamination.py dataset/sft_t2t_mini.jsonl \
        --output dataset/sft_t2t_mini.identity_clean.jsonl \
        --dropped dataset/review_candidates/sft_t2t_mini.identity_dropped.jsonl
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_loader.source_format import classify, source_files  # noqa: E402
from scripts.data_builder.audit_identity_contamination import (  # noqa: E402
    TARGET_IDENTITY,
    classify_row,
    sha256_file,
)


# Each rule drops the whole row. A contaminated assistant target is
# identity-centric by construction, so rewriting it would mean authoring new
# supervision rather than cleaning existing data.
DROP_RULES = (
    ("foreign_model_self_identity", "assistant_content_identity_claims", "assistant_reasoning_identity_claims"),
    ("own_brand_attached_to_foreign_vendor", "assistant_brand_affiliation_facts"),
    ("commercial_developer_or_owner_claim", "assistant_commercial_affiliation_claims"),
    ("assistant_accepts_user_asserted_identity", "assistant_agreement_with_user_claim"),
)

SCHEMA_KEYS = ("conversations", "messages", "chosen", "rejected")


def decide(verdict: dict[str, Any]) -> str | None:
    """Return the drop rule that applies, or None to keep the row."""
    for rule, *keys in DROP_RULES:
        for key in keys:
            value = verdict.get(key)
            if isinstance(value, bool):
                if value:
                    return rule
            elif value:
                return rule
    return None


def iter_records(path: Path) -> Iterator[tuple[int, str, Any]]:
    """Yield (row_number, raw_line, parsed_row); raw_line is "" for parquet."""
    if classify(path) == "parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        row_number = 0
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, "", row
        return
    opener = gzip.open if str(path).lower().endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        for row_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            yield row_number, line, row


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(source: Path, output: Path, dropped_path: Path, report_path: Path, overwrite: bool) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    dropped_path = dropped_path.resolve()
    report_path = report_path.resolve()
    for path in (output, dropped_path, report_path):
        if path == source:
            raise ValueError("Output paths must not overwrite the input dataset.")
    if any(path.exists() for path in (output, dropped_path, report_path)) and not overwrite:
        raise FileExistsError("Output exists; pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counts: collections.Counter[str] = collections.Counter()
    dropped_by_rule: collections.Counter[str] = collections.Counter()
    terms: collections.Counter[str] = collections.Counter()
    output_hash = hashlib.sha256()
    kept = dropped = 0
    temp_output = output.with_name(output.name + ".tmp")
    temp_dropped = dropped_path.with_name(dropped_path.name + ".tmp")
    try:
        with temp_output.open("wb") as out_stream, temp_dropped.open("wb") as drop_stream:
            for row_number, raw_line, row in iter_records(source):
                counts["rows_read"] += 1
                if row is None:
                    counts["dropped_invalid_json"] += 1
                    dropped += 1
                    continue
                if not any(key in row for key in SCHEMA_KEYS):
                    counts["rows_without_chat_schema"] += 1
                verdict = classify_row(row)
                rule = decide(verdict) if verdict else None
                if verdict:
                    terms.update(verdict["qwen_terms"])
                if rule:
                    dropped += 1
                    dropped_by_rule[rule] += 1
                    if verdict:
                        if verdict["assistant_content_identity_claims"]:
                            counts["dropped_content_claim"] += 1
                        if verdict["assistant_reasoning_identity_claims"]:
                            counts["dropped_reasoning_claim"] += 1
                    evidence = {
                        key: value for key, value in (verdict or {}).items()
                        if key not in {"qwen_terms", "hard_claim", "reasoning_only", "mention_only"}
                    }
                    record = {
                        "_review": {
                            "source": str(source),
                            "row_number": row_number,
                            "rule": rule,
                            "target_identity": TARGET_IDENTITY,
                            "evidence": evidence,
                        },
                        "record": row,
                    }
                    drop_stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                    continue
                kept += 1
                # Byte-identical passthrough: cleaning never rewrites kept text.
                encoded = raw_line.encode("utf-8") if raw_line else (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                out_stream.write(encoded)
                output_hash.update(encoded)
        os.replace(temp_output, output)
        os.replace(temp_dropped, dropped_path)
    except Exception:
        temp_output.unlink(missing_ok=True)
        temp_dropped.unlink(missing_ok=True)
        raise

    report = {
        "tool": "clean_identity_contamination",
        "version": 1,
        "target_identity": TARGET_IDENTITY,
        "repository_revision": git_revision(),
        "source": {"path": str(source), "rows": counts["rows_read"], "sha256": sha256_file(source)},
        "output": {
            "path": str(output),
            "rows": kept,
            "sha256": output_hash.hexdigest(),
            "policy": "contaminated rows dropped; kept rows copied byte for byte",
        },
        "dropped": {
            "path": str(dropped_path),
            "rows": dropped,
            "by_rule": dict(sorted(dropped_by_rule.items())),
            "invalid_json": counts["dropped_invalid_json"],
        },
        "schema": {"rows_without_chat_schema": counts["rows_without_chat_schema"]},
        "identity_keyword_hits": dict(sorted(terms.items(), key=lambda item: -item[1])),
        "review_only_kept": "Rows flagged for review without a drop rule (speculation, role mentions, factual "
                            "third-party discussion) are kept; see the audit's review bucket.",
        "decision": "The source dataset was not modified.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL/.jsonl.gz/.parquet corpus")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned corpus path")
    parser.add_argument("--dropped", type=Path, help="Sidecar listing dropped rows (default: <output>.dropped.jsonl)")
    parser.add_argument("--report", type=Path, help="Report path (default: <output>.report.json)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args()
    if "identity_clean" not in args.output.name:
        raise SystemExit("Output name must carry an explicit identity_clean suffix.")
    dropped = args.dropped or Path(str(args.output) + ".dropped.jsonl")
    report = args.report or Path(str(args.output) + ".report.json")
    try:
        result = run(args.input, args.output, dropped, report, args.overwrite)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
    print(json.dumps({k: result[k] for k in ("source", "output", "dropped")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
