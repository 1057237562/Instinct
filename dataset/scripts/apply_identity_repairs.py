"""Apply identity-repair patches to a corpus and report what changed.

Patches were authored per row as exact-substring edits (and occasional whole-turn
regenerations). This tool replays them against the untouched source, validates
every edit, and writes a repaired corpus plus an audit of applied/skipped edits.
Rows without patches are copied byte for byte, so the repair is auditable and
the source stays intact.

Run from the repository root::

    python dataset/scripts/apply_identity_repairs.py dataset/sft_t2t_mini.jsonl \
        --patches dataset/review_candidates/_repair_batches \
        --output dataset/sft_t2t_mini.identity_repaired.jsonl
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.source_format import classify  # noqa: E402


def load_patches(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    patches: dict[int, dict[str, Any]] = {}
    problems: list[str] = []
    for path in paths:
        for number, line in enumerate(path.open(encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                problems.append(f"{path.name}:{number}: invalid JSON ({error.msg})")
                continue
            row_number = record.get("row_number")
            if not isinstance(row_number, int):
                problems.append(f"{path.name}:{number}: missing row_number")
                continue
            if row_number in patches:
                problems.append(f"{path.name}:{number}: duplicate patch for row {row_number}")
                continue
            patches[row_number] = record
    return patches, problems


def apply_edits(field: str, edits: list[dict[str, Any]], row_number: int, applied: collections.Counter, skipped: list[dict[str, Any]]) -> str:
    for edit in edits:
        old, new = edit.get("old", ""), edit.get("new", "")
        if not old or old == new:
            skipped.append({"row_number": row_number, "target": edit.get("target"), "reason": "empty_or_noop", "old": old[:120]})
            continue
        count = field.count(old)
        if count != 1:
            skipped.append({"row_number": row_number, "target": edit.get("target"), "reason": f"match_count_{count}", "old": old[:120]})
            continue
        if "... [TRUNCATED]" in old:
            skipped.append({"row_number": row_number, "target": edit.get("target"), "reason": "truncation_marker", "old": old[:120]})
            continue
        field = field.replace(old, new, 1)
        applied["edits"] += 1
    return field


def iter_records(path: Path) -> Iterator[tuple[int, str, Any]]:
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
                yield row_number, line, json.loads(line)
            except json.JSONDecodeError:
                yield row_number, line, None


def run(source: Path, patch_dir: Path, output: Path, report_path: Path, overwrite: bool) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    report_path = report_path.resolve()
    for path in (output, report_path):
        if path == source:
            raise ValueError("Output paths must not overwrite the input dataset.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    patch_files = sorted(patch_dir.glob("*.patches.jsonl"))
    if not patch_files:
        raise ValueError(f"No *.patches.jsonl files under {patch_dir}")
    patches, problems = load_patches(patch_files)

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    applied: collections.Counter = collections.Counter()
    skipped: list[dict[str, Any]] = []
    per_patch_file = collections.Counter()
    changed_rows: list[int] = []
    output_hash = hashlib.sha256()
    temp_output = output.with_name(output.name + ".tmp")
    try:
        with temp_output.open("wb") as stream:
            for row_number, raw_line, row in iter_records(source):
                applied["rows_read"] += 1
                patch = patches.pop(row_number, None)
                if patch is None or row is None:
                    encoded = raw_line.encode("utf-8") if raw_line else (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    stream.write(encoded)
                    output_hash.update(encoded)
                    continue
                conversations = row.get("conversations") or row.get("messages")
                if not isinstance(conversations, list):
                    skipped.append({"row_number": row_number, "reason": "no_conversations"})
                    encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    stream.write(encoded)
                    output_hash.update(encoded)
                    continue
                for edit in patch.get("edits", []):
                    index = edit.get("message_index")
                    target = edit.get("target", "content")
                    if not isinstance(index, int) or not (0 <= index < len(conversations)):
                        skipped.append({"row_number": row_number, "target": target, "reason": "bad_message_index", "index": index})
                        continue
                    message = conversations[index]
                    if target not in message and not isinstance(message.get(target), str):
                        skipped.append({"row_number": row_number, "target": target, "reason": "field_absent", "index": index})
                        continue
                    conversations[index][target] = apply_edits(message.get(target) or "", [edit], row_number, applied, skipped)
                for regeneration in patch.get("regenerate", []):
                    index = regeneration.get("message_index")
                    if not isinstance(index, int) or not (0 <= index < len(conversations)):
                        skipped.append({"row_number": row_number, "reason": "bad_message_index", "index": index})
                        continue
                    if isinstance(regeneration.get("content"), str):
                        conversations[index]["content"] = regeneration["content"]
                        applied["regenerated_turns"] += 1
                    if isinstance(regeneration.get("reasoning_content"), str):
                        conversations[index]["reasoning_content"] = regeneration["reasoning_content"]
                changed_rows.append(row_number)
                applied["rows_patched"] += 1
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                stream.write(encoded)
                output_hash.update(encoded)
        os.replace(temp_output, output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise

    for name in patch_files:
        per_patch_file[name.name] = sum(1 for _ in name.open(encoding="utf-8"))
    report = {
        "tool": "apply_identity_repairs",
        "version": 1,
        "source": {"path": str(source), "rows": applied["rows_read"]},
        "patches": {
            "files": [name.name for name in patch_files],
            "rows_with_patches": len(changed_rows),
            "load_problems": problems,
        },
        "applied": {
            "rows_patched": applied["rows_patched"],
            "edits_applied": applied["edits"],
            "turns_regenerated": applied["regenerated_turns"],
            "edits_skipped": len(skipped),
        },
        "skipped_detail": skipped[:200],
        "patches_never_matched": sorted(patches)[:200],
        "output": {"path": str(output), "rows": applied["rows_read"], "sha256": output_hash.hexdigest()},
        "policy": "Unpatched rows are copied byte for byte; only patched rows are re-serialized.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--patches", required=True, type=Path, help="Directory holding *.patches.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Default: <output>.report.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report_path = args.report or Path(str(args.output) + ".report.json")
    try:
        report = run(args.input, args.patches, args.output, report_path, args.overwrite)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
    print(json.dumps({k: report[k] for k in ("patches", "applied", "output")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
