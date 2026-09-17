"""Collect a balanced, high-quality open-source code slice from Common Pile."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import orjson

if __package__ in (None, ""):
    __package__ = "scripts"
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataset.scripts.collect_arxiv_pretrain import CountingReader, shard_url, USER_AGENT


REPO = "common-pile/stackv2_edu_filtered"
REVISION = "c354dbe88469a1153e97c6a63ac50591849654de"

# Text-byte targets sum to 750 MB.  Shards are grouped by language in this
# dataset; multiple shards are used for large languages to improve diversity.
LANGUAGE_PLAN = {
    "Markdown": (112_500_000, (4, 7, 10)),
    "Python": (150_000_000, (73, 78, 84)),
    "C++": (112_500_000, (20, 24, 27)),
    "JavaScript": (90_000_000, (50, 57, 64)),
    "TypeScript": (60_000_000, (91, 94)),
    "Java": (75_000_000, (38, 43, 49)),
    "Go": (45_000_000, (34, 36)),
    "Rust": (30_000_000, (86,)),
    "C": (30_000_000, (18, 19)),
    "C#": (22_500_000, (28, 30)),
    "Shell": (22_500_000, (87, 88)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/pretrain_stackv2_open_750mb.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/pretrain_stackv2_open_750mb.report.json"),
    )
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def make_record(row: dict, language: str) -> dict:
    metadata = row.get("metadata") or {}
    return {
        "text": row["text"],
        "source": f"{REPO}:{language}",
        "source_id": str(row.get("id") or ""),
        "license": str(metadata.get("license") or ""),
        "url": str(metadata.get("url") or ""),
    }


def main() -> None:
    args = parse_args()
    if args.min_score < 1 or args.min_characters < 1 or args.timeout < 1:
        raise ValueError("score, character, and timeout limits must be positive")
    output = args.output.resolve()
    report_path = args.report.resolve()
    temporary = output.with_name(output.name + ".tmp")
    for path in (output, report_path, temporary):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    stats: Counter[str] = Counter()
    language_reports: dict[str, dict] = {}
    output_bytes = 0
    text_bytes = 0
    rows = 0

    try:
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination:
            for language, (target, shard_numbers) in LANGUAGE_PLAN.items():
                language_text_bytes = 0
                language_output_bytes = 0
                language_rows = 0
                shards_report = []
                for shard_index, shard_number in enumerate(shard_numbers):
                    path = f"stack-edu-{shard_number:04d}.json.gz"
                    cumulative_target = target * (shard_index + 1) // len(shard_numbers)
                    scanned = 0
                    compressed_bytes = 0
                    request = Request(
                        shard_url(REPO, REVISION, path),
                        headers={"User-Agent": USER_AGENT},
                    )
                    with urlopen(request, timeout=args.timeout) as response:
                        counted = CountingReader(response)
                        with gzip.GzipFile(fileobj=counted, mode="rb") as archive:
                            for raw_line in archive:
                                scanned += 1
                                row = orjson.loads(raw_line)
                                metadata = row.get("metadata") or {}
                                if metadata.get("language") != language:
                                    stats["rejected_language_mismatch"] += 1
                                    continue
                                if int(row.get("int_score") or 0) < args.min_score:
                                    stats["rejected_score"] += 1
                                    continue
                                if metadata.get("is_vendor") or metadata.get("is_generated"):
                                    stats["rejected_vendor_or_generated"] += 1
                                    continue
                                text = row.get("text")
                                if not isinstance(text, str) or len(text) < args.min_characters:
                                    stats["rejected_short_or_empty"] += 1
                                    continue
                                source_id = str(row.get("id") or "")
                                if not source_id or source_id in seen_ids:
                                    stats["rejected_duplicate_id"] += 1
                                    continue
                                current_text_bytes = len(text.encode("utf-8"))
                                remaining = cumulative_target - language_text_bytes
                                if current_text_bytes > remaining:
                                    stats["rejected_over_language_budget"] += 1
                                    if remaining < 64_000:
                                        break
                                    continue
                                line = orjson.dumps(
                                    make_record(row, language),
                                    option=orjson.OPT_APPEND_NEWLINE,
                                )
                                destination.write(line)
                                digest.update(line)
                                seen_ids.add(source_id)
                                rows += 1
                                language_rows += 1
                                output_bytes += len(line)
                                language_output_bytes += len(line)
                                text_bytes += current_text_bytes
                                language_text_bytes += current_text_bytes
                        compressed_bytes = counted.bytes_read
                    shards_report.append(
                        {
                            "path": path,
                            "rows_scanned": scanned,
                            "compressed_bytes_read": compressed_bytes,
                        }
                    )
                language_reports[language] = {
                    "target_text_bytes": target,
                    "text_bytes": language_text_bytes,
                    "output_bytes": language_output_bytes,
                    "rows": language_rows,
                    "shards": shards_report,
                }
                print(
                    f"{language}: {language_text_bytes / 1_000_000:.1f}/"
                    f"{target / 1_000_000:.1f} MB, rows={language_rows:,}",
                    flush=True,
                )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    report = {
        "name": "pretrain_stackv2_open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO,
            "revision": REVISION,
            "url": f"https://huggingface.co/datasets/{REPO}",
        },
        "selection": {
            "minimum_educational_score": args.min_score,
            "minimum_characters": args.min_characters,
            "exclude_vendor": True,
            "exclude_generated": True,
            "whole_documents_only": True,
            "text_truncated": False,
            "text_split": False,
        },
        "output": str(output),
        "output_rows": rows,
        "output_bytes": output_bytes,
        "text_utf8_bytes": text_bytes,
        "output_sha256": digest.hexdigest(),
        "languages": language_reports,
        "selection_stats": dict(sorted(stats.items())),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
