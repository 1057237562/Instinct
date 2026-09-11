"""Stream complete, openly licensed arXiv papers into Instinct JSONL.

The source corpus is sharded chronologically.  This collector takes roughly the
same byte budget from every shard so a small local subset is not dominated by
the oldest papers.  Each source paper remains one complete JSONL record: the
paper text is neither normalized, tokenized, truncated, nor split.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import orjson


DEFAULT_REPO = "common-pile/arxiv_papers"
USER_AGENT = "Instinct-arxiv-collector/1.0"


class CountingReader:
    """Count compressed bytes consumed by ``gzip.GzipFile``."""

    def __init__(self, raw) -> None:
        self.raw = raw
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.bytes_read += len(data)
        return data

    def close(self) -> None:
        self.raw.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=500_000_000,
        help="Approximate final JSONL size; whole papers may leave a small gap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/pretrain_arxiv_open_500mb.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/pretrain_arxiv_open_500mb.report.json"),
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="Use only the first N chronological shards (0 means all shards).",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def classify_license(value: object) -> str | None:
    """Map source license labels to the three licenses promised by the corpus."""

    if not isinstance(value, str):
        return None
    lowered = value.casefold()
    if "public domain" in lowered:
        return "Public Domain"
    if "cc0" in lowered or "publicdomain/zero" in lowered or "zero/1.0" in lowered:
        return "CC0"
    if "attribution share-alike" in lowered or "/by-sa/" in lowered:
        return "CC BY-SA"
    if "creative commons - attribution" in lowered or "/licenses/by/" in lowered:
        return "CC BY"
    return None


def quota_end(target_bytes: int, shard_index: int, shard_count: int) -> int:
    """Return the cumulative output budget after a given shard."""

    return target_bytes * (shard_index + 1) // shard_count


def _request_json(url: str, timeout: int) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return orjson.loads(response.read())


def discover_shards(repo: str, revision: str, timeout: int) -> list[dict]:
    encoded_repo = "/".join(quote(part, safe="") for part in repo.split("/"))
    encoded_revision = quote(revision, safe="")
    url = (
        f"https://huggingface.co/api/datasets/{encoded_repo}/tree/"
        f"{encoded_revision}?recursive=true&expand=false&limit=1000"
    )
    manifest = _request_json(url, timeout)
    if not isinstance(manifest, list):
        raise ValueError("Unexpected Hugging Face repository manifest")
    shards = [
        item
        for item in manifest
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and item["path"].endswith(".jsonl.gz")
    ]
    shards.sort(key=lambda item: item["path"])
    if not shards:
        raise FileNotFoundError(f"No .jsonl.gz shards found in {repo}@{revision}")
    return shards


def shard_url(repo: str, revision: str, path: str) -> str:
    encoded_repo = "/".join(quote(part, safe="") for part in repo.split("/"))
    encoded_revision = quote(revision, safe="")
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
    return (
        f"https://huggingface.co/datasets/{encoded_repo}/resolve/"
        f"{encoded_revision}/{encoded_path}"
    )


def make_output_record(source: dict) -> dict:
    """Keep full source text and the attribution metadata needed to trace it."""

    return {
        "text": source["text"],
        "arxiv_id": source.get("id"),
        "source": source.get("source", "arxiv-papers"),
        "created": source.get("created"),
        "metadata": source.get("metadata") or {},
    }


def main() -> None:
    args = parse_args()
    if args.target_bytes <= 0 or args.timeout <= 0 or args.max_shards < 0:
        raise ValueError("target-bytes and timeout must be positive; max-shards >= 0")

    output = args.output.resolve()
    report_path = args.report.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    if not args.overwrite:
        for path in (output, report_path):
            if path.exists():
                raise FileExistsError(path)

    shards = discover_shards(args.repo, args.revision, args.timeout)
    if args.max_shards:
        shards = shards[: args.max_shards]

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    license_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    shard_reports: list[dict] = []
    output_bytes = 0
    output_rows = 0
    text_utf8_bytes = 0
    min_text_chars: int | None = None
    max_text_chars = 0

    try:
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination:
            for shard_index, shard in enumerate(shards):
                path = shard["path"]
                budget_end = quota_end(args.target_bytes, shard_index, len(shards))
                shard_rows = 0
                shard_output_bytes = 0
                shard_scanned = 0
                compressed_read = 0
                request = Request(
                    shard_url(args.repo, args.revision, path),
                    headers={"User-Agent": USER_AGENT},
                )
                with urlopen(request, timeout=args.timeout) as response:
                    counted = CountingReader(response)
                    with gzip.GzipFile(fileobj=counted, mode="rb") as archive:
                        for raw_line in archive:
                            shard_scanned += 1
                            source = orjson.loads(raw_line)
                            if not isinstance(source, dict):
                                rejects["not_an_object"] += 1
                                continue
                            text = source.get("text")
                            if not isinstance(text, str) or not text:
                                rejects["empty_text"] += 1
                                continue
                            metadata = source.get("metadata")
                            license_value = (
                                metadata.get("license")
                                if isinstance(metadata, dict)
                                else None
                            )
                            license_name = classify_license(license_value)
                            if license_name is None:
                                rejects["unexpected_license"] += 1
                                continue

                            line = orjson.dumps(
                                make_output_record(source),
                                option=orjson.OPT_APPEND_NEWLINE,
                            )
                            # Never split or truncate a paper merely to hit the
                            # requested byte target exactly.
                            if output_bytes + len(line) > budget_end:
                                break

                            destination.write(line)
                            output_digest.update(line)
                            output_bytes += len(line)
                            shard_output_bytes += len(line)
                            output_rows += 1
                            shard_rows += 1
                            current_text_bytes = len(text.encode("utf-8"))
                            text_utf8_bytes += current_text_bytes
                            text_chars = len(text)
                            min_text_chars = (
                                text_chars
                                if min_text_chars is None
                                else min(min_text_chars, text_chars)
                            )
                            max_text_chars = max(max_text_chars, text_chars)
                            license_counts[license_name] += 1
                            created = source.get("created")
                            year = (
                                created[:4]
                                if isinstance(created, str) and len(created) >= 4
                                else "unknown"
                            )
                            year_counts[year] += 1
                    compressed_read = counted.bytes_read

                shard_reports.append(
                    {
                        "path": path,
                        "source_compressed_bytes": shard.get("size"),
                        "compressed_bytes_read": compressed_read,
                        "rows_scanned": shard_scanned,
                        "rows_written": shard_rows,
                        "output_bytes": shard_output_bytes,
                    }
                )
                print(
                    f"[{shard_index + 1}/{len(shards)}] {path}: "
                    f"papers={shard_rows:,}, total={output_bytes / 1_000_000:.1f} MB",
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
        "name": "pretrain_arxiv_open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "format": "JSON Lines; one complete paper per row; text is not modified",
        "source": {
            "repository": args.repo,
            "revision": args.revision,
            "url": f"https://huggingface.co/datasets/{args.repo}",
            "shards_used": len(shards),
            "source_compressed_bytes": sum(
                int(shard.get("size") or 0) for shard in shards
            ),
        },
        "selection": {
            "method": "equal cumulative output-byte quota per chronological shard",
            "target_bytes": args.target_bytes,
            "whole_papers_only": True,
            "text_normalized": False,
            "text_truncated": False,
            "text_split": False,
        },
        "output_rows": output_rows,
        "output_bytes": output_bytes,
        "text_utf8_bytes": text_utf8_bytes,
        "output_sha256": output_digest.hexdigest(),
        "min_text_characters": min_text_chars or 0,
        "max_text_characters": max_text_chars,
        "licenses": dict(sorted(license_counts.items())),
        "years": dict(sorted(year_counts.items())),
        "rejected": dict(sorted(rejects.items())),
        "shards": shard_reports,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
