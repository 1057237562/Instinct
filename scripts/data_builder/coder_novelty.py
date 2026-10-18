"""Read-only prior-corpus fingerprints and review triage for novel coder data.

The fingerprint index excludes complete texts after trimming outer whitespace.
It does not establish semantic novelty. Identity hits are review candidates,
never automatic deletion decisions; collected shards remain unapproved raw data.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import orjson


def text_fingerprint(text: str) -> bytes:
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).digest()


def build_exclusion_index(source: Path, destination: Path, source_report: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    report_path = destination.with_suffix(destination.suffix + ".report.json")
    if destination.exists() and report_path.exists():
        index = ExclusionIndex(destination)
        return index.report
    expected = json.loads(source_report.read_text(encoding="utf-8"))
    expected_hash = expected["output_sha256"]
    signature = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    rows = 0
    with source.open("rb") as stream, temporary.open("wb") as output:
        for line in stream:
            digest.update(line)
            row = orjson.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid baseline text at row {rows + 1}")
            output.write(text_fingerprint(text))
            rows += 1
            if rows % 500_000 == 0:
                print(f"[baseline index] {rows:,} rows", flush=True)
        output.flush()
        os.fsync(output.fileno())
    if digest.hexdigest() != expected_hash or rows != expected["rows"]:
        raise ValueError("Baseline SHA-256/row count differs from its release report")
    after = source.stat()
    if (signature.st_size, signature.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Baseline changed while indexing")
    fingerprints = np.fromfile(temporary, dtype="V16")
    fingerprints.sort()
    fingerprints = np.unique(fingerprints)
    fingerprints.tofile(temporary)
    index_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
    report = {
        "format": "sorted-blake2b-128-v1", "normalization": "outer whitespace stripped",
        "source": str(source), "source_report": str(source_report.resolve()),
        "source_sha256": digest.hexdigest(), "source_rows": rows,
        "source_bytes": signature.st_size, "source_mtime_ns": signature.st_mtime_ns,
        "fingerprints": len(fingerprints), "index_bytes": temporary.stat().st_size,
        "index_sha256": index_hash,
    }
    os.replace(temporary, destination)
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(report_tmp, report_path)
    print(f"[baseline index] ready: {len(fingerprints):,} fingerprints", flush=True)
    return report


def build_merged_exclusion_index(base_path: Path, additional_sources: list[dict], destination: Path) -> dict:
    """Extend a verified prior index with immutable, hash-audited raw shards."""
    base = ExclusionIndex(base_path)
    destination = destination.resolve()
    report_path = destination.with_suffix(destination.suffix + ".report.json")
    base_sources = base.report.get("sources") or [{
        "path": base.report["source"], "bytes": base.report["source_bytes"],
        "mtime_ns": base.report["source_mtime_ns"], "sha256": base.report["source_sha256"],
        "rows": base.report["source_rows"],
    }]
    additions = []
    for source in additional_sources:
        path = Path(source["path"]).resolve()
        stat = path.stat()
        additions.append({**source, "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    sources = base_sources + additions
    manifest_hash = hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if destination.exists() and report_path.exists():
        existing = ExclusionIndex(destination)
        if existing.report.get("source_sha256") != manifest_hash:
            raise ValueError("Prior-corpus manifest changed; use a new exclusion index")
        return existing.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    added_rows = 0
    with temporary.open("wb") as output:
        with base.path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                output.write(block)
        for source in additions:
            path = Path(source["path"])
            digest = hashlib.sha256()
            rows = 0
            with path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    row = orjson.loads(line)
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Invalid prior text: {path}:{rows + 1}")
                    output.write(text_fingerprint(text))
                    rows += 1
                    if (added_rows + rows) % 500_000 == 0:
                        print(f"[prior index] {added_rows + rows:,} additional rows", flush=True)
            if digest.hexdigest() != source["sha256"] or rows != source["rows"]:
                raise ValueError(f"Prior shard differs from its audit report: {path}")
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
                raise ValueError(f"Prior shard changed while indexing: {path}")
            added_rows += rows
            print(f"[prior index] verified {path.name}: {rows:,} rows", flush=True)
        output.flush()
        os.fsync(output.fileno())
    values = np.unique(np.fromfile(temporary, dtype="V16"))
    values.tofile(temporary)
    index_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
    report = {
        "format": "sorted-blake2b-128-v2", "normalization": "outer whitespace stripped",
        "sources": sources, "source_sha256": manifest_hash,
        "source_rows": sum(item["rows"] for item in sources),
        "base_index_sha256": base.report["index_sha256"],
        "fingerprints": len(values), "index_bytes": temporary.stat().st_size,
        "index_sha256": index_hash,
    }
    os.replace(temporary, destination)
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    report_temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(report_temporary, report_path)
    print(f"[prior index] ready: {len(values):,} fingerprints across {len(sources)} prior files", flush=True)
    return report


class ExclusionIndex:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.report = json.loads(self.path.with_suffix(self.path.suffix + ".report.json").read_text(encoding="utf-8"))
        if self.report["format"] not in ("sorted-blake2b-128-v1", "sorted-blake2b-128-v2"):
            raise ValueError("Unsupported exclusion index format")
        sources = self.report.get("sources") or [{"path": self.report["source"], "bytes": self.report["source_bytes"], "mtime_ns": self.report["source_mtime_ns"]}]
        for source in sources:
            stat = Path(source["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
                raise ValueError("Baseline changed since exclusion index creation")
        if self.path.stat().st_size != self.report["fingerprints"] * 16:
            raise ValueError("Invalid exclusion index size")
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.report["index_sha256"]:
            raise ValueError("Exclusion index checksum mismatch")
        self.values = np.memmap(self.path, dtype="V16", mode="r") if self.report["fingerprints"] else np.empty(0, dtype="V16")

    def contains(self, text: str) -> bool:
        value = np.void(text_fingerprint(text))
        position = int(np.searchsorted(self.values, value))
        return position < len(self.values) and bool(self.values[position] == value)


class IdentityReview:
    """Apply first-pass triage and retain every keyword hit for review."""

    def __init__(self):
        from scripts.data_builder.filter_anomaly_candidates import inspect_row
        from scripts.data_builder.audit_identity_contamination import VENDOR_ANY_RE, _claim_hits
        self.inspect_row = inspect_row
        self.terms = VENDOR_ANY_RE
        self.claim_hits = _claim_hits

    def inspect(self, row: dict) -> tuple[list[dict], Counter]:
        text = row["text"]
        hits = Counter(match.group(0).lower() for match in self.terms.finditer(text))
        reasons = self.inspect_row(row, "all")
        if hits:
            reasons.extend(self.inspect_row({"conversations": [{"role": "assistant", "content": text}]}, "identity"))
            reasons.append({"rule": "foreign_model_or_provider_mention", "detail": "Review factual mentions, identity claims, promotional/model-card text, and embedded Q&A in context; do not blindly replace names."})
            for claim in self.claim_hits(text):
                reasons.append({"rule": "possible_foreign_identity_" + claim["tier"], "detail": claim["snippet"]})
        return reasons, hits
