"""Bounded JSONL chunk plans for cache-constrained pretraining.

The source corpus stays untouched.  A plan records newline-aligned byte ranges
and token totals; only one range is materialized into the managed Arrow cache at
a time.  Plans are tiny, deterministic, and safe to reuse after a resume.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from filelock import FileLock


_TOKEN_COUNT = re.compile(rb'"token_count"\s*:\s*(\d+)')
_PLAN_VERSION = 2


def _plan_identity(path: Path, chunk_bytes: int, max_length: int) -> dict:
    stat = path.stat()
    sample = hashlib.sha256()
    with path.open("rb") as stream:
        sample.update(stream.read(64 * 1024))
        if stat.st_size > 64 * 1024:
            stream.seek(max(0, stat.st_size - 64 * 1024))
            sample.update(stream.read(64 * 1024))
    return {
        "version": _PLAN_VERSION,
        "source": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": sample.hexdigest(),
        "chunk_bytes": int(chunk_bytes),
        "max_length": int(max_length),
    }


def _default_plan_dir() -> Path:
    root = Path(os.environ.get(
        "INSTINCT_DATA_CACHE_BUDGET_ROOT",
        os.environ.get("HF_DATASETS_CACHE", Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "datasets"),
    )).resolve()
    return root / "instinct-stream-plans"


def _row_tokens(line: bytes, tokenizer, max_length: int) -> int:
    # Instinct writers place metadata after ``text``. Use the final occurrence
    # so source code containing the literal string "token_count" cannot spoof
    # the planning total.
    position = line.rfind(b'"token_count"')
    match = _TOKEN_COUNT.match(line, position) if position >= 0 else None
    if match:
        return min(int(match.group(1)), max_length)
    try:
        import orjson
        row = orjson.loads(line)
    except ImportError:
        row = json.loads(line)
    text = str(row.get("text", ""))
    tokens = tokenizer(
        text, add_special_tokens=False,
        max_length=max_length - 2, truncation=True,
    ).input_ids
    return min(len(tokens) + 2, max_length)


def build_jsonl_chunk_plan(
    data_path,
    *,
    chunk_bytes: int,
    max_length: int,
    tokenizer,
    plan_dir=None,
) -> dict:
    """Scan one JSONL sequentially and cache newline-aligned chunk metadata.

    Corpora produced by Instinct contain ``token_count`` so the scan does not
    tokenize or decode their large text fields.  Generic JSONL remains
    supported by falling back to the supplied tokenizer.
    """
    source = Path(data_path).resolve()
    chunk_bytes = int(chunk_bytes)
    max_length = int(max_length)
    if not source.is_file():
        raise FileNotFoundError(source)
    if chunk_bytes <= 0:
        raise ValueError("streaming chunk size must be positive")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    identity = _plan_identity(source, chunk_bytes, max_length)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    directory = Path(plan_dir).resolve() if plan_dir else _default_plan_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.json"
    with FileLock(str(target) + ".lock"):
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            if cached.get("identity") == identity and cached.get("chunks"):
                return cached
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

        chunks = []
        chunk_start = 0
        rows = tokens = 0
        next_boundary = chunk_bytes
        scanned = 0
        source_hash = hashlib.sha256()
        started = time.perf_counter()
        with source.open("rb") as stream:
            while True:
                line_start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                source_hash.update(line)
                # Close before this complete row, never through a UTF-8/JSON
                # record.  Oversized individual rows form a chunk of their own.
                if rows and line_start >= next_boundary:
                    chunks.append({
                        "index": len(chunks), "start": chunk_start,
                        "end": line_start, "rows": rows, "tokens": tokens,
                    })
                    chunk_start = line_start
                    rows = tokens = 0
                    next_boundary = chunk_start + chunk_bytes
                rows += 1
                tokens += _row_tokens(line, tokenizer, max_length)
                scanned = stream.tell()
                if scanned // (1024 ** 3) != line_start // (1024 ** 3):
                    print(
                        f"[Streaming Plan] scanned={scanned / 1024 ** 3:.1f}GiB/"
                        f"{identity['size'] / 1024 ** 3:.1f}GiB, chunks={len(chunks) + 1}",
                        flush=True,
                    )
        if rows:
            chunks.append({
                "index": len(chunks), "start": chunk_start,
                "end": identity["size"], "rows": rows, "tokens": tokens,
            })
        if not chunks:
            raise ValueError(f"JSONL dataset is empty: {source}")

        plan = {
            "identity": identity,
            "chunks": chunks,
            "rows": sum(item["rows"] for item in chunks),
            "tokens": sum(item["tokens"] for item in chunks),
            "source_sha256": source_hash.hexdigest(),
            "scan_seconds": time.perf_counter() - started,
        }
        temp = target.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        return plan


def materialize_jsonl_range(source, target, start: int, end: int) -> int:
    """Copy one already-aligned byte range without loading it into memory."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > source.stat().st_size:
        raise ValueError(f"invalid JSONL byte range [{start}, {end})")
    target.parent.mkdir(parents=True, exist_ok=True)
    remaining = end - start
    with source.open("rb") as reader, target.open("wb") as writer:
        reader.seek(start)
        while remaining:
            block = reader.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise EOFError(f"source ended inside JSONL byte range [{start}, {end})")
            writer.write(block)
            remaining -= len(block)
    return end - start


def remove_plan_cache(plan_dir=None) -> None:
    """Test/helper API; training relies on ordinary cache-budget eviction."""
    directory = Path(plan_dir).resolve() if plan_dir else _default_plan_dir()
    if directory.exists() and not directory.is_symlink():
        shutil.rmtree(directory)
