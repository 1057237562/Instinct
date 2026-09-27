"""Bounded chunk plans for cache-constrained pretraining.

The source corpus stays untouched.  A plan records contiguous ranges and token
totals; only one range is materialized into the managed Arrow cache at a time.
Plans are tiny, deterministic, and safe to reuse after a resume.

JSONL ranges are newline-aligned byte offsets, so materializing one is a byte
copy.  Parquet ranges are row offsets aligned to row groups, so materializing
one re-encodes just those groups; both are sized by the Arrow footprint a chunk
will occupy rather than by the compressed size on disk.
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

from dataset import source_format


_TOKEN_COUNT = re.compile(rb'"token_count"\s*:\s*(\d+)')
_PLAN_VERSION = 4

# Ranges are byte offsets for JSONL and row offsets for parquet.
UNIT_BYTES = 'bytes'
UNIT_ROWS = 'rows'

# Footer keys written by ``dataset_compiler --align-chunk-bytes``.  The names
# are the contract with ``ALIGNED_CHUNK_*_KEY`` in dataset_compiler/src/compile.rs.
_ALIGNED_CHUNK_BYTES = 'instinct.aligned_chunk_bytes'
_ALIGNED_CHUNK_ROWS = 'instinct.aligned_chunk_rows'


def _plan_identity(path: Path, chunk_bytes: int, max_length: int, unit: str) -> dict:
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
        "format": source_format.classify(path) or "unknown",
        "unit": unit,
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


def _validate_plan_request(source: Path, chunk_bytes, max_length):
    if not source.is_file():
        raise FileNotFoundError(source)
    chunk_bytes = int(chunk_bytes)
    max_length = int(max_length)
    if chunk_bytes <= 0:
        raise ValueError("streaming chunk size must be positive")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    return chunk_bytes, max_length


def _cached_plan(source: Path, *, chunk_bytes, max_length, unit, plan_dir, build):
    """Return the plan for one source revision, rebuilding it when it changed."""
    identity = _plan_identity(source, chunk_bytes, max_length, unit)
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
        plan = build(identity)
        temp = target.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        return plan


def build_chunk_plan(
    data_path,
    *,
    chunk_bytes: int,
    max_length: int,
    tokenizer,
    plan_dir=None,
) -> dict:
    """Plan bounded chunks for a JSONL or parquet pretraining corpus."""
    source = Path(data_path).resolve()
    builder = (
        build_parquet_chunk_plan
        if source_format.is_parquet_source(source)
        else build_jsonl_chunk_plan
    )
    return builder(
        source,
        chunk_bytes=chunk_bytes,
        max_length=max_length,
        tokenizer=tokenizer,
        plan_dir=plan_dir,
    )


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
    chunk_bytes, max_length = _validate_plan_request(source, chunk_bytes, max_length)

    def build(identity):
        chunks = []
        chunk_start = 0
        rows = tokens = 0
        next_boundary = chunk_bytes
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
        return {
            "identity": identity,
            "chunks": chunks,
            "rows": sum(item["rows"] for item in chunks),
            "tokens": sum(item["tokens"] for item in chunks),
            "source_sha256": source_hash.hexdigest(),
            "scan_seconds": time.perf_counter() - started,
        }

    return _cached_plan(
        source, chunk_bytes=chunk_bytes, max_length=max_length,
        unit=UNIT_BYTES, plan_dir=plan_dir, build=build,
    )


def build_parquet_chunk_plan(
    data_path,
    *,
    chunk_bytes: int,
    max_length: int,
    tokenizer,
    plan_dir=None,
) -> dict:
    """Plan row-aligned chunks for a compiled parquet corpus.

    A file compiled with ``--align-chunk-bytes`` carries the chunk boundaries
    the JSONL planner would have produced, and those are used verbatim, so a
    training cursor keeps pointing at the same rows after switching containers.
    Without them, ``chunk_bytes`` is measured in the Arrow footprint a chunk
    will expand to and chunks are cut on row group boundaries — the smallest
    unit that can be materialized without re-encoding rows piecemeal.
    """
    source = Path(data_path).resolve()
    chunk_bytes, max_length = _validate_plan_request(source, chunk_bytes, max_length)

    def build(identity):
        import pyarrow.parquet as parquet

        started = time.perf_counter()
        handle = parquet.ParquetFile(source)
        metadata = handle.metadata
        group_rows = [
            int(metadata.row_group(index).num_rows)
            for index in range(metadata.num_row_groups)
        ]
        group_bytes = [
            int(metadata.row_group(index).total_byte_size)
            for index in range(metadata.num_row_groups)
        ]
        if not group_rows or not any(group_rows):
            raise ValueError(f"Parquet dataset is empty: {source}")

        aligned = _aligned_boundaries(handle, int(metadata.num_rows))
        if aligned is not None:
            aligned_bytes = _aligned_chunk_bytes(handle)
            identity["aligned_chunk_bytes"] = aligned_bytes
            if aligned_bytes != chunk_bytes:
                print(
                    f"[Streaming Plan] {source.name} is aligned to "
                    f"{aligned_bytes / 1024 ** 2:.0f}MiB source chunks; using those "
                    f"instead of the requested {chunk_bytes / 1024 ** 2:.0f}MiB "
                    "so the chunk plan matches the JSONL source",
                    flush=True,
                )
            boundaries = aligned
        else:
            boundaries = _footprint_boundaries(group_rows, group_bytes, chunk_bytes)

        tokens = _parquet_token_sums(
            handle, boundaries, tokenizer, max_length, identity,
        )
        starts = [0] + boundaries[:-1]
        chunks = [
            {
                "index": index, "start": start, "end": end,
                "rows": end - start, "tokens": tokens[index],
            }
            for index, (start, end) in enumerate(zip(starts, boundaries))
        ]
        return {
            "identity": identity,
            "chunks": chunks,
            "rows": sum(item["rows"] for item in chunks),
            "tokens": sum(item["tokens"] for item in chunks),
            "source_sha256": _file_sha256(source),
            "scan_seconds": time.perf_counter() - started,
        }

    return _cached_plan(
        source, chunk_bytes=chunk_bytes, max_length=max_length,
        unit=UNIT_ROWS, plan_dir=plan_dir, build=build,
    )


def _aligned_boundaries(handle, row_count):
    """Chunk boundaries recorded by ``--align-chunk-bytes``, or ``None``.

    Returns cumulative row counts, so entry ``i`` is the last row of chunk
    ``i``.  A malformed or stale list is ignored rather than trusted: the
    planner falls back to footprint-sized chunks.
    """
    raw = (handle.metadata.metadata or {}).get(_ALIGNED_CHUNK_ROWS.encode())
    if not raw:
        return None
    try:
        boundaries = [int(part) for part in raw.decode('utf-8').split(',') if part]
    except (UnicodeDecodeError, ValueError):
        print(
            f"[Streaming Plan] ignoring unreadable {_ALIGNED_CHUNK_ROWS} metadata",
            flush=True,
        )
        return None
    if not boundaries or boundaries[-1] != row_count:
        return None
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        return None
    return boundaries


def _aligned_chunk_bytes(handle):
    raw = (handle.metadata.metadata or {}).get(_ALIGNED_CHUNK_BYTES.encode())
    try:
        return int(raw.decode('utf-8'))
    except (AttributeError, UnicodeDecodeError, ValueError):
        return 0


def _footprint_boundaries(group_rows, group_bytes, chunk_bytes):
    """Cumulative row ends when chunks are cut by the Arrow-byte budget."""
    boundaries = []
    rows = footprint = 0
    total = 0
    for group_rows_count, group_bytes_count in zip(group_rows, group_bytes):
        # An oversized single row group still gets a chunk of its own.
        if rows and footprint + group_bytes_count > chunk_bytes:
            total += rows
            boundaries.append(total)
            rows = footprint = 0
        rows += group_rows_count
        footprint += group_bytes_count
    if rows:
        boundaries.append(total + rows)
    return boundaries


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet_token_sums(handle, boundaries, tokenizer, max_length, identity):
    """Token totals per chunk, capped exactly like the JSONL planner.

    ``boundaries`` are cumulative row counts, so the same routine serves both
    aligned chunks and row-group chunks.
    """
    names = set(handle.schema_arrow.names)
    if "token_count" in names:
        column = "token_count"
    elif "text" in names:
        column = "text"
    else:
        raise ValueError(
            "parquet pretraining corpora need a 'text' column (and ideally a "
            "'token_count' column); compile the file with the pretrain preset"
        )
    totals = [0] * len(boundaries)
    group = 0
    seen = 0
    announced = -1
    for batch in handle.iter_batches(columns=[column], batch_size=8192):
        values = batch.column(0).to_pylist()
        offset = 0
        while offset < len(values):
            while group < len(boundaries) and seen >= boundaries[group]:
                group += 1
            if group >= len(boundaries):
                break
            take = min(boundaries[group] - seen, len(values) - offset)
            segment = values[offset:offset + take]
            if column == "token_count":
                totals[group] += sum(
                    min(int(value or 0), max_length) for value in segment
                )
            else:
                encoded = tokenizer(
                    [str(value) for value in segment],
                    add_special_tokens=False,
                    max_length=max_length - 2,
                    truncation=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                totals[group] += sum(
                    min(len(ids) + 2, max_length) for ids in encoded.input_ids
                )
            seen += take
            offset += take
        progress = seen * 10 // max(boundaries[-1], 1)
        if progress != announced:
            announced = progress
            print(
                f"[Streaming Plan] tokenized={progress * 10}% of "
                f"{identity['size'] / 1024 ** 3:.1f}GiB parquet source",
                flush=True,
            )
    return totals


def materialize_range(source, target, start: int, end: int) -> int:
    """Copy one planned chunk out of a JSONL or parquet source."""
    if source_format.is_parquet_source(source):
        return materialize_parquet_range(source, target, start, end)
    return materialize_jsonl_range(source, target, start, end)


def materialize_parquet_range(source, target, start: int, end: int) -> int:
    """Write rows ``[start, end)`` of a parquet corpus as a standalone file.

    Planned chunks are row-group aligned, but the bounds are honored exactly so
    a plan written by another tool still materializes the rows it names.
    """
    import pyarrow.parquet as parquet

    source = Path(source).resolve()
    target = Path(target).resolve()
    start, end = int(start), int(end)
    handle = parquet.ParquetFile(source)
    metadata = handle.metadata
    total_rows = int(metadata.num_rows)
    if start < 0 or end <= start or end > total_rows:
        raise ValueError(f"invalid parquet row range [{start}, {end}) of {total_rows}")
    group_rows = [
        int(metadata.row_group(index).num_rows)
        for index in range(metadata.num_row_groups)
    ]
    groups = []
    group_start = 0
    first_start = None
    for index, count in enumerate(group_rows):
        if group_start < end and group_start + count > start:
            if first_start is None:
                first_start = group_start
            groups.append(index)
        group_start += count
    if not groups:
        raise ValueError(f"no parquet row groups cover [{start}, {end})")

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = 0
    cursor = int(first_start)
    try:
        # ``pyarrow.ParquetWriter.write_batch`` closes one row group per call, so
        # the copy re-encodes the range in small batches.  An aligned chunk maps
        # onto a single source row group, which is what bounds the decode work;
        # the batch size only trades footer size against peak memory, and stays
        # small so a long-row corpus cannot inflate it.
        for batch in handle.iter_batches(row_groups=groups, batch_size=4096):
            batch_start = cursor
            cursor += batch.num_rows
            low = max(start, batch_start)
            high = min(end, cursor)
            if high <= low:
                continue
            if writer is None:
                writer = parquet.ParquetWriter(
                    target, handle.schema_arrow, compression='zstd',
                )
            writer.write_batch(batch.slice(low - batch_start, high - low))
            written += high - low
    finally:
        if writer is not None:
            writer.close()
    if written != end - start:
        raise EOFError(
            f"parquet chunk [{start}, {end}) materialized {written} rows"
        )
    return written


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
