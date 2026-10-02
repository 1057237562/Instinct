"""Resolve JSONL/Parquet dataset sources for the trainers.

A dataset argument is a ``.jsonl`` file, a ``.parquet`` file, or a directory
holding either.  Both formats are handed to ``datasets`` so that everything
downstream — source fingerprints, Arrow caches, feature casting, packing — is
unchanged; only the builder name and the file list differ.

Parquet is the compiled form of the same corpora (see ``dataset_compiler/``):
it is roughly 3x smaller than JSONL and needs no Python JSON parsing, at the
cost of a build step when the source changes.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

PARQUET_SUFFIXES = ('.parquet', '.pq')
# ``.json`` is deliberately absent: report sidecars are ``*.report.json`` and a
# directory scan must never mistake one for a corpus.
JSON_SUFFIXES = ('.jsonl', '.ndjson', '.jsonl.gz', '.json.gz')

PARQUET_FORMAT = 'parquet'
JSON_FORMAT = 'json'


class DatasetSource:
    """Where one dataset comes from and how ``datasets`` should read it."""

    __slots__ = ('path', 'format', 'files')

    def __init__(self, path, format, files):
        self.path = str(path)
        self.format = format
        self.files = list(files)

    @property
    def is_parquet(self):
        return self.format == PARQUET_FORMAT

    @property
    def builder(self):
        return self.format

    @property
    def data_files(self):
        """A single path stays a string so ``datasets`` names the split once."""
        return self.files[0] if len(self.files) == 1 else list(self.files)

    def __repr__(self):
        return (
            f"DatasetSource(path={self.path!r}, format={self.format!r}, "
            f"files={len(self.files)})"
        )


def _suffix_of(name):
    lowered = str(name).lower()
    for suffix in ('.jsonl.gz', '.json.gz', '.parquet', '.pq', '.jsonl', '.ndjson'):
        if lowered.endswith(suffix):
            return suffix
    return None


def classify(path):
    """Return ``'parquet'``, ``'json'`` or ``None`` for one path."""
    suffix = _suffix_of(os.fspath(path))
    if suffix in PARQUET_SUFFIXES:
        return PARQUET_FORMAT
    if suffix in JSON_SUFFIXES:
        return JSON_FORMAT
    return None


def is_parquet_source(path):
    return classify(path) == PARQUET_FORMAT


def canonical_source_key(path):
    """Identity of a corpus that ignores which container holds it.

    ``x.jsonl`` and ``x.parquet`` hold the same rows in the same order once the
    corpus has been compiled, so a training resume must not treat a container
    switch as a different dataset.  Two files in different directories, or with
    different stems, still count as different corpora.
    """
    text = os.path.normcase(os.path.abspath(os.fspath(path)))
    suffix = _suffix_of(text)
    return text[: -len(suffix)] if suffix else text


def source_files(path):
    """Resolve a file or directory into the ordered list of data files.

    Directories are expanded here rather than left to the ``datasets`` glob so
    that shard order is deterministic and a directory holding both formats
    resolves to one of them instead of mixing.
    """
    path = Path(path)
    if path.is_file():
        if classify(path) is None:
            raise ValueError(
                f"unsupported dataset file {path}; expected .jsonl or .parquet"
            )
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    parquet = sorted(
        str(item) for item in path.rglob('*')
        if item.is_file() and classify(item) == PARQUET_FORMAT
    )
    if parquet:
        return parquet
    jsonl = sorted(
        str(item) for item in path.rglob('*')
        if item.is_file() and classify(item) == JSON_FORMAT
    )
    if jsonl:
        return jsonl
    raise FileNotFoundError(f"no .jsonl or .parquet files under {path}")


def resolve_source(path):
    """Return a :class:`DatasetSource` for ``datasets.load_dataset``."""
    files = source_files(path)
    format = classify(files[0])
    for item in files:
        if classify(item) != format:
            raise ValueError(
                f"dataset directory mixes formats: {files[0]} and {item}"
            )
    return DatasetSource(path, format, files)


def source_bytes(path):
    """Total size of the source files on disk."""
    return sum(os.path.getsize(item) for item in source_files(path))


def estimated_arrow_bytes(path):
    """Expected Arrow cache size, used to reserve dataset-cache budget.

    A JSONL source expands to roughly its own size plus parsing overhead.  A
    parquet file is compressed, so the footer's per-column uncompressed totals
    are a far better estimate than the file size; the fallback keeps the
    historical 1.25x when the footer cannot be read.
    """
    files = source_files(path)
    total = 0
    for item in files:
        size = os.path.getsize(item)
        if classify(item) != PARQUET_FORMAT:
            total += int(size * 1.25)
            continue
        uncompressed = _parquet_uncompressed_bytes(item)
        total += int(uncompressed * 1.05) if uncompressed else int(size * 1.25)
    return total


def _parquet_uncompressed_bytes(path):
    try:
        import pyarrow.parquet as parquet
        metadata = parquet.ParquetFile(path).metadata
        return int(metadata.serialized_size + sum(
            metadata.row_group(index).total_byte_size
            for index in range(metadata.num_row_groups)
        ))
    except Exception:
        return 0


def iter_source_rows(path):
    """Yield rows as plain dicts, streaming, for either format.

    Used where a dataset object would be handed straight to Python (the agent
    RL loader keeps rows in memory) so the reader never materializes the whole
    file at once.
    """
    source = resolve_source(path)
    if source.is_parquet:
        yield from _iter_parquet_rows(source.files)
    else:
        yield from _iter_json_rows(source.files)


def _iter_parquet_rows(files):
    import pyarrow.parquet as parquet

    for item in files:
        handle = parquet.ParquetFile(item)
        for batch in handle.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield row


def _iter_json_rows(files):
    for item in files:
        opener = gzip.open if str(item).lower().endswith('.gz') else open
        with opener(item, 'rt', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
