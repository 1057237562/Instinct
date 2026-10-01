"""Shared bounded-concurrency helpers for dataset generation scripts."""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os


def default_workers():
    """Return a conservative I/O/validation worker count, overridable by env."""
    configured = os.environ.get("INSTINCT_DATA_WORKERS")
    if configured:
        return max(1, int(configured))
    return min(32, max(4, (os.cpu_count() or 1) + 4))


def batched(iterable, size):
    """Yield bounded lists without materializing a large input stream."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, max(1, int(size)))):
        yield batch


def ordered_thread_map(function, iterable, *, workers=None, batch_size=2048):
    """Map in threads with deterministic order and bounded pending futures."""
    worker_count = default_workers() if workers is None else max(1, int(workers))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dataset") as executor:
        for batch in batched(iterable, batch_size):
            yield from executor.map(function, batch)
