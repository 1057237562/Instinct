"""Parallel-range downloader for single large Hub files (mirror-friendly).

The hf-mirror endpoint throttles per connection; fetching one file with N
ranged connections multiplies throughput. Used to pre-place upstream shards in
the collectors' raw directories so both collect stages read locally.

Usage:
  python fetch_parallel.py URL OUT [CONNECTIONS]
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

CHUNK = 64 * 1024 * 1024


def head_size(url: str, timeout: int = 60) -> int | None:
    request = Request(url, headers={"User-Agent": "Instinct-fetch/1.0", "Range": "bytes=0-0"})
    with urlopen(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return response.headers.get("Content-Length") and int(response.headers["Content-Length"]) or None


def fetch_range(url: str, start: int, end: int, destination: Path, attempts: int = 5) -> int:
    for attempt in range(1, attempts + 1):
        written = 0
        try:
            request = Request(url, headers={
                "User-Agent": "Instinct-fetch/1.0",
                "Range": f"bytes={start + written}-{end}",
                "Accept-Encoding": "identity",
            })
            with urlopen(request, timeout=120) as response, destination.open("r+b") as handle:
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    raise OSError(f"range request not honored: HTTP {status}")
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise EOFError(f"range {start}-{end}: stream ended early")
                    handle.write(block)
                    written += len(block)
                    remaining -= len(block)
            return written
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** attempt, 20))
            # retry resumes inside the range from the persisted offset
            start = start + written
    raise RuntimeError("unreachable")


def fetch_range_into(url: str, start: int, end: int, part: Path) -> int:
    """Stream one byte range directly into its offset in ``part``."""
    for attempt in range(5):
        written = 0
        try:
            request = Request(url, headers={
                "User-Agent": "Instinct-fetch/1.0",
                "Range": f"bytes={start + written}-{end}",
                "Accept-Encoding": "identity",
            })
            with urlopen(request, timeout=120) as response, part.open("r+b") as handle:
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    raise OSError(f"range request not honored: HTTP {status}")
                handle.seek(start + written)
                remaining = end - start + 1 - written
                while remaining > 0:
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise EOFError(f"range {start}-{end}: stream ended early")
                    handle.write(block)
                    written += len(block)
                    remaining -= len(block)
            return written
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("unreachable")


def download(url: str, out: Path, connections: int = 8) -> None:
    """Parallel-range download; peak disk = file size (no piece copies).

    Ranges stream straight into a preallocated .ppart at their offsets; a tiny
    .pjournal records completed range indexes so reruns resume exactly.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".ppart")
    journal = out.with_suffix(out.suffix + ".pjournal")
    size = head_size(url)
    if size is None:
        raise RuntimeError(f"no size for {url}")
    state = None
    if journal.exists():
        try:
            state = json.loads(journal.read_text())
        except Exception:
            state = None
    if not state or state.get("size") != size or not part.exists() or part.stat().st_size != size:
        with part.open("wb") as handle:
            handle.truncate(size)
        state = {"size": size, "chunk": CHUNK, "done": []}
    chunk = state.get("chunk") or CHUNK
    bounds = [(offset, min(offset + CHUNK, size) - 1) for offset in range(0, size, CHUNK)]
    done = set(state.get("done") or [])
    pending = [index for index, _ in enumerate(bounds) if index not in done]
    started = time.time()
    done_bytes = sum(bounds[index][1] - bounds[index][0] + 1 for index in done)

    def work(index):
        start, end = bounds[index]
        fetch_range_into(url, start, end, part)
        return index

    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        for index in pool.map(work, pending):
            start, end = bounds[index]
            done.add(index)
            done_bytes += end - start + 1
            rate = done_bytes / max(time.time() - started, 1) / 1e6
            print(f"{out.name}: {done_bytes / 1e9:.2f}/{size / 1e9:.2f} GB ({rate:.1f} MB/s)", flush=True)
            state["done"] = sorted(done)
            journal.write_text(json.dumps(state))
    if len(done) != len(bounds):
        raise RuntimeError(f"incomplete: {len(done)}/{len(bounds)} ranges")
    os.replace(part, out)
    journal.unlink(missing_ok=True)
    print(f"{out.name}: DONE {size:,} bytes", flush=True)


if __name__ == "__main__":
    download(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
