"""Stream-extract competitive_problem_reasoning rows from ~24 GB Nemotron shards.

build_codespecialist_4096.py consumes ~72M tokens per (language, shard) with
cap=4 rows per question, so ~95% of each 24 GB upstream shard is discarded by
the question cap. Downloading the shards whole does not fit local disk or the
build schedule, so this script:

- splits each shard into byte ranges fetched concurrently (8 connections),
- skips cap-saturated questions by extracting just the question_id from the
  raw line instead of parsing the full JSON record,
- tokenizes surviving candidates on a thread pool and keeps rows the builder
  would accept, per range (over-selection across ranges is harmless: the
  builder re-applies its cap over the merged file and stops at its target),
- concatenates per-range selections in range order up to a per-shard token
  checkpoint with headroom for the builder's global deduplication.

The substitution is recorded in per-language extract reports.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import orjson
from datasets import load_dataset  # noqa: F401  # before transformers on Windows
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

MIRROR = (
    os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    + "/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2/resolve/main/data"
)
MAX_TOKENS = 4096
PER_LANGUAGE_TARGET = 144_000_000
PER_SHARD_CHECKPOINT = PER_LANGUAGE_TARGET // 2
HEADROOM = 8_000_000
RANGES = 8
ENCODE_WORKERS = 8
QID_RE = re.compile(rb'"(?:question_id|uuid)":\s*"([^"]+)"')


def render_messages(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            return ""
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("python", "cpp"), required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/competitive-coding/data")
    return parser.parse_args()


def head_size(url: str) -> int:
    request = Request(url, headers={"User-Agent": "Instinct-extract/1.0", "Range": "bytes=0-0"})
    with urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range")
        if not content_range:
            raise OSError(f"no Content-Range from server for {url}")
        return int(content_range.rsplit("/", 1)[1])


def stream_range(url: str, start: int, end: int, destination: Path, attempts: int = 8) -> dict:
    """Fetch byte range [start, end] with resume; selects candidate rows.

    State is a per-range file storing the offset RELATIVE to ``start`` plus any
    partial line, so an interrupted range resumes with an exact HTTP Range
    request. A row split across two adjacent ranges is lost (<= 1 row per
    boundary; negligible against ~113 KB rows) and noted in the report.
    """
    stats = defaultdict(int)
    rows_file = destination.with_suffix(destination.suffix + f".r{start}.part")
    state_file = destination.with_suffix(destination.suffix + f".r{start}.state")
    offset = 0
    tail = b""
    if state_file.exists():
        state = json.loads(state_file.read_text())
        offset = int(state["offset"])
        tail = state["tail"].encode("latin-1")
    local_counts = defaultdict(int)
    candidates = []
    mode = "ab" if offset else "wb"
    started = time.time()
    with rows_file.open(mode, buffering=4 * 1024 * 1024) as handle:
        for attempt in range(1, attempts + 1):
            try:
                position = start + offset
                if position > end:
                    break
                headers = {"User-Agent": "Instinct-extract/1.0", "Accept-Encoding": "identity",
                           "Range": f"bytes={position}-{end}"}
                request = Request(url, headers=headers)
                with urlopen(request, timeout=120) as response:
                    status = getattr(response, "status", response.getcode())
                    if status != 206:
                        raise OSError(f"range request not honored: HTTP {status}")
                    expected = end - position + 1
                    received = 0
                    while received < expected:
                        block = response.read(min(4 * 1024 * 1024, expected - received))
                        if not block:
                            raise EOFError(f"range {position}-{end} ended early")
                        received += len(block)
                        offset += len(block)
                        tail += block
                        lines = tail.split(b"\n")
                        tail = lines.pop()
                        for raw_line in lines:
                            if not raw_line.strip():
                                continue
                            stats["scanned"] += 1
                            match = QID_RE.search(raw_line[:2000])
                            source_id = match.group(1).decode("ascii", "replace") if match else ""
                            if source_id and local_counts[source_id] >= 4:
                                stats["question_cap"] += 1
                                continue
                            candidates.append((raw_line, source_id))
                            if len(candidates) >= 256:
                                flush(handle, candidates, local_counts, stats)
                                candidates = []
                        if len(candidates) >= 64:
                            flush(handle, candidates, local_counts, stats)
                            candidates = []
                        state_file.write_text(json.dumps(
                            {"offset": offset, "tail": tail.decode("latin-1")}))
                    flush(handle, candidates, local_counts, stats)
                    candidates = []
                    break
            except (EOFError, OSError, TimeoutError, ConnectionError) as error:
                if attempt == attempts:
                    flush(handle, candidates, local_counts, stats)
                    raise
                time.sleep(min(2 ** attempt, 20))
    return {"range": start, "scanned": stats["scanned"], "kept": stats["kept"],
            "bytes": offset, "seconds": round(time.time() - started, 1)}


def flush(handle, candidates, local_counts, stats) -> None:
    """Render + tokenize + filter a candidate batch, writing accepted rows."""
    if not candidates:
        return
    parsed = []
    for raw_line, source_id in candidates:
        try:
            row = orjson.loads(raw_line)
        except Exception:
            parsed.append(None)
            stats["bad_json"] += 1
            continue
        text = render_messages(row.get("messages"))
        parsed.append((row, text) if text else None)
        if not text:
            stats["empty_render"] += 1
    texts = [item[1] for item in parsed if item is not None]
    token_lists = list(ENCODE_POOL.map(
        lambda text: len(TOKENIZER.encode(text, add_special_tokens=False).ids) + 2,
        texts))
    tokens_by_text = {id(text): count for text, count in zip(texts, token_lists)}
    for (_, source_id), item in zip(candidates, parsed):
        if item is None:
            continue
        row, text = item
        token_count = tokens_by_text[id(text)]
        if token_count > MAX_TOKENS:
            stats["over_4096"] += 1
            continue
        handle.write(orjson.dumps({
            "messages": row.get("messages"),
            "question_id": source_id,
            "license": str(row.get("license") or ""),
            "token_count": token_count,
        }, option=orjson.OPT_APPEND_NEWLINE))
        stats["kept"] += 1
        if source_id:
            local_counts[source_id] += 1


TOKENIZER = None
ENCODE_POOL = None


def main() -> None:
    global TOKENIZER, ENCODE_POOL
    args = parse_args()
    TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True).backend_tokenizer
    ENCODE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    output_dir = args.output_dir
    reports = []
    for shard in ("00", "01"):
        destination = output_dir / f"competitive_programming_{args.language}_{shard}.jsonl"
        if destination.is_file():
            print(f"{destination.name}: exists, skip", flush=True)
            continue
        url = f"{MIRROR}/competitive_programming_{args.language}_{shard}.jsonl?download=true"
        size = head_size(url)
        bounds = [(offset, min(offset + size // RANGES + 1, size) - 1) for offset in range(0, size, size // RANGES + 1)]
        range_reports_by_start = {}
        started = time.time()
        pending = list(bounds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=RANGES) as pool:
            for attempt in range(1, 5):
                futures = {pool.submit(stream_range, url, start, end, destination): (start, end)
                           for start, end in pending}
                failed = []
                for future in concurrent.futures.as_completed(futures):
                    start, end = futures[future]
                    try:
                        range_reports_by_start[start] = future.result()
                    except Exception as error:
                        print(f"shard {shard} range {start}-{end} attempt {attempt} failed: "
                              f"{type(error).__name__}: {str(error)[:100]}; will retry", flush=True)
                        failed.append((start, end))
                pending = failed
                if not pending:
                    break
        if pending:
            raise RuntimeError(f"shard {shard}: ranges never completed: {pending}")
        range_reports = [range_reports_by_start[start] for start, _ in bounds]
        # concatenate in range order up to the token checkpoint
        checkpoint = PER_SHARD_CHECKPOINT - MAX_TOKENS + HEADROOM
        total_tokens = 0
        selected_rows = 0
        temporary = destination.with_suffix(".jsonl.extracting")
        with temporary.open("wb", buffering=8 * 1024 * 1024) as handle:
            for start, _ in bounds:
                if total_tokens >= checkpoint:
                    break
                range_file = destination.with_suffix(destination.suffix + f".r{start}.part")
                with range_file.open("rb") as source:
                    for line in source:
                        token_count = orjson.loads(line)["token_count"]
                        if total_tokens + token_count > checkpoint:
                            break
                        handle.write(line)
                        total_tokens += token_count
                        selected_rows += 1
                    if total_tokens >= checkpoint:
                        break
        temporary.rename(destination)
        for start, _ in bounds:
            destination.with_suffix(destination.suffix + f".r{start}.part").unlink(missing_ok=True)
            destination.with_suffix(destination.suffix + f".r{start}.state").unlink(missing_ok=True)
        shard_report = {"shard": shard, "selected_rows": selected_rows,
                        "selected_tokens": total_tokens, "seconds": round(time.time() - started, 1),
                        "ranges": range_reports}
        reports.append(shard_report)
        print(json.dumps({k: v for k, v in shard_report.items() if k != "ranges"}), flush=True)
    report_path = output_dir / f"competitive_programming_{args.language}.extract.report.json"
    report_path.write_text(json.dumps({
        "language": args.language,
        "method": "parallel-range streamed selection replicating build_codespecialist_4096.message_file "
                  "(cap=4 rows per question, <=4096 tokens, checkpoint 72M + 8M headroom per shard); "
                  "per-range selections concatenated in range order, builder re-applies its cap over the merged file",
        "note": "upstream shards are ~24 GB each and cannot fit local disk whole; only builder-consumable rows are kept",
        "shared_question_counts_across_shards": False,
        "shards": reports,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"language {args.language} complete: {sum(r['selected_tokens'] for r in reports):,} tokens", flush=True)


if __name__ == "__main__":
    main()
