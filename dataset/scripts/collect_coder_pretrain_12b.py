"""Collect resumable supplement shards for the 12B Instinct Coder corpus.

The final DeepSeek-Coder-V2-style target is assembled elsewhere.  This script
only collects exact-novel headroom in four auditable components.  Each upstream
shard is written atomically with its own report, so interruption never loses a
completed shard and rerunning resumes naturally.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
from urllib.parse import quote

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401  # before transformers on Windows
from huggingface_hub import HfApi
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from dataset.scripts.collect_arxiv_pretrain import USER_AGENT


OUT = ROOT / "dataset/coder_pretrain_12b"
TOKENIZER = ROOT / "model"
MAX_TOKENS = 4096
TARGETS = {
    "code": 6_100_000_000,
    "math": 1_100_000_000,
    "natural_en": 2_100_000_000,
    "natural_zh": 1_100_000_000,
}
SOURCES = {
    "code": {
        "repo": "common-pile/stackv2_edu_filtered",
        "revision": "c354dbe88469a1153e97c6a63ac50591849654de",
        "prefix": "stack-edu-",
        "suffix": ".json.gz",
        "license": "per-file open license; consult metadata",
    },
    "math": {
        "repo": "HuggingFaceTB/finemath",
        "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "prefix": "finemath-4plus/",
        "suffix": ".parquet",
        "license": "ODC-By-1.0",
    },
    "natural_en": {
        "repo": "HuggingFaceTB/smollm-corpus",
        "revision": "3ba9d605774198c5868892d7a8deda78031a781f",
        "prefix": "fineweb-edu-dedup/",
        "suffix": ".parquet",
        "license": "ODC-By-1.0",
    },
    "natural_zh": {
        "repo": "opencsg/chinese-fineweb-edu",
        "revision": "784f118864c0741ce32aa74d9588795bdf1a896b",
        "prefix": "",
        "suffix": ".parquet",
        "license": "Apache-2.0; review upstream source terms",
    },
}
NON_CODE_LANGUAGES = {
    "Markdown", "TeX", "reStructuredText", "RMarkdown", "JSON", "YAML",
    "Text", "CSV", "Jupyter Notebook",
}
BAD_MARKERS = (
    "\ufffd", "write a paper online", "online casino", "click here to buy",
    "<|im_start|>", "<|im_end|>", "[assistant/analysis]",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component", choices=["all", *TARGETS], default="all",
        help="Collect one component or all components in code/math/en/zh order.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Per-component debug limit; 0 means no limit.")
    return parser.parse_args()


def clean_text(text, minimum=200, maximum=200_000):
    if not isinstance(text, str) or not minimum <= len(text) <= maximum:
        return False
    lowered = text.lower()
    if "\x00" in text or any(marker.lower() in lowered for marker in BAD_MARKERS):
        return False
    if any(ord(char) < 32 and char not in "\r\n\t" for char in text):
        return False
    normalized_lines = [" ".join(line.lower().split()) for line in text.splitlines()]
    substantial = [line for line in normalized_lines if len(line) >= 80]
    if substantial and max(Counter(substantial).values()) >= 4:
        return False
    return True


def stable_order(files):
    """Interleave language/time-grouped shards deterministically."""
    if not files:
        return []
    step = 37
    while len(files) % step == 0:
        step += 2
    ordered, used = [], set()
    index = 0
    for _ in files:
        while index in used:
            index = (index + 1) % len(files)
        ordered.append(files[index])
        used.add(index)
        index = (index + step) % len(files)
    return ordered


def list_source_files(component):
    source = SOURCES[component]
    if component == "code":
        return stable_order([f"stack-edu-{index:04d}.json.gz" for index in range(95)])
    info = HfApi().dataset_info(source["repo"], revision=source["revision"])
    files = [item.rfilename for item in info.siblings
             if item.rfilename.startswith(source["prefix"])
             and item.rfilename.endswith(source["suffix"])]
    return stable_order(sorted(files))


def completed_state(component):
    reports_dir = OUT / component / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    completed, tokens = set(), 0
    for path in reports_dir.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        completed.add(report["upstream_file"])
        tokens += int(report["tokens"])
    return completed, tokens


def record_for(component, row, text, token_count, shard_name, row_index):
    source = SOURCES[component]
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if component == "code":
        source_id = row.get("id") or f"{shard_name}:{row_index}"
        language = metadata.get("language")
        license_name = metadata.get("license") or source["license"]
        url = metadata.get("url") or ""
    elif component == "natural_en":
        source_id = row.get("id") or f"{shard_name}:{row_index}"
        language = metadata.get("language") or "en"
        license_name = source["license"]
        url = metadata.get("url") or ""
    else:
        source_id = row.get("id") or row.get("url") or f"{shard_name}:{row_index}"
        language = row.get("language") or ("zh" if component == "natural_zh" else "en")
        license_name = source["license"]
        url = row.get("url") or ""
    return {
        "text": text,
        "source": f"{source['repo']}:{component}",
        "source_id": str(source_id),
        "license": str(license_name),
        "url": str(url),
        "token_count": token_count,
        "language": str(language or ""),
        "upstream_shard": shard_name,
    }


def accepted_text(component, row):
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    text = row.get("text") or row.get("content")
    if not clean_text(text):
        return None, "text_quality"
    if component == "code":
        language = str(metadata.get("language") or "")
        if not language or language in NON_CODE_LANGUAGES:
            return None, "non_code_language"
        if int(row.get("int_score") or 0) < 3:
            return None, "low_score"
        if metadata.get("is_vendor") or metadata.get("is_generated"):
            return None, "vendor_or_generated"
    elif component == "math":
        if int(row.get("int_score") or 0) < 4:
            return None, "low_score"
        if str(row.get("language") or "en") != "en":
            return None, "wrong_language"
    elif component == "natural_en":
        score = metadata.get("int_score")
        if score is not None and int(score) < 4:
            return None, "low_score"
        if str(metadata.get("language") or "en") != "en":
            return None, "wrong_language"
    return text.strip(), None


def process_rows(component, rows, shard_name, destination, tokenizer, remaining):
    digest = hashlib.sha256()
    stats = Counter()
    seen_ids = set()
    tokens = rows_written = bytes_written = 0
    for row_index, row in enumerate(rows):
        stats["scanned"] += 1
        text, rejection = accepted_text(component, row)
        if rejection:
            stats[f"rejected_{rejection}"] += 1
            continue
        source_id = str(row.get("id") or row.get("url") or f"{shard_name}:{row_index}")
        if source_id in seen_ids:
            stats["rejected_duplicate_id"] += 1
            continue
        token_count = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
        if not 64 <= token_count <= MAX_TOKENS:
            stats["rejected_token_length"] += 1
            continue
        if tokens + token_count > remaining:
            stats["rejected_over_budget"] += 1
            if remaining - tokens < MAX_TOKENS:
                break
            continue
        record = record_for(component, row, text, token_count, shard_name, row_index)
        line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        destination.write(line)
        digest.update(line)
        seen_ids.add(source_id)
        tokens += token_count
        rows_written += 1
        bytes_written += len(line)
    return {
        "tokens": tokens,
        "rows": rows_written,
        "bytes": bytes_written,
        "sha256": digest.hexdigest(),
        "selection": dict(stats),
    }


def parquet_rows(path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=512):
        yield from batch.to_pylist()


def direct_dataset_download(repo, revision, upstream_file, raw_dir, timeout):
    """Download a Hub file with an atomic, HTTP-Range-resumable partial file."""
    destination = raw_dir / upstream_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
        f"{quote(upstream_file, safe='/')}?download=true"
    )
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        append = existing > 0 and status == 206
        if existing and not append:
            existing = 0
        content_length = response.headers.get("Content-Length")
        expected = existing + int(content_length) if content_length else None
        mode = "ab" if append else "wb"
        downloaded = existing
        next_report = ((downloaded // (256 * 1024 * 1024)) + 1) * 256 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    print(
                        f"[download] {upstream_file}: {downloaded / 1e9:.2f} GB",
                        flush=True,
                    )
                    next_report += 256 * 1024 * 1024
            output.flush()
            os.fsync(output.fileno())
    if expected is not None and partial.stat().st_size != expected:
        raise EOFError(
            f"short download for {upstream_file}: {partial.stat().st_size}/{expected} bytes"
        )
    os.replace(partial, destination)
    return destination


def collect_shard(component, upstream_file, tokenizer, remaining, timeout):
    source = SOURCES[component]
    component_dir = OUT / component
    shards_dir = component_dir / "shards"
    reports_dir = component_dir / "reports"
    raw_dir = component_dir / "raw"
    for directory in (shards_dir, reports_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = upstream_file.replace("/", "__").replace(".json.gz", "").replace(".parquet", "")
    output = shards_dir / f"{stem}.jsonl"
    report_path = reports_dir / f"{stem}.json"
    temporary = output.with_suffix(".jsonl.tmp")
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if temporary.exists():
        temporary.unlink()

    compressed_bytes = None
    try:
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination:
            if component == "code":
                local_path = direct_dataset_download(
                    source["repo"], source["revision"], upstream_file,
                    raw_dir, timeout,
                )
                compressed_bytes = local_path.stat().st_size
                with gzip.open(local_path, mode="rb") as archive:
                    rows = (orjson.loads(line) for line in archive)
                    result = process_rows(
                        component, rows, upstream_file, destination, tokenizer, remaining
                    )
            else:
                local_path = direct_dataset_download(
                    source["repo"], source["revision"], upstream_file,
                    raw_dir, timeout,
                )
                compressed_bytes = local_path.stat().st_size
                result = process_rows(
                    component, parquet_rows(local_path), upstream_file,
                    destination, tokenizer, remaining,
                )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    report = {
        "component": component,
        "upstream_repo": source["repo"],
        "upstream_revision": source["revision"],
        "upstream_file": upstream_file,
        "downloaded_bytes": compressed_bytes,
        "output": str(output),
        **result,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def collect_component(component, args, tokenizer):
    completed, total_tokens = completed_state(component)
    files = list_source_files(component)
    files = files[args.worker_index::args.shard_workers]
    processed_now = 0
    print(f"[{component}] worker={args.worker_index}/{args.shard_workers} "
          f"resume={total_tokens:,}/{TARGETS[component]:,} tokens, "
          f"completed_shards={len(completed)}", flush=True)
    for upstream_file in files:
        if total_tokens >= TARGETS[component] - MAX_TOKENS:
            break
        if upstream_file in completed:
            continue
        report = None
        for attempt in range(1, args.retries + 1):
            try:
                report = collect_shard(
                    component, upstream_file, tokenizer,
                    TARGETS[component] - total_tokens, args.timeout,
                )
                break
            except (EOFError, OSError, ConnectionError, TimeoutError) as error:
                if attempt == args.retries:
                    raise
                delay = min(2 ** attempt, 30)
                print(
                    f"[{component}] retry {attempt}/{args.retries - 1} for "
                    f"{upstream_file} after {type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(delay)
        assert report is not None
        # Re-read global progress so disjoint workers stop against the same
        # component target. A small final-shard overshoot is intentional
        # headroom for global deduplication during final assembly.
        _, total_tokens = completed_state(component)
        processed_now += 1
        print(f"[{component}] {upstream_file}: +{report['tokens']:,} => "
              f"{total_tokens:,}/{TARGETS[component]:,}", flush=True)
        if args.max_shards and processed_now >= args.max_shards:
            break
    if (args.shard_workers == 1 and not args.max_shards
            and total_tokens < TARGETS[component] - MAX_TOKENS):
        raise RuntimeError(
            f"{component} exhausted upstream shards at {total_tokens:,}/"
            f"{TARGETS[component]:,} tokens"
        )
    return total_tokens


def main():
    args = parse_args()
    if (args.timeout < 1 or args.max_shards < 0 or args.retries < 1
            or args.shard_workers < 1
            or not 0 <= args.worker_index < args.shard_workers):
        raise ValueError(
            "timeout/retries/shard-workers must be positive, max-shards >= 0, "
            "and worker-index must be within shard-workers"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True).backend_tokenizer
    components = list(TARGETS) if args.component == "all" else [args.component]
    summary = {}
    for component in components:
        summary[component] = collect_component(component, args, tokenizer)
    print(json.dumps({"targets": TARGETS, "collected": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
