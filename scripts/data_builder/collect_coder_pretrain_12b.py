"""Collect resumable supplement shards for the 12B Instinct Coder corpus.

The final DeepSeek-Coder-V2-style target is assembled elsewhere.  This script
only collects exact-novel headroom in four auditable components.  Each upstream
shard is written atomically with its own report, so interruption never loses a
completed shard and rerunning resumes naturally.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen
from urllib.parse import quote

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401  # before transformers on Windows
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_builder.collect_arxiv_pretrain import USER_AGENT


OUT = ROOT / "dataset/coder_pretrain_12b"
ACTIVE_OUT = OUT  # overridable via --out-dir so a follow-up 12B batch can
                  # collect the NEXT upstream shards without colliding with
                  # the first batch's reports
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
EXCLUSION_INDEX = None
IDENTITY_REVIEW = None


class CollectionPaused(Exception):
    """Pause at a safe boundary while keeping resumable raw downloads."""


def check_pause():
    if (ACTIVE_OUT / ".pause_request").exists():
        raise CollectionPaused("Pause requested; completed shards and raw download partials retained")


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
    parser.add_argument("--segment", type=int, default=1,
                        help=(
                            "Which consecutive batch of upstream shards to collect "
                            "(1 = the default batch from the start; N>1 = skip every "
                            "shard consumed by segments 1..N-1 and continue after them). "
                            "Segment N>1 defaults its output root to "
                            "dataset/coder_pretrain_12b_seg{N}."
                        ))
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output root override (default is segment-derived).")
    parser.add_argument("--skip-reports-from", action="append", default=None,
                        help=(
                            "Additional completed run directory whose per-shard "
                            "reports mark upstream shards to skip (repeatable)."
                        ))
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="Override the collection quota for one --component.")
    parser.add_argument("--novel-index", type=Path, default=None,
                        help="Read-only prior-corpus fingerprint index; reject matching texts.")
    parser.add_argument("--identity-review", action="store_true",
                        help="Emit whole-row identity review candidates; shards remain raw, unapproved data.")
    return parser.parse_args()


def segment_out_root(segment):
    """Segment 1 keeps the historical path; later segments get suffixed dirs."""
    return OUT if segment == 1 else ROOT / f"dataset/coder_pretrain_12b_seg{segment}"


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


def list_source_files(component, timeout=90, retries=5):
    check_pause()
    source = SOURCES[component]
    if component == "code":
        return stable_order([f"stack-edu-{index:04d}.json.gz" for index in range(95)])
    cache = ACTIVE_OUT / component / "upstream_files.json"
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if (saved.get("upstream_repo") == source["repo"]
                    and saved.get("upstream_revision") == source["revision"]
                    and isinstance(saved.get("files"), list) and saved["files"]
                    and all(isinstance(path, str) and path.startswith(source["prefix"])
                            and path.endswith(source["suffix"]) for path in saved["files"])):
                return stable_order(sorted(saved["files"]))
        except (ValueError, OSError):
            pass
    endpoints = [os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")]
    fallback = os.environ.get("HF_FALLBACK_ENDPOINT", "").rstrip("/")
    if fallback and fallback not in endpoints:
        endpoints.append(fallback)
    for attempt in range(1, retries + 1):
        for endpoint in endpoints:
            check_pause()
            request = Request(f"{endpoint}/api/datasets/{source['repo']}/revision/{source['revision']}",
                              headers={"User-Agent": USER_AGENT})
            try:
                # Use the same verified TLS/proxy path as data downloads.
                # httpx's certifi store may differ from urllib's system trust.
                with urlopen(request, timeout=timeout) as response:
                    info = json.load(response)
            except (OSError, TimeoutError) as error:
                last_error = error
                print(f"[{component}] metadata attempt {attempt}/{retries} at {endpoint}: {type(error).__name__}", flush=True)
                continue
            if info.get("sha") != source["revision"]:
                raise ValueError("Upstream metadata does not match the pinned revision")
            files = sorted({item["rfilename"] for item in info["siblings"]
                            if item["rfilename"].startswith(source["prefix"])
                            and item["rfilename"].endswith(source["suffix"])})
            if not files:
                raise ValueError(f"No upstream files for {component}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"upstream_repo": source["repo"], "upstream_revision": source["revision"], "files": files}, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, cache)
            return stable_order(files)
        if attempt < retries:
            time.sleep(min(2 ** attempt, 30))
    raise last_error


def completed_state(component):
    reports_dir = ACTIVE_OUT / component / "reports"
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


def process_rows(component, rows, shard_name, destination, tokenizer, remaining, review_destination=None):
    digest = hashlib.sha256()
    stats = Counter()
    seen_ids = set()
    review_hash = hashlib.sha256()
    keyword_hits = Counter()
    review_rows = keyword_rows = 0
    tokens = rows_written = bytes_written = 0
    for row_index, row in enumerate(rows):
        if row_index % 1024 == 0:
            check_pause()
        stats["scanned"] += 1
        text, rejection = accepted_text(component, row)
        if rejection:
            stats[f"rejected_{rejection}"] += 1
            continue
        source_id = str(row.get("id") or row.get("url") or f"{shard_name}:{row_index}")
        if source_id in seen_ids:
            stats["rejected_duplicate_id"] += 1
            continue
        if EXCLUSION_INDEX is not None and EXCLUSION_INDEX.contains(text):
            stats["rejected_prior_corpus_text"] += 1
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
        if IDENTITY_REVIEW is not None:
            if review_destination is None:
                raise ValueError("Identity review requires a candidate output stream")
            reasons, hits = IDENTITY_REVIEW.inspect(record)
            keyword_hits.update(hits)
            keyword_rows += bool(hits)
            if reasons:
                candidate = {
                    "_review": {"source": f"{SOURCES[component]['repo']}@{SOURCES[component]['revision']}/{shard_name}",
                                "row_number": row_index + 1, "rules": [item["rule"] for item in reasons],
                                "reasons": reasons, "instruction": "Review and decide keep/drop/edit before training. Candidates are not automatic deletion decisions."},
                    "record": record,
                }
                review_line = orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE)
                review_destination.write(review_line)
                review_hash.update(review_line)
                review_rows += 1
        line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        destination.write(line)
        digest.update(line)
        seen_ids.add(source_id)
        tokens += token_count
        rows_written += 1
        bytes_written += len(line)
    result = {
        "tokens": tokens,
        "rows": rows_written,
        "bytes": bytes_written,
        "sha256": digest.hexdigest(),
        "selection": dict(stats),
    }
    if EXCLUSION_INDEX is not None:
        result["novelty"] = {"index_sha256": EXCLUSION_INDEX.report["index_sha256"],
                             "prior_corpus_sha256": EXCLUSION_INDEX.report["source_sha256"],
                             "method": "complete text, outer whitespace stripped, BLAKE2b-128", "cross_shard_dedup_pending": True}
    if IDENTITY_REVIEW is not None:
        result["identity_review"] = {"candidate_rows": review_rows, "keyword_rows": keyword_rows,
                                     "keyword_hit_counts": dict(keyword_hits), "candidate_sha256": review_hash.hexdigest(),
                                     "dropped_rows": 0, "rewritten_rows": 0, "training_ready": False,
                                     "decision": "Raw candidates; contextual review and clean output required before training."}
    return result


def parquet_rows(path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=512):
        yield from batch.to_pylist()


def direct_dataset_download(repo, revision, upstream_file, raw_dir, timeout):
    """Download a Hub file with an atomic, HTTP-Range-resumable partial file.

    Endpoints are tried in order: HF_ENDPOINT (primary) then
    HF_FALLBACK_ENDPOINT (default hf-mirror), sharing one resumable partial.
    """
    destination = raw_dir / upstream_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    endpoints = [os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")]
    fallback = os.environ.get("HF_FALLBACK_ENDPOINT", "").rstrip("/")
    if fallback and fallback not in endpoints:
        endpoints.append(fallback)
    last_error = None
    for endpoint in endpoints:
        # re-read the partial size: a failed attempt may have extended it
        existing = partial.stat().st_size if partial.exists() else 0
        url = (
            f"{endpoint}/datasets/{repo}/resolve/{revision}/"
            f"{quote(upstream_file, safe='/')}?download=true"
        )
        try:
            return _download_to(url, partial, existing, timeout, upstream_file, destination)
        except (EOFError, OSError, ConnectionError, TimeoutError) as error:
            last_error = error
            continue
    raise last_error if last_error else RuntimeError("no endpoint attempted")


def _download_to(url, partial, existing, timeout, upstream_file, destination):
    check_pause()
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
        if status == 206:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if content_range is None or int(content_range[1]) != existing:
                raise OSError(f"Invalid Content-Range for {upstream_file}; refusing to concatenate misaligned data")
            # Some endpoints return a capped range even when asked for the rest
            # of a file. Keep the partial and retry until the *total* is present.
            expected = int(content_range[3])
        mode = "ab" if append else "wb"
        downloaded = existing
        next_report = ((downloaded // (256 * 1024 * 1024)) + 1) * 256 * 1024 * 1024
        with partial.open(mode) as output:
            while True:
                check_pause()
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
    component_dir = ACTIVE_OUT / component
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
    review_output = component_dir / "review_candidates" / f"{stem}.jsonl"
    review_temporary = review_output.with_suffix(".jsonl.tmp")
    if IDENTITY_REVIEW is not None:
        review_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        review_context = review_temporary.open("wb") if IDENTITY_REVIEW is not None else nullcontext(None)
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination, review_context as review_destination:
            if component == "code":
                local_path = direct_dataset_download(
                    source["repo"], source["revision"], upstream_file,
                    raw_dir, timeout,
                )
                compressed_bytes = local_path.stat().st_size
                with gzip.open(local_path, mode="rb") as archive:
                    rows = (orjson.loads(line) for line in archive)
                    result = process_rows(
                        component, rows, upstream_file, destination, tokenizer, remaining, review_destination
                    )
            else:
                local_path = direct_dataset_download(
                    source["repo"], source["revision"], upstream_file,
                    raw_dir, timeout,
                )
                compressed_bytes = local_path.stat().st_size
                result = process_rows(
                    component, parquet_rows(local_path), upstream_file,
                    destination, tokenizer, remaining, review_destination,
                )
            destination.flush()
            os.fsync(destination.fileno())
            if review_destination is not None:
                review_destination.flush()
                os.fsync(review_destination.fileno())
        os.replace(temporary, output)
        if IDENTITY_REVIEW is not None:
            os.replace(review_temporary, review_output)
            result["identity_review"]["candidate_output"] = str(review_output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        if review_temporary.exists():
            review_temporary.unlink()
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
    report_temporary = report_path.with_suffix(".json.tmp")
    report_temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(report_temporary, report_path)
    return report


def skip_roots(args):
    """Directories whose per-shard reports define earlier-consumed shards."""
    roots = []
    if args.segment > 1:
        for prior in range(1, args.segment):
            prior_dir = segment_out_root(prior)
            if not prior_dir.is_dir():
                raise FileNotFoundError(
                    f"segment {args.segment} requires prior segment directory {prior_dir}"
                )
            roots.append(prior_dir)
    for extra in args.skip_reports_from or []:
        path = Path(extra)
        if not path.is_dir():
            raise FileNotFoundError(f"--skip-reports-from {path} is not a directory")
        roots.append(path)
    return roots


def skipped_upstream_files(component, args):
    skip = set()
    for root in skip_roots(args):
        reports_dir = root / component / "reports"
        if not reports_dir.is_dir():
            continue
        for path in reports_dir.glob("*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            skip.add(report["upstream_file"])
    return skip


def verify_prior_segments_complete(component, args):
    """Chained prior segments must have reached the component target.

    Skipping the shards of an unfinished segment would let this run consume
    shards a prior segment still intends to process, so a short prior segment
    is a hard error; explicit --skip-reports-from dirs are exempt.
    """
    if args.segment <= 1:
        return
    for prior in range(1, args.segment):
        reports_dir = segment_out_root(prior) / component / "reports"
        tokens = 0
        if reports_dir.is_dir():
            for path in reports_dir.glob("*.json"):
                tokens += int(json.loads(path.read_text(encoding="utf-8"))["tokens"])
        if tokens < TARGETS[component] - MAX_TOKENS:
            raise RuntimeError(
                f"segment {args.segment} requires segment {prior} to be complete for "
                f"{component}: {tokens:,}/{TARGETS[component]:,} tokens collected"
            )


def collect_component(component, args, tokenizer):
    check_pause()
    completed, total_tokens = completed_state(component)
    if total_tokens >= TARGETS[component] - MAX_TOKENS:
        print(f"[{component}] already complete: {total_tokens:,}/{TARGETS[component]:,} tokens; no network request needed", flush=True)
        return total_tokens
    verify_prior_segments_complete(component, args)
    files = list_source_files(component, timeout=args.timeout, retries=args.retries)
    skip = skipped_upstream_files(component, args)
    if skip:
        before = len(files)
        files = [name for name in files if name not in skip]
        print(
            f"[{component}] segment {args.segment}: skipping {before - len(files)} "
            f"upstream shards already consumed by earlier segments", flush=True,
        )
    files = files[args.worker_index::args.shard_workers]
    processed_now = 0
    print(f"[{component}] worker={args.worker_index}/{args.shard_workers} "
          f"resume={total_tokens:,}/{TARGETS[component]:,} tokens, "
          f"completed_shards={len(completed)}", flush=True)
    for upstream_file in files:
        check_pause()
        if total_tokens >= TARGETS[component] - MAX_TOKENS:
            break
        if upstream_file in completed:
            continue
        report = None
        for attempt in range(1, args.retries + 1):
            check_pause()
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
            or args.shard_workers < 1 or args.segment < 1
            or not 0 <= args.worker_index < args.shard_workers):
        raise ValueError(
            "timeout/retries/shard-workers must be positive, max-shards >= 0, "
            "segment >= 1, and worker-index must be within shard-workers"
        )
    global ACTIVE_OUT, EXCLUSION_INDEX, IDENTITY_REVIEW
    if args.target_tokens is not None:
        if args.component == "all" or args.target_tokens < MAX_TOKENS:
            raise ValueError("--target-tokens requires one component and at least 4096 tokens")
        TARGETS[args.component] = args.target_tokens
    if args.novel_index is not None:
        from scripts.data_builder.coder_novelty import ExclusionIndex
        EXCLUSION_INDEX = ExclusionIndex(args.novel_index)
    if args.identity_review:
        from scripts.data_builder.coder_novelty import IdentityReview
        IDENTITY_REVIEW = IdentityReview()
    ACTIVE_OUT = (Path(args.out_dir).resolve() if args.out_dir
                  else segment_out_root(args.segment))
    roots = skip_roots(args)  # validated eagerly so a bad chain fails fast
    print(
        f"[segment {args.segment}] output root {ACTIVE_OUT}; "
        f"skipping shards from {len(roots)} prior run(s)",
        flush=True,
    )
    ACTIVE_OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True).backend_tokenizer
    components = list(TARGETS) if args.component == "all" else [args.component]
    summary = {}
    for component in components:
        component_root = ACTIVE_OUT / component
        component_root.mkdir(parents=True, exist_ok=True)
        selection = {"source": SOURCES[component], "target_tokens": TARGETS[component],
                     "prior_shard_roots": [str(root.resolve()) for root in roots],
                     "exclusion_index_sha256": EXCLUSION_INDEX.report["index_sha256"] if EXCLUSION_INDEX else None,
                     "identity_review": bool(IDENTITY_REVIEW), "max_tokens": MAX_TOKENS}
        selection_path = component_root / "selection.json"
        if selection_path.exists():
            if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
                raise ValueError(f"Collection selection changed: {selection_path}; use a fresh output directory")
        else:
            if (EXCLUSION_INDEX is not None or IDENTITY_REVIEW is not None) and list((component_root / "reports").glob("*.json")):
                raise ValueError("Cannot attach novelty/review rules to previously collected shards")
            selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[component] = collect_component(component, args, tokenizer)
    print(json.dumps({"targets": TARGETS, "collected": summary}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except CollectionPaused as error:
        print(f"[paused] {error}", flush=True)
        sys.exit(42)
