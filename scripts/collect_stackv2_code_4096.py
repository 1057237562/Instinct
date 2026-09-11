"""Collect 480M <=4096-token open-repository code tokens, without splitting."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import orjson
from transformers import AutoTokenizer

if __package__ in (None, ""):
    __package__ = "scripts"
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.collect_arxiv_pretrain import CountingReader, USER_AGENT, shard_url
from scripts.collect_stackv2_code import LANGUAGE_PLAN, REPO, REVISION


TARGET_TOKENS = 480_000_000
MAX_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/pretrain_stackv2_code_480m_tokens_4096.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("dataset/pretrain_stackv2_code_480m_tokens_4096.report.json"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, report_path = args.output.resolve(), args.report.resolve()
    temporary = output.with_name(output.name + ".tmp")
    for path in (output, report_path, temporary):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path.resolve(), trust_remote_code=True)
    backend = tokenizer.backend_tokenizer

    total_weight = sum(target for target, _ in LANGUAGE_PLAN.values())
    targets = {}
    assigned = 0
    for index, (language, (weight, _)) in enumerate(LANGUAGE_PLAN.items()):
        target = TARGET_TOKENS - assigned if index + 1 == len(LANGUAGE_PLAN) else TARGET_TOKENS * weight // total_weight
        targets[language] = target
        assigned += target

    digest = hashlib.sha256()
    seen: set[str] = set()
    stats: Counter[str] = Counter()
    language_reports = {}
    total_tokens = total_rows = output_bytes = 0
    try:
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination:
            for language, (_, shard_numbers) in LANGUAGE_PLAN.items():
                target = targets[language]
                lang_tokens = lang_rows = lang_bytes = 0
                shard_reports = []
                for shard_index, shard_number in enumerate(shard_numbers):
                    cumulative = target * (shard_index + 1) // len(shard_numbers)
                    path = f"stack-edu-{shard_number:04d}.json.gz"
                    scanned = 0
                    request = Request(shard_url(REPO, REVISION, path), headers={"User-Agent": USER_AGENT})
                    with urlopen(request, timeout=args.timeout) as response:
                        counted = CountingReader(response)
                        with gzip.GzipFile(fileobj=counted, mode="rb") as archive:
                            for raw_line in archive:
                                scanned += 1
                                row = orjson.loads(raw_line)
                                metadata = row.get("metadata") or {}
                                if metadata.get("language") != language:
                                    stats["rejected_language"] += 1
                                    continue
                                if int(row.get("int_score") or 0) < 3:
                                    stats["rejected_score"] += 1
                                    continue
                                if metadata.get("is_vendor") or metadata.get("is_generated"):
                                    stats["rejected_vendor_or_generated"] += 1
                                    continue
                                text = row.get("text")
                                source_id = str(row.get("id") or "")
                                if not isinstance(text, str) or len(text) < 200 or not source_id:
                                    stats["rejected_short_or_empty"] += 1
                                    continue
                                if source_id in seen:
                                    stats["rejected_duplicate"] += 1
                                    continue
                                token_count = len(backend.encode(text, add_special_tokens=False).ids) + 2
                                if token_count > MAX_TOKENS:
                                    stats["rejected_over_4096"] += 1
                                    continue
                                remaining = cumulative - lang_tokens
                                if token_count > remaining:
                                    stats["rejected_over_budget"] += 1
                                    if remaining < MAX_TOKENS:
                                        break
                                    continue
                                record = {
                                    "text": text,
                                    "source": f"{REPO}:{language}",
                                    "source_id": source_id,
                                    "license": str(metadata.get("license") or ""),
                                    "url": str(metadata.get("url") or ""),
                                    "token_count": token_count,
                                }
                                line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
                                destination.write(line); digest.update(line); seen.add(source_id)
                                lang_tokens += token_count; lang_rows += 1; lang_bytes += len(line)
                                total_tokens += token_count; total_rows += 1; output_bytes += len(line)
                    shard_reports.append({"path": path, "rows_scanned": scanned, "compressed_bytes_read": counted.bytes_read})
                language_reports[language] = {"target_tokens": target, "tokens": lang_tokens, "rows": lang_rows, "output_bytes": lang_bytes, "shards": shard_reports}
                print(f"{language}: {lang_tokens:,}/{target:,} tokens, rows={lang_rows:,}", flush=True)
            destination.flush(); os.fsync(destination.fileno())
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise

    report = {
        "name": "pretrain_stackv2_code_4096", "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"repository": REPO, "revision": REVISION, "url": f"https://huggingface.co/datasets/{REPO}"},
        "tokenizer": str(args.tokenizer_path.resolve()), "max_tokens_including_bos_eos": MAX_TOKENS,
        "whole_documents_only": True, "text_split": False, "text_truncated": False,
        "output": str(output), "output_rows": total_rows, "output_tokens": total_tokens,
        "output_bytes": output_bytes, "output_sha256": digest.hexdigest(),
        "languages": language_reports, "selection_stats": dict(sorted(stats.items())),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
