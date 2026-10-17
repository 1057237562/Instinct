"""Orchestrate the remaining raw downloads with parallel-range fetching.

3 files in flight x 6 ranged connections each; resumable via .ppart pieces;
files land directly at their final destinations (no cache detour).
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data_builder.fetch_parallel import download

MIRROR = "https://hf-mirror.com/datasets"
CODEFORCES_REV = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
SMOLLM_REV = "3ba9d605774198c5868892d7a8deda78031a781f"
CHINESE_REV = "784f118864c0741ce32aa74d9588795bdf1a896b"

JOBS = []
# The four ~24 GB competitive_programming_*.jsonl shards are NOT downloaded
# whole; extract_competitive.py streams and selects only the rows the builder
# consumes (see that script). text_to_sql and exercism are small enough to keep.
for name in ("text_to_sql", "exercism"):
    JOBS.append((f"{MIRROR}/nvidia/Nemotron-SFT-Competitive-Programming-v2/resolve/main/data/{name}.jsonl?download=true",
                 f"dataset/competitive-coding/data/{name}.jsonl"))
for shard in range(12):
    kind = "test-00000-of-00001" if shard == 0 else f"train-{shard-1:05d}-of-00011"
    JOBS.append((f"{MIRROR}/open-r1/codeforces/resolve/{CODEFORCES_REV}/data/{kind}.parquet?download=true",
                 f"dataset/codeforces-problems/data/{kind}.parquet"))
# sub parquets before the 36 submissions: they gate the general_bilingual
# substitute on the codespecialist critical path
for shard in ("00000", "00037"):
    JOBS.append((f"{MIRROR}/HuggingFaceTB/smollm-corpus/resolve/{SMOLLM_REV}/fineweb-edu-dedup/train-{shard}-of-00234.parquet?download=true",
                 f"dataset/coder_pretrain_12b/natural_en/raw/fineweb-edu-dedup/train-{shard}-of-00234.parquet"))
JOBS.append((f"{MIRROR}/opencsg/chinese-fineweb-edu/resolve/{CHINESE_REV}/IndustryCorpus/00000.parquet?download=true",
             "dataset/coder_pretrain_12b/natural_zh/raw/IndustryCorpus/00000.parquet"))
JOBS.append((f"{MIRROR}/opencsg/chinese-fineweb-edu/resolve/{CHINESE_REV}/IndustryCorpus/00037.parquet?download=true",
             "dataset/coder_pretrain_12b/natural_zh/raw/IndustryCorpus/00037.parquet"))
for shard in range(36):
    JOBS.append((f"{MIRROR}/open-r1/codeforces-submissions/resolve/main/data/train-{shard:05d}-of-00036.parquet?download=true",
                 f"dataset/codeforces-submissions/data/train-{shard:05d}-of-00036.parquet"))

ROOT = Path(__file__).resolve().parents[2]


def run_job(job):
    url, destination = job
    out = ROOT / destination
    if out.is_file() and out.stat().st_size > 0:
        return f"SKIP {destination}"
    for attempt in range(3):
        try:
            download(url, out, connections=6)
            return f"DONE {destination}"
        except Exception as error:
            out.with_suffix(out.suffix + ".ppart").unlink(missing_ok=True)
            if attempt == 2:
                return f"FAIL {destination}: {type(error).__name__}: {str(error)[:150]}"
            time.sleep(5)


def main() -> None:
    pending = [job for job in JOBS if not (ROOT / job[1]).is_file()]
    print(f"{len(JOBS)} jobs, {len(pending)} pending", flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run_job, pending):
            print(result, flush=True)
            if result.startswith("FAIL"):
                failures.append(result)
    print(f"complete; failures={len(failures)}", flush=True)
    for failure in failures:
        print(failure, flush=True)


if __name__ == "__main__":
    main()
