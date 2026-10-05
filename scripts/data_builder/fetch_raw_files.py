"""Fetch pinned raw dataset files from the Hub into their card directories.

Usage: HF_ENDPOINT=https://hf-mirror.com python scripts/data_builder/fetch_raw_files.py [--only name ...]
Files land exactly where the builders expect them. hf_hub_download caches under
dataset/.hf_raw_cache and copies into place; reruns are cheap no-ops when the
target already exists.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from huggingface_hub import hf_hub_download

# name -> (repo, revision, filename, destination relative to dataset/)
FILES = {
    "mathinstruct": ("allenai/MathInstruct", None, "MathInstruct.json", "math-instruct/MathInstruct.json"),
    "bigcode": ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "356bb069eee815daa6e23e9a282eeefe1490ad44", "data/train-00000-of-00001.parquet", "bigcode-self-oss-instruct-50k/data/train-00000-of-00001.parquet"),
    "vezora": ("Vezora/Tested-22k-Python-Alpaca", "578b29091cce76ec1b464db1bb76cd37c4e9a7bf", "188k-Vezora-PyCode-Alpaca.json", "vezora-tested-python-22k/188k-Vezora-PyCode-Alpaca.json"),
    "numina_0": ("AI-MO/NuminaMath-1.5", "1b05109f9e5c1ad06c0663519502416c30b300f8", "data/train-00000-of-00003.parquet", "numinamath-1.5/data/train-00000-of-00003.parquet"),
    "numina_1": ("AI-MO/NuminaMath-1.5", "1b05109f9e5c1ad06c0663519502416c30b300f8", "data/train-00001-of-00003.parquet", "numinamath-1.5/data/train-00001-of-00003.parquet"),
    "numina_2": ("AI-MO/NuminaMath-1.5", "1b05109f9e5c1ad06c0663519502416c30b300f8", "data/train-00002-of-00003.parquet", "numinamath-1.5/data/train-00002-of-00003.parquet"),
    "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K", "5f839b1f368a76b161028bb9edff055db34022b2", "data-oss_instruct-decontaminated.jsonl", "magicoder-oss-75k/data-oss_instruct-decontaminated.jsonl"),
    "verified_math_3k": ("HSH-Intelligence/verified-math-reasoning-3k", "3726afcbcf556adbb3df2611a6d0d14b45cc49fa", "data/train.jsonl", "verified-math-reasoning-3k/data/train.jsonl"),
    "competitive_python_00": ("nvidia/Nemotron-SFT-Competitive-Programming-v2", None, "data/competitive_programming_python_00.jsonl", "competitive-coding/data/competitive_programming_python_00.jsonl"),
    "competitive_python_01": ("nvidia/Nemotron-SFT-Competitive-Programming-v2", None, "data/competitive_programming_python_01.jsonl", "competitive-coding/data/competitive_programming_python_01.jsonl"),
    "competitive_cpp_00": ("nvidia/Nemotron-SFT-Competitive-Programming-v2", None, "data/competitive_programming_cpp_00.jsonl", "competitive-coding/data/competitive_programming_cpp_00.jsonl"),
    "competitive_cpp_01": ("nvidia/Nemotron-SFT-Competitive-Programming-v2", None, "data/competitive_programming_cpp_01.jsonl", "competitive-coding/data/competitive_programming_cpp_01.jsonl"),
    "text_to_sql": ("nvidia/Nemotron-SFT-Competitive-Programming-v2", None, "data/text_to_sql.jsonl", "competitive-coding/data/text_to_sql.jsonl"),
    "exercism": ("nvidia/Nemotron-SFT-Competitive-Programming-v2", None, "data/exercism.jsonl", "competitive-coding/data/exercism.jsonl"),
}

# open-r1/codeforces problems + submissions: raw for reconstructing the
# verified_competitive_submissions component (original copy-from-old-mix path
# is unreproducible; see scripts/data_builder/build_verified_submissions.py).
CODEFORCES_PROBLEMS_REV = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
FILES["codeforces_test"] = ("open-r1/codeforces", CODEFORCES_PROBLEMS_REV, "data/test-00000-of-00001.parquet", "codeforces-problems/data/test-00000-of-00001.parquet")
for _index in range(11):
    FILES[f"codeforces_train_{_index:02d}"] = (
        "open-r1/codeforces", CODEFORCES_PROBLEMS_REV,
        f"data/train-{_index:05d}-of-00011.parquet",
        f"codeforces-problems/data/train-{_index:05d}-of-00011.parquet",
    )
for _index in range(36):
    FILES[f"codeforces_submissions_{_index:02d}"] = (
        "open-r1/codeforces-submissions", None,
        f"data/train-{_index:05d}-of-00036.parquet",
        f"codeforces-submissions/data/train-{_index:05d}-of-00036.parquet",
    )

# Pre-place the first-in-stable-order supplement raws so the 12B collector
# (collect_coder_pretrain_12b.py) reuses them instead of re-downloading; the
# same files feed the general_bilingual substitute build.
FILES["sub_en_00"] = ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", "fineweb-edu-dedup/train-00000-of-00234.parquet", "coder_pretrain_12b/natural_en/raw/fineweb-edu-dedup/train-00000-of-00234.parquet")
FILES["sub_en_37"] = ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", "fineweb-edu-dedup/train-00037-of-00234.parquet", "coder_pretrain_12b/natural_en/raw/fineweb-edu-dedup/train-00037-of-00234.parquet")
FILES["sub_zh_00"] = ("opencsg/chinese-fineweb-edu", "784f118864c0741ce32aa74d9588795bdf1a896b", "IndustryCorpus/00000.parquet", "coder_pretrain_12b/natural_zh/raw/IndustryCorpus/00000.parquet")
FILES["sub_zh_37"] = ("opencsg/chinese-fineweb-edu", "784f118864c0741ce32aa74d9588795bdf1a896b", "IndustryCorpus/00037.parquet", "coder_pretrain_12b/natural_zh/raw/IndustryCorpus/00037.parquet")


def fetch(name: str, repo: str, revision: str | None, filename: str, destination: str) -> str:
    target = ROOT / "dataset" / destination
    if target.is_file() and target.stat().st_size > 0:
        return f"{name}: exists ({target.stat().st_size:,} bytes)"
    cached = hf_hub_download(
        repo, filename, repo_type="dataset",
        revision=revision or "main",
        local_dir=ROOT / "dataset/.hf_raw_cache" / name,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)
    return f"{name}: fetched {target.stat().st_size:,} bytes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    for name, (repo, revision, filename, destination) in FILES.items():
        if args.only and name not in args.only:
            continue
        for attempt in range(1, 4):
            try:
                print(fetch(name, repo, revision, filename, destination), flush=True)
                break
            except Exception as error:
                print(f"{name}: attempt {attempt} failed: {type(error).__name__}: {str(error)[:200]}", flush=True)
                if attempt == 3:
                    print(f"{name}: FAILED", flush=True)


if __name__ == "__main__":
    main()
