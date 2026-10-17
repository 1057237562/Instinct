"""Pre-place the LANGUAGE_PLAN stackv2 shards in the collectors' raw directory.

Both collect_stackv2_code_4096.py and collect_coder_pretrain_12b.py read these
raws when present, so downloading them once (with parallel ranges) avoids
duplicated single-stream downloads. Reruns skip completed files and resume
partial ones via fetch_parallel's journal.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data_builder.collect_arxiv_pretrain import hf_endpoint
from scripts.data_builder.collect_stackv2_code import LANGUAGE_PLAN, REPO, REVISION
from scripts.data_builder.fetch_parallel import download


def main() -> None:
    raw = Path(__file__).resolve().parents[2] / "dataset/coder_pretrain_12b/code/raw"
    raw.mkdir(parents=True, exist_ok=True)
    entries = [(language, shard) for language, (_, shards) in LANGUAGE_PLAN.items() for shard in shards]
    done = failed = skipped = 0
    for language, shard in entries:
        name = f"stack-edu-{shard:04d}.json.gz"
        out = raw / name
        if out.is_file():
            print(f"{name}: exists, skip", flush=True)
            skipped += 1
            continue
        if out.with_suffix(".json.gz.part").exists():
            print(f"{name}: being downloaded by the 12B collector, skip", flush=True)
            skipped += 1
            continue
        url = f"{hf_endpoint()}/datasets/{REPO}/resolve/{REVISION}/{name}?download=true"
        for attempt in range(1, 4):
            try:
                download(url, out, connections=8)
                done += 1
                break
            except Exception as error:
                print(f"{name}: attempt {attempt} failed: {type(error).__name__}: {str(error)[:140]}", flush=True)
                if attempt == 3:
                    failed += 1
                else:
                    time.sleep(10)
    print(f"prefetch complete: downloaded={done} skipped={skipped} failed={failed} of {len(entries)}", flush=True)


if __name__ == "__main__":
    main()
