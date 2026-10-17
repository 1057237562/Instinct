"""Prefetch the first shards of segment-1's not-yet-started components.

Segment-1's math/natural_en/natural_zh boundary runs gate the segment-2 relay;
pre-placing their first candidate shards turns each collector cycle from
download-bound into tokenize-bound.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_builder.collect_arxiv_pretrain import hf_endpoint  # noqa: E402
from scripts.data_builder.collect_coder_pretrain_12b import (  # noqa: E402
    list_source_files, segment_out_root,
)
from scripts.data_builder.fetch_parallel import download  # noqa: E402

SEG1 = segment_out_root(1)
PLAN = {"math": 8, "natural_en": 4, "natural_zh": 4}
SOURCES = {
    "math": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594"),
    "natural_en": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f"),
    "natural_zh": ("opencsg/chinese-fineweb-edu", "784f118864c0741ce32aa74d9588795bdf1a896b"),
}


def main() -> None:
    for component, count in PLAN.items():
        repo, revision = SOURCES[component]
        try:
            files = list_source_files(component)[:count]
        except Exception as error:
            print(f"{component}: listing failed: {error}", flush=True)
            continue
        raw = SEG1 / component / "raw"
        for name in files:
            out = raw / name
            if out.is_file() or out.with_suffix(out.suffix + ".ppart").exists():
                print(f"{component}/{name}: skip", flush=True)
                continue
            url = f"{hf_endpoint()}/datasets/{repo}/resolve/{revision}/{name}?download=true"
            for attempt in range(1, 3):
                try:
                    download(url, out, connections=8)
                    break
                except Exception as error:
                    print(f"{component}/{name} attempt {attempt}: {type(error).__name__}: {str(error)[:100]}", flush=True)
                    time.sleep(10)
    print("seg1 next-component prefetch complete", flush=True)


if __name__ == "__main__":
    main()
