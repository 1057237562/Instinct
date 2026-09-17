"""Recover source caches from interrupted ExternalRandomShuffler chunk files."""
from collections import Counter
import json
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
CACHE = DATA / "continue_pretrain_1b" / "filtered_cache"


def scan(directory):
    stats = {}
    for path in sorted(directory.glob("chunk_*.jsonl")):
        with path.open("rb") as f:
            for line in f:
                _, body = line.rstrip(b"\n").split(b"\t", 1)
                row = orjson.loads(body); source = row["source"]
                value = stats.setdefault(source, Counter())
                value["rows"] += 1; value["tokens"] += int(row["token_count"]); value["bytes"] += len(body) + 1
    return stats


def main():
    directories = sorted(DATA.glob(".sft_mix_*"))
    scans = {str(path): scan(path) for path in directories}
    print(json.dumps({key: {s: dict(v) for s, v in value.items()} for key, value in scans.items()}, indent=2))
    wanted = ("python_edu_novel", "cosmopedia_novel", "verified_math_code", "open_code_cot_novel")
    best = {}
    for source in wanted:
        candidates = [(stats.get(source, Counter())["tokens"], path) for path, stats in scans.items()]
        tokens, path = max(candidates)
        if not tokens: raise RuntimeError(f"No recoverable rows for {source}")
        best[source] = Path(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    outputs = {}; handles = {}
    try:
        for source, path in best.items():
            handles[source] = (CACHE / f"{source}.jsonl").open("wb")
            outputs[source] = Counter(source_directory=str(path))
        # A source's best directory contains that source exactly once; split it
        # directly without re-tokenizing or changing serialized records.
        by_dir = {}
        for source, path in best.items(): by_dir.setdefault(path, set()).add(source)
        for path, sources in by_dir.items():
            for chunk in sorted(path.glob("chunk_*.jsonl")):
                with chunk.open("rb") as f:
                    for line in f:
                        _, body = line.rstrip(b"\n").split(b"\t", 1)
                        row = orjson.loads(body); source = row["source"]
                        if source not in sources: continue
                        handles[source].write(body + b"\n")
                        outputs[source]["rows"] += 1; outputs[source]["tokens"] += int(row["token_count"])
                        outputs[source]["bytes"] += len(body) + 1
    finally:
        for handle in handles.values(): handle.close()
    report = {source: dict(values) for source, values in outputs.items()}
    (CACHE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recovered": report}, indent=2))


if __name__ == "__main__": main()
