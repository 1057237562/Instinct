"""Separate >4096 and <128 exclusions from the existing combined counter.

The project's non-normalizing ByteLevel BPE covers at most 21 input bytes per
token. Therefore a document longer than 126*21 bytes cannot be <128 tokens
including BOS/EOS. Only shorter documents need retokenization. The long count
is then the recorded length rejection count minus this exact short count.
Raw data and previously filtered artifacts remain unchanged.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scripts.data_builder import clean_ultradata_l3_cpt as clean
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

WORK = clean.WORK
TOKENIZER = CROSS = None
BYTE_BOUND = None


def init():
    global TOKENIZER, CROSS, BYTE_BOUND
    definition = json.loads((ROOT / "model/tokenizer.json").read_text(encoding="utf-8"))
    if definition["normalizer"] is not None or definition["pre_tokenizer"]["type"] != "ByteLevel" or definition["truncation"] is not None:
        raise ValueError("The byte-coverage proof requires non-normalizing, untruncated ByteLevel BPE")
    TOKENIZER = Tokenizer.from_file(str(ROOT / "model/tokenizer.json"))
    maximum = max(max(map(len, definition["model"]["vocab"])),
                  max(len(t.content.encode("utf-8")) for t in TOKENIZER.get_added_tokens_decoder().values()))
    BYTE_BOUND = 126 * maximum
    CROSS = np.memmap(WORK / "prior.blake2b128", dtype="V16", mode="r")


def count_short(report):
    destination = Path(report["output"]).with_suffix(".lengths.json")
    if destination.exists():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if cached["source_sha256"] != report["source_sha256"]:
            raise ValueError("Length cache input differs")
        return cached
    source = Path(report["source"])
    rows_to_check = []
    row_number = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=2048, columns=["full_content"]):
        for text in batch.column(0).to_pylist():
            row_number += 1
            if isinstance(text, str) and len(text.encode("utf-8")) <= BYTE_BOUND:
                if len(TOKENIZER.encode(text, add_special_tokens=False).ids) + 2 < 128:
                    rows_to_check.append(row_number)
    shorts = 0
    if rows_to_check:
        wanted = set(rows_to_check)
        offset = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1024, columns=["task", "analysis", "solution", "test", "full_content"]):
            local = [i for i in range(len(batch)) if offset + i + 1 in wanted]
            if local:
                for row in batch.take(local).to_pylist():
                    if clean.quality_reason(row, report["language"]):
                        continue
                    digest = np.void(clean.fp(row["full_content"]))
                    pos = int(np.searchsorted(CROSS, digest))
                    if pos < len(CROSS) and CROSS[pos] == digest:
                        continue
                    shorts += 1
            offset += len(batch)
    total = report["counts"].get("drop:token_length", 0)
    if shorts > total or row_number != report["counts"]["input_rows"]:
        raise ValueError("Length census does not reconcile with filtering report")
    result = {"source": report["source"], "source_sha256": report["source_sha256"],
              "language": report["language"], "input_rows": row_number,
              "below_128": shorts, "above_4096": total - shorts,
              "byte_bound_for_short_candidates": BYTE_BOUND}
    clean.dump(destination, result)
    return result


if __name__ == "__main__":
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(WORK.glob("UltraData*.report.json"))]
    reports = [r for r in reports if r.get("kind") != "bias_correction_supplement"]
    totals, languages, results = Counter(), {}, []
    with ProcessPoolExecutor(max_workers=6, initializer=init) as pool:
        for future in as_completed([pool.submit(count_short, r) for r in reports]):
            r = future.result()
            results.append(r)
            values = {k: r[k] for k in ("input_rows", "below_128", "above_4096")}
            totals.update(values)
            languages.setdefault(r["language"], Counter()).update(values)
            print(f"[lengths] {len(results)}/{len(reports)}", flush=True)
    result = {"completed_shards": len(results), **dict(totals),
              "above_4096_fraction": totals["above_4096"] / totals["input_rows"],
              "languages": {k: dict(v) for k, v in languages.items()},
              "method": "Exact short-document census with a proven tokenizer byte upper bound; >4096 = length rejection count minus <128", "shards": results}
    clean.dump(WORK / "length_distribution.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}), flush=True)
