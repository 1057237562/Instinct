"""Deterministic language-stratified audit of L3 exclusion rules (read only)."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scripts.data_builder import clean_ultradata_l3_cpt as clean
from scripts.data_builder.coder_pretrain_common import words
import numpy as np
import orjson
import pyarrow.parquet as pq


def sample_shard(task):
    path, lang, limit, seed = task
    rng = random.Random(seed)
    parquet = pq.ParquetFile(path)
    group = rng.randrange(parquet.num_row_groups)
    offset = sum(parquet.metadata.row_group(i).num_rows for i in range(group))
    table = parquet.read_row_group(group, columns=["uuid", "task", "analysis", "solution", "test", "full_content"])
    indices = sorted(rng.sample(range(len(table)), min(limit, len(table))))
    rows = table.take(indices).to_pylist()
    counts, examples, reason_count = Counter(), [], Counter()
    for index, row in zip(indices, rows):
        counts["sampled"] += 1
        reason = clean.quality_reason(row, lang)
        token_count = None
        matching = []
        if not reason:
            digest = np.void(clean.fp(row["full_content"]))
            pos = int(np.searchsorted(clean._CROSS, digest))
            if pos < len(clean._CROSS) and clean._CROSS[pos] == digest:
                reason = "prior_corpus_exact_duplicate"
            else:
                token_count = len(clean._TOKENIZER.encode(row["full_content"], add_special_tokens=False).ids) + 2
                if token_count < 128:
                    reason = "below_128"
                elif token_count > 4096:
                    reason = "above_4096"
                else:
                    tokens = words(row["full_content"])
                    matching = [list(tokens[i:i+13]) for i in range(len(tokens)-12)
                                if tuple(tokens[i:i+13]) in clean._BENCH]
                    if matching:
                        reason = "benchmark_13gram"
        counts[reason or "pass_quality_length_benchmark"] += 1
        if reason and reason_count[reason] < 8:
            reason_count[reason] += 1
            examples.append({"reason": reason, "language": lang, "source": str(path),
                             "source_row": offset + index + 1, "uuid": row["uuid"],
                             "token_count": token_count, "benchmark_ngrams": matching[:6],
                             "field_lengths": {k: len(row.get(k) or "") for k in ("task", "analysis", "solution", "test")},
                             "record": row})
    return {"language": lang, "source": str(path), "counts": dict(counts), "examples": examples}


if __name__ == "__main__":
    rng, tasks = random.Random(20261014), []
    for lang in clean.WEIGHTS:
        paths = sorted((clean.upstream.RAW / "data/UltraData-Code-L3" / lang).glob("*.parquet"))
        chosen = rng.sample(paths, min(len(paths), 4 if lang in ("py", "cpp") else 1))
        for path in chosen:
            tasks.append((path, lang, 4000 if lang == "py" else 2000, rng.randrange(2**32)))
    results, counts, languages = [], Counter(), {}
    destination = clean.WORK / "filter_review_samples.jsonl"
    with destination.open("wb") as output, ProcessPoolExecutor(max_workers=4, initializer=clean.init_worker) as pool:
        for future in as_completed([pool.submit(sample_shard, task) for task in tasks]):
            result = future.result()
            examples = result.pop("examples")
            for example in examples:
                output.write(orjson.dumps(example) + b"\n")
            results.append(result)
            counts.update(result["counts"])
            languages.setdefault(result["language"], Counter()).update(result["counts"])
            print(f"[sample-review] {len(results)}/{len(tasks)} sources, {counts['sampled']:,} rows", flush=True)
    clean.dump(clean.WORK / "filter_review_samples.report.json",
               {"seed": 20261014, "sampling": "random shard selection, random row group and random rows, stratified by language; not a corpus-weighted prevalence estimate",
                "counts": dict(counts), "languages": {k: dict(v) for k, v in languages.items()},
                "sources": results, "samples": str(destination),
                "samples_sha256": clean.upstream.sha256_file(destination)})
