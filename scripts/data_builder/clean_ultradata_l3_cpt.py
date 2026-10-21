"""Offline, resumable quality filtering of downloaded UltraData-Code-L3 shards.

Run `python scripts/data_builder/clean_ultradata_l3_cpt.py filter --workers 20`,
then `... assemble`. Identity/anomaly candidates require contextual review;
assembly quarantines vendor mentions until fingerprint-bound decisions exist.
Source code is parsed, never executed. Raw shards are never modified.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Preserve the project's Windows datasets-before-torch import order.
from datasets import load_dataset  # noqa: E402,F401
import numpy as np  # noqa: E402
import orjson  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from scripts.data_builder import build_ultradata_code_l3_12b as upstream  # noqa: E402
from scripts.data_builder.filter_anomaly_candidates import inspect_row  # noqa: E402
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler  # noqa: E402

WORK = upstream.WORK / "quality_clean_v1"
OUTPUT = ROOT / "dataset/pretrain_ultradata_code_l3_cpt_12b_identity_clean.jsonl"
REPORT = OUTPUT.with_suffix(".report.json")
POLICY = "l3-quality-v1"
TARGET = 12_000_000_000
WEIGHTS = {"py": .35, "cpp": .40, "java": .12, "js": .04, "cs": .03,
           "go": .02, "php": .02, "rs": .01, "rb": .005, "r": .003, "sh": .002}
UNFINISHED = re.compile(r"^\s*(?://|#|/\*)\s*(?:YOUR_CODE_HERE|IMPLEMENT_ME|\[(?:insert\s+|placeholder\b)[^\]]*\])", re.I | re.M)
SPECIAL = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_TOKENIZER = _BENCH = _CROSS = None


def dump(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fp(text):
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).digest()


def quality_reason(row, lang):
    for key in ("task", "analysis", "solution", "test", "full_content"):
        if not isinstance(row.get(key), str) or len(row[key].strip()) < 40:
            return "missing_or_short_" + key
    text = row["full_content"]
    if "\x00" in text or "\ufffd" in text:
        return "encoding_corruption"
    if SPECIAL.search(text):
        return "embedded_control_token"
    if UNFINISHED.search(row["solution"]):
        return "unresolved_implementation_placeholder"
    if lang == "py":
        try:
            tree = ast.parse(row["solution"])
        except (SyntaxError, ValueError, RecursionError):
            return "python_solution_syntax"
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for function in functions:
            body = function.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            # A pass-only function may intentionally be an abstract hook: triage
            # keeps those. A function containing only a raised NotImplementedError
            # is an objectively unimplemented answer.
            if len(body) == 1 and isinstance(body[0], ast.Raise):
                exc = body[0].exc
                if isinstance(exc, ast.Call):
                    exc = exc.func
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                    return "python_unimplemented_function"
    return None


def init_worker():
    global _TOKENIZER, _BENCH, _CROSS
    _TOKENIZER = Tokenizer.from_file(str(ROOT / "model/tokenizer.json"))
    with (WORK / "benchmark.pkl").open("rb") as stream:
        _BENCH = pickle.load(stream)
    _CROSS = np.memmap(WORK / "prior.blake2b128", dtype="V16", mode="r")


def process_shard(item):
    lang, upstream_path, expected_bytes, signature = item
    raw = upstream.shard_destination(upstream_path)
    stem = WORK / (Path(upstream_path).stem + ".filtered")
    out = stem.with_suffix(".jsonl")
    candidates = stem.with_suffix(".candidates.jsonl")
    report_path = stem.with_suffix(".report.json")
    identity = {"policy": POLICY, "bytes": raw.stat().st_size,
                "mtime_ns": raw.stat().st_mtime_ns, "signature": signature}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["input_signature"] != identity:
            raise ValueError(f"Changed source/policy for cached shard: {raw}")
        for path, key in ((out, "output_sha256"), (candidates, "candidates_sha256")):
            if upstream.sha256_file(path) != report[key]:
                raise ValueError(f"Changed cached output: {path}")
        return report
    if identity["bytes"] != expected_bytes:
        raise ValueError(f"Incomplete download: {raw}")
    started = time.monotonic()
    counts, keywords = Counter(), Counter()
    hashes = [hashlib.sha256(), hashlib.sha256()]
    temporary = Path(str(out) + ".tmp")
    candidate_tmp = Path(str(candidates) + ".tmp")
    row_number = 0
    with temporary.open("wb") as writer, candidate_tmp.open("wb") as reviewer:
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=256, columns=["uuid", "task", "analysis", "solution", "test", "full_content"]):
            for row in batch.to_pylist():
                row_number += 1
                counts["input_rows"] += 1
                reason = quality_reason(row, lang)
                if reason:
                    counts["drop:" + reason] += 1
                    continue
                text = row["full_content"]
                text_id = fp(text)
                pos = int(np.searchsorted(_CROSS, np.void(text_id)))
                if pos < len(_CROSS) and _CROSS[pos] == np.void(text_id):
                    counts["drop:prior_corpus_exact_duplicate"] += 1
                    continue
                count = len(_TOKENIZER.encode(text, add_special_tokens=False).ids) + 2
                if not 128 <= count <= 4096:
                    counts["drop:token_length"] += 1
                    continue
                if upstream.has_benchmark_overlap(text, _BENCH):
                    counts["drop:benchmark_13gram"] += 1
                    continue
                # Use the mandated first-pass triage before contextual screening.
                triage = inspect_row({"text": text, "solution": row["solution"]}, "all")
                verdict, hits, vendor_counts = upstream.identity_scan(text)
                keywords.update(vendor_counts)
                record = {"text": text, "token_count": count,
                          "source": "ultradata_code_l3:" + lang, "mix_category": lang,
                          "mixture": "ultradata-l3-python-cpp-cpt", "license": "apache-2.0",
                          "uuid": row["uuid"], "source_shard": upstream_path,
                          "source_row": row_number, "source_revision": upstream.REVISION,
                          "text_id": text_id.hex(),
                          "exercise_id": fp(row["task"].strip() + "\n\x1f\n" + row["solution"].strip()).hex()}
                if triage or vendor_counts or hits:
                    payload = {"record": record, "triage": triage, "identity_verdict": verdict,
                               "identity_hits": hits, "vendor_keywords": dict(vendor_counts),
                               "instruction": "Contextually review; triage is not an automatic deletion decision."}
                    line = orjson.dumps(payload) + b"\n"
                    reviewer.write(line)
                    hashes[1].update(line)
                    counts["triage_candidates"] += 1
                if vendor_counts or hits:
                    counts["identity_quarantined_rows"] += 1
                    counts["identity_quarantined_tokens"] += count
                    continue
                line = orjson.dumps(record) + b"\n"
                writer.write(line)
                hashes[0].update(line)
                counts["eligible_rows"] += 1
                counts["eligible_tokens"] += count
    os.replace(temporary, out)
    os.replace(candidate_tmp, candidates)
    report = {"source": str(raw), "language": lang, "input_signature": identity,
              "source_sha256": upstream.sha256_file(raw), "output": str(out),
              "output_sha256": hashes[0].hexdigest(), "candidates": str(candidates),
              "candidates_sha256": hashes[1].hexdigest(), "counts": dict(counts),
              "vendor_keyword_hits": dict(keywords), "seconds": time.monotonic() - started}
    dump(report_path, report)
    return report


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    prior = WORK / "prior.blake2b128"
    metadata = WORK / "prior.report.json"
    inputs = [ROOT / name for name in upstream.EXISTING_CORPORA if (ROOT / name).exists()]
    manifest = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in inputs]
    if metadata.exists():
        report = json.loads(metadata.read_text(encoding="utf-8"))
        if report["manifest"] != manifest or upstream.sha256_file(prior) != report["sha256"]:
            raise ValueError("Prior-corpus fingerprint cache changed")
    else:
        temp = Path(str(prior) + ".tmp")
        sources = []
        with temp.open("wb") as output:
            for path in inputs:
                cached = ROOT / "dataset/coder_pretrain_12b_seg2_novel_raw/baseline_texts.blake2b128"
                if path.name == "pretrain_coder_12b.jsonl" and cached.exists():
                    from scripts.data_builder.coder_novelty import ExclusionIndex
                    index = ExclusionIndex(cached)
                    with cached.open("rb") as stream:
                        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
                            output.write(block)
                    sources.append({"path": str(path), "rows": index.report["source_rows"],
                                    "sha256": index.report["source_sha256"],
                                    "verified_index_reused": str(cached)})
                    print(f"[prior] reused verified index for {path.name}", flush=True)
                    continue
                h, n = hashlib.sha256(), 0
                with path.open("rb") as stream:
                    for line in stream:
                        h.update(line)
                        row = orjson.loads(line)
                        if isinstance(row.get("text"), str) and row["text"].strip():
                            output.write(fp(row["text"]))
                        n += 1
                        if n % 2_000_000 == 0:
                            print(f"[prior] {path.name} {n:,} rows", flush=True)
                sources.append({"path": str(path), "rows": n, "sha256": h.hexdigest()})
                print(f"[prior] indexed {path.name}: {n:,}", flush=True)
        values = np.unique(np.fromfile(temp, dtype="V16"))
        values.tofile(temp)
        os.replace(temp, prior)
        dump(metadata, {"manifest": manifest, "sources": sources,
                        "fingerprints": len(values), "sha256": upstream.sha256_file(prior)})
    benchmark_path = WORK / "benchmark.pkl"
    if not benchmark_path.exists():
        benchmarks, sources = upstream.benchmark_ngrams()
        with benchmark_path.open("wb") as stream:
            pickle.dump(benchmarks, stream)
        dump(WORK / "benchmark.report.json", {"sources": sources, "ngrams": len(benchmarks),
                                              "sha256": upstream.sha256_file(benchmark_path)})
    return hashlib.sha256((upstream.sha256_file(prior) + upstream.sha256_file(benchmark_path)
                           + upstream.sha256_file(ROOT / "model/tokenizer.json")
                           + upstream.sha256_file(Path(__file__))).encode()).hexdigest()


def filter_all(workers):
    signature = prepare()
    plan = json.loads(upstream.PLAN_PATH.read_text(encoding="utf-8"))
    tasks = [(lang, shard["path"], shard["size"], signature)
             for lang, value in plan["languages"].items() for shard in value["chosen_shards"]]
    planned_paths = {task[1] for task in tasks}
    # Include already-downloaded probe shards as well; do not redownload data.
    for raw in sorted(upstream.RAW.rglob("*.parquet")):
        relative = raw.relative_to(upstream.RAW).as_posix()
        if relative not in planned_paths and raw.parent.name in WEIGHTS:
            tasks.append((raw.parent.name, relative, raw.stat().st_size, signature))
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        for future in as_completed([pool.submit(process_shard, task) for task in tasks]):
            result = future.result()
            results.append(result)
            tokens = sum(r["counts"].get("eligible_tokens", 0) for r in results)
            print(f"[filter] {len(results)}/{len(tasks)} shards; {tokens:,} eligible tokens; "
                  f"{Path(result['source']).name}", flush=True)
            dump(WORK / "status.json", {"phase": "filter", "completed_shards": len(results),
                                       "total_shards": len(tasks), "eligible_tokens": tokens})
    results.sort(key=lambda r: r["source"])
    dump(WORK / "filtered_manifest.json", {"policy": POLICY, "signature": signature, "shards": results})
    print("[filter] complete; review candidates before assemble", flush=True)


def quotas_for(available):
    if sum(available.values()) < TARGET:
        raise ValueError(f"Only {sum(available.values()):,} available tokens; need {TARGET:,}")
    quotas = {lang: min(round(TARGET * WEIGHTS[lang]), available[lang]) for lang in WEIGHTS}
    left = TARGET - sum(quotas.values())
    # Prefer the two requested languages whenever either has spare capacity.
    for group in (("py", "cpp"), tuple(k for k in WEIGHTS if k not in ("py", "cpp"))):
        for lang in sorted(group, key=lambda k: available[k] - quotas[k], reverse=True):
            add = min(left, available[lang] - quotas[lang])
            quotas[lang] += add
            left -= add
    if (quotas["py"] + quotas["cpp"]) / TARGET < .65:
        raise ValueError("Downloaded eligible Python/C++ cannot meet a 65% combined share")
    return quotas


def assemble():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("Final artifact already exists; verify it instead of overwriting")
    manifest = json.loads((WORK / "filtered_manifest.json").read_text(encoding="utf-8"))
    decisions_path = WORK / "identity_decisions.jsonl"
    decisions = {}
    if decisions_path.exists():
        for line in decisions_path.open("rb"):
            decision = orjson.loads(line)
            if decision["decision"] not in ("keep", "drop") or not decision.get("reason"):
                raise ValueError("Invalid contextual identity decision")
            decisions[decision["text_id"]] = decision
    available = Counter()
    for item in manifest["shards"]:
        available[item["language"]] += item["counts"]["eligible_tokens"]
    quotas = quotas_for(available)
    print("[assemble] quotas " + json.dumps(quotas), flush=True)
    stats = {lang: Counter() for lang in WEIGHTS}
    rejected, seen_text, seen_exercise = Counter(), set(), set()
    shuffler = ExternalRandomShuffler(OUTPUT, upstream.SEED, 60_000,
                                     chunk_bytes=256 * 1024 ** 2, deduplicate=False)
    def add(row):
        lang, count = row["mix_category"], row["token_count"]
        text_id, exercise_id = bytes.fromhex(row["text_id"]), bytes.fromhex(row["exercise_id"])
        if text_id in seen_text or exercise_id in seen_exercise:
            rejected[lang + ":global_text_or_exercise_duplicate"] += 1
            return
        if stats[lang]["tokens"] + count > quotas[lang]:
            rejected[lang + ":quota"] += 1
            return
        seen_text.add(text_id)
        seen_exercise.add(exercise_id)
        shuffler.add_serialized(orjson.dumps(row))
        stats[lang]["tokens"] += count
        stats[lang]["rows"] += 1
    try:
        for item in manifest["shards"]:
            if upstream.sha256_file(Path(item["output"])) != item["output_sha256"]:
                raise ValueError("Filtered shard checksum changed")
            with Path(item["output"]).open("rb") as stream:
                for line in stream:
                    add(orjson.loads(line))
            with Path(item["candidates"]).open("rb") as stream:
                for line in stream:
                    candidate = orjson.loads(line)
                    if not (candidate["vendor_keywords"] or candidate["identity_hits"]):
                        continue
                    row = candidate["record"]
                    decision = decisions.get(row["text_id"])
                    if decision and decision["decision"] == "keep":
                        add(row)
                    else:
                        rejected["identity_review_drop" if decision else "identity_unreviewed_quarantine"] += 1
            print(f"[assemble] {Path(item['output']).name}: {sum(s['tokens'] for s in stats.values()):,} tokens", flush=True)
        shortfall = {lang: quotas[lang] - value["tokens"] for lang, value in stats.items()
                     if quotas[lang] - value["tokens"] > 4096}
        if shortfall:
            raise ValueError(f"Deduplicated language shortfalls: {shortfall}; extend/adjust quotas before release")
        result = shuffler.finish(progress_label="CPT shuffle")
    except BaseException:
        shuffler.temporary_dir.cleanup()
        raise
    mixing = upstream.audit_contiguous_mixing(OUTPUT, {lang: {"quota": quota} for lang, quota in quotas.items()})
    counts, keywords = Counter(), Counter()
    for item in manifest["shards"]:
        counts.update(item["counts"])
        keywords.update(item["vendor_keyword_hits"])
    report = {"name": "InstinctV1Moe UltraData-Code-L3 Coding CPT",
              "source_repository": upstream.REPO, "source_revision": upstream.REVISION,
              "output": str(OUTPUT), "output_sha256": result["sha256"], "rows": result["rows"],
              "tokens": sum(s["tokens"] for s in stats.values()), "target_tokens": TARGET,
              "tokenizer_sha256": upstream.sha256_file(ROOT / "model/tokenizer.json"),
              "token_count_policy": "actual project tokenizer, add_special_tokens=False, plus BOS/EOS=2",
              "min_tokens": 128, "max_tokens": 4096, "rewritten_rows": 0, "text_truncated": False,
              "components": {k: dict(v) for k, v in stats.items()}, "language_quotas": quotas,
              "python_cpp_share": (stats["py"]["tokens"] + stats["cpp"]["tokens"]) / TARGET,
              "filter_counts": dict(counts), "assembly_rejections": dict(rejected),
              "identity_clean": {"target_identity": "InstinctV1Moe", "vendor_keyword_hits": dict(keywords),
                                 "review_decisions": str(decisions_path),
                                 "unreviewed_mentions": "quarantined, never automatically deleted by triage"},
              "prior_corpus_exclusion": json.loads((WORK / "prior.report.json").read_text(encoding="utf-8")),
              "benchmark_screening": json.loads((WORK / "benchmark.report.json").read_text(encoding="utf-8")),
              "inputs": manifest, "shuffle": {**result, "seed": upstream.SEED, "mixing_audit": mixing},
              "quality_policy": POLICY,
              "limitations": ["Heuristic quality selection; generated solutions, analyses and tests are not execution-verified.",
                              "Python syntax is checked; other languages are not compiler-verified.",
                              "Exact text and task+solution dedup do not establish semantic uniqueness.",
                              "13-word benchmark screening cannot prove absence of semantic contamination.",
                              "Apache-2.0 release terms do not replace original seed-repository licenses."],
              "decision": "quality-filtered CPT artifact; see measured audit and stated limitations"}
    dump(REPORT, report)
    dump(WORK / "status.json", {"phase": "complete", "output": str(OUTPUT), "report": str(REPORT),
                               "rows": result["rows"], "tokens": report["tokens"]})
    print(json.dumps({k: report[k] for k in ("output", "rows", "tokens", "python_cpp_share")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("filter", "assemble"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.command == "filter":
        filter_all(args.workers)
    else:
        assemble()
