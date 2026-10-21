"""Recover demonstrated false positives without replacing v1 artifacts.

Keep the <=4096 cap. Remove low-information/benchmark-common 13-grams from
automatic exclusion, while retaining distinctive benchmark names and 30-word
prompt matches. Accept intentional class interfaces with functional concrete
implementations; still reject genuinely unimplemented standalone functions.
All recovered records retain source coordinates and fingerprint-bound reasons.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scripts.data_builder import clean_ultradata_l3_cpt as clean
from scripts.data_builder.coder_pretrain_common import ngrams, words
from scripts.data_builder.filter_anomaly_candidates import inspect_row
from datasets import load_dataset
import numpy as np
import orjson
import pyarrow.parquet as pq

WORK = clean.WORK
POLICY = "l3-quality-v2-bias-correction"
LOW = STRICT = PROMPTS = NAMES = None


def low_information(gram):
    return all(w.isdecimal() for w in gram) or all(len(w) == 1 and w.isascii() and w.isalpha() for w in gram) or len(set(gram)) <= 3


def intentional_interface(row):
    tree = ast.parse(row["solution"])
    top_functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    def stub(n):
        body = n.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Raise):
            return False
        exc = body[0].exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    if any(stub(n) for n in top_functions):
        return False
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    methods = [n for c in classes for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    concrete = [n for n in methods if not stub(n) and n.name != "__init__" and
                any(isinstance(child, (ast.Return, ast.For, ast.While, ast.If)) for child in ast.walk(n))]
    stated_interface = re.search(r"abstract|interface|subclass|overrid|base class|notimplementederror|not supported|unsupported", row["task"], re.I)
    return bool(classes and (concrete or stated_interface))


def build_policy():
    path = WORK / "benchmark_v2.pkl"
    if path.exists():
        return
    frequency, prompts, names = Counter(), set(), set()
    with gzip.open(ROOT / "dataset/humaneval/HumanEval.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            frequency.update(ngrams("\n".join(str(v) for v in row.values())))
            prompts.update(ngrams(row["prompt"], 30))
            name = row.get("entry_point", "")
            if len(name) >= 10 and "_" in name:
                names.add(name.lower())
    with (ROOT / "dataset/gsm8k/test.jsonl").open(encoding="utf-8") as source:
        for line in source:
            question = json.loads(line)["question"]
            frequency.update(ngrams(question))
            prompts.update(ngrams(question, 30))
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for row in mbpp:
        frequency.update(ngrams("\n".join([str(row.get("prompt") or ""), str(row.get("code") or ""), "\n".join(row.get("test_list") or [])])))
        prompts.update(ngrams(row.get("prompt") or "", 30))
        try:
            for node in ast.parse(row.get("code") or "").body:
                if isinstance(node, ast.FunctionDef) and len(node.name) >= 10 and "_" in node.name:
                    names.add(node.name.lower())
        except SyntaxError:
            pass
    with (WORK / "benchmark.pkl").open("rb") as stream:
        original = pickle.load(stream)
    removed = {gram for gram in original if low_information(gram) or frequency[gram] >= 3}
    strict = original - removed
    with path.open("wb") as output:
        pickle.dump((removed, strict, prompts, names), output)
    clean.dump(WORK / "benchmark_v2.report.json", {"policy": POLICY, "original_13grams": len(original),
               "exempted_low_information_or_common_13grams": len(removed), "remaining_13grams": len(strict),
               "prompt_30grams": len(prompts), "distinctive_benchmark_names": sorted(names),
               "common_means": "13-gram present in at least three distinct held-out records",
               "limitations": "Lexical signatures remain conservative and do not prove semantic contamination or its absence"})


def init():
    global LOW, STRICT, PROMPTS, NAMES
    clean.init_worker()
    with (WORK / "benchmark_v2.pkl").open("rb") as stream:
        LOW, STRICT, PROMPTS, NAMES = pickle.load(stream)


def distinctive_overlap(row, tokens):
    if any(tuple(tokens[i:i+13]) in STRICT for i in range(len(tokens)-12)):
        return True
    task_words = words(row["task"])
    if any(name in NAMES for name in task_words):
        return True
    return any(tuple(task_words[i:i+30]) in PROMPTS for i in range(len(task_words)-29))


def recover(path):
    lang = path.parent.name
    stem = WORK / (path.stem + ".rescued")
    out, candidates, ledger = Path(str(stem) + ".jsonl"), Path(str(stem) + ".candidates.jsonl"), Path(str(stem) + ".decisions.jsonl")
    report_path = Path(str(stem) + ".report.json")
    if report_path.exists():
        r = json.loads(report_path.read_text(encoding="utf-8"))
        if r["policy"] != POLICY:
            raise ValueError("Changed rescue policy")
        raw_signature = path.stat()
        original = json.loads((WORK / (path.stem + ".report.json")).read_text(encoding="utf-8"))
        if (raw_signature.st_size, raw_signature.st_mtime_ns) != (original["input_signature"]["bytes"], original["input_signature"]["mtime_ns"]) or r["source_sha256"] != original["source_sha256"]:
            raise ValueError("Raw source changed since bias correction")
        return r
    signature = path.stat()
    counts, keywords = Counter({"eligible_rows": 0, "eligible_tokens": 0}), Counter()
    hashes = [hashlib.sha256(), hashlib.sha256()]
    row_number = 0
    with out.open("wb") as writer, candidates.open("wb") as reviewer, ledger.open("wb") as decisions:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["uuid", "task", "analysis", "solution", "test", "full_content"]):
            for row in batch.to_pylist():
                row_number += 1
                text = row.get("full_content")
                if not isinstance(text, str):
                    continue
                # Cheap prefilter for class stubs; benchmark rescue is screened
                # with just the exempted signatures before tokenization.
                possible_stub = lang == "py" and "NotImplementedError" in (row.get("solution") or "")
                tokens = words(text)
                has_exempted = any(tuple(tokens[i:i+13]) in LOW for i in range(len(tokens)-12))
                if not (possible_stub or has_exempted):
                    continue
                old_reason = clean.quality_reason(row, lang)
                reasons = []
                if old_reason:
                    if old_reason != "python_unimplemented_function" or not intentional_interface(row):
                        continue
                    reasons.append("intentional_class_interface_or_unsupported_method")
                elif not has_exempted:
                    continue
                digest = clean.fp(text)
                position = int(np.searchsorted(clean._CROSS, np.void(digest)))
                if position < len(clean._CROSS) and clean._CROSS[position] == np.void(digest):
                    continue
                count = len(clean._TOKENIZER.encode(text, add_special_tokens=False).ids) + 2
                if not 128 <= count <= 4096:
                    continue
                old_overlap = any(tuple(tokens[i:i+13]) in clean._BENCH for i in range(len(tokens)-12))
                if old_overlap:
                    if distinctive_overlap(row, tokens):
                        continue
                    reasons.append("benchmark_hit_only_low_information_or_common_template")
                if not old_reason and not old_overlap:
                    continue
                relative = path.relative_to(clean.upstream.RAW).as_posix()
                record = {"text": text, "token_count": count, "source": "ultradata_code_l3:" + lang,
                          "mix_category": lang, "mixture": "ultradata-l3-python-cpp-cpt", "license": "apache-2.0",
                          "uuid": row["uuid"], "source_shard": relative, "source_row": row_number,
                          "source_revision": clean.upstream.REVISION, "text_id": digest.hex(),
                          "exercise_id": clean.fp(row["task"].strip() + "\n\x1f\n" + row["solution"].strip()).hex()}
                decision = {"text_id": digest.hex(), "decision": "recover", "reasons": reasons,
                            "source_shard": relative, "source_row": row_number, "policy": POLICY}
                decisions.write(orjson.dumps(decision) + b"\n")
                counts["recovered_rows"] += 1
                for reason in reasons:
                    counts["recover:" + reason] += 1
                # Subtract the ORIGINAL first rejection only, preserving reconciliation.
                counts["drop:" + (old_reason or "benchmark_13gram")] -= 1
                triage = inspect_row({"text": text, "solution": row["solution"]}, "all")
                verdict, hits, vendor = clean.upstream.identity_scan(text)
                keywords.update(vendor)
                if triage or vendor or hits:
                    line = orjson.dumps({"record": record, "triage": triage, "identity_verdict": verdict,
                                         "identity_hits": hits, "vendor_keywords": dict(vendor),
                                         "instruction": "Contextual review of recovered record; triage never automatically deletes"}) + b"\n"
                    reviewer.write(line)
                    hashes[1].update(line)
                if vendor or hits:
                    counts["identity_quarantined_rows"] += 1
                    counts["identity_quarantined_tokens"] += count
                    continue
                line = orjson.dumps(record) + b"\n"
                writer.write(line)
                hashes[0].update(line)
                counts["eligible_rows"] += 1
                counts["eligible_tokens"] += count
    after = path.stat()
    if (signature.st_size, signature.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Raw source changed during rescue")
    original_report = WORK / (path.stem + ".report.json")
    source_hash = (json.loads(original_report.read_text(encoding="utf-8"))["source_sha256"]
                   if original_report.exists() else clean.upstream.sha256_file(path))
    report = {"kind": "bias_correction_supplement", "policy": POLICY, "source": str(path), "language": lang,
              "source_sha256": source_hash, "output": str(out), "output_sha256": hashes[0].hexdigest(),
              "candidates": str(candidates), "candidates_sha256": hashes[1].hexdigest(),
              "counts": dict(counts), "vendor_keyword_hits": dict(keywords), "recovery_decisions": str(ledger)}
    clean.dump(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    build_policy()
    paths = sorted(clean.upstream.RAW.rglob("*.parquet"))
    results, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init) as pool:
        for future in as_completed([pool.submit(recover, path) for path in paths]):
            r = future.result()
            results.append(r)
            counts.update(r["counts"])
            print(f"[rescue] {len(results)}/{len(paths)}; recovered {counts['recovered_rows']:,} records", flush=True)
            clean.dump(WORK / "rescue_status.json", {"completed_shards": len(results), "total_shards": len(paths), "counts": dict(counts)})
    results.sort(key=lambda r: r["source"])
    clean.dump(WORK / "rescue_manifest.json", {"policy": POLICY, "shards": results, "counts": dict(counts)})
