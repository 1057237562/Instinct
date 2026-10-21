"""Review, clean, deduplicate, shuffle and compile the collected second segment.

Requires explicit decisions for contextual persona/model-card candidates.
Ordinary factual/code triage hits are preserved under the reviewed policy.
No replay, source mutation, truncation or identity string replacement occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import datasets  # before torch/transformers on Windows
import orjson
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "dataset/coder_pretrain_12b_seg2_novel_raw"
WORK = ROOT / "dataset/coder_pretrain_12b_seg2_review"
sys.path.insert(0, str(WORK / "_deps"))
from scripts.data_builder.coder_novelty import ExclusionIndex, text_fingerprint
from scripts.data_builder.coder_identity_review import review_reasons
from scripts.data_builder.audit_identity_contamination import VENDOR_ANY_RE
from scripts.data_builder.coder_pretrain_common import ngrams, words
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler
from scripts.data_builder.build_coder_pretrain_12b import audit_contiguous_mixing

QUOTAS = {"code": 7_200_000_000, "math": 1_200_000_000, "natural_en": 2_400_000_000, "natural_zh": 1_200_000_000}
OUTPUT = ROOT / "dataset/pretrain_coder_12b_seg2_identity_clean.jsonl"
REPORT = ROOT / "dataset/pretrain_coder_12b_seg2_identity_clean.report.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b"\n")
    os.replace(temporary, path)


def phase(name, **fields):
    save(WORK / "status.json", {"phase": name, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields})
    print(f"[segment2] {name} {fields}", flush=True)


def prepare_review():
    approved_path = WORK / "approved_decisions.json"
    approved = json.loads(approved_path.read_text(encoding="utf-8"))["decisions"]
    output = WORK / "review_decisions.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    counts = Counter(); missing = {}
    sources = []
    with temporary.open("wb") as f:
        for component in QUOTAS:
            for p in sorted((RAW / component / "review_candidates").glob("*.jsonl")):
                rows = input_rows = 0; digest = hashlib.sha256()
                for line in p.open("rb"):
                    digest.update(line); input_rows += 1; row = orjson.loads(line); text = row["record"]["text"]
                    key = text_fingerprint(text).hex(); reasons = review_reasons(text, component)
                    decision = approved.get(key)
                    if reasons and decision is None:
                        missing[key] = {"id": key, "component": component, "reasons": reasons, "record": row["record"]}
                        continue
                    if decision is None:
                        decision = {"decision": "keep", "reason": "Reviewed preservation policy: factual/provider knowledge, author/character names, library identifiers and ordinary code TODOs are not assistant identity claims."}
                    if decision["decision"] not in ("keep", "drop") or not decision.get("reason"):
                        raise ValueError(f"Invalid approved decision: {key}")
                    keyword_hits = Counter(m.group(0).lower() for m in VENDOR_ANY_RE.finditer(text)) if "foreign_model_or_provider_mention" in row["_review"]["rules"] else Counter()
                    f.write(orjson.dumps({"id": key, **decision, "component": component, "keyword_hits": dict(keyword_hits),
                                          "triage_rules": row["_review"]["rules"], "source": row["_review"]["source"], "source_row": row["_review"]["row_number"],
                                          "contextual_override": key in approved}, option=orjson.OPT_APPEND_NEWLINE))
                    counts[decision["decision"]] += 1; rows += 1
                collection_audit = json.loads((RAW / component / "reports" / p.with_suffix(".json").name).read_text(encoding="utf-8"))
                if digest.hexdigest() != collection_audit["identity_review"]["candidate_sha256"] or input_rows != collection_audit["identity_review"]["candidate_rows"]:
                    raise ValueError(f"Triage candidates changed after collection: {p}")
                sources.append({"path": str(p), "sha256": digest.hexdigest(), "reviewed_rows": rows})
            print(f"[review] {component}: {sum(counts.values()):,} rows", flush=True)
    if missing:
        with (WORK / "missing_decisions.jsonl").open("wb") as f:
            for row in missing.values():f.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
        raise ValueError(f"{len(missing)} contextual decisions are still required")
    os.replace(temporary, output)
    result = {"method": "contextual policy review plus fingerprint-bound explicit persona/model-card decisions", "counts": dict(counts),
              "sources": sources, "decisions": str(output), "decisions_sha256": sha(output), "approved_decisions_sha256": sha(approved_path),
              "review_policy_sha256": sha(ROOT / "scripts/data_builder/coder_identity_review.py"),
              "raw_preserved": True, "blind_replacement": False}
    save(WORK / "review.report.json", result)
    return result


def load_decisions():
    return {bytes.fromhex(row["id"]): row for row in (orjson.loads(line) for line in (WORK / "review_decisions.jsonl").open("rb"))}


_DECISIONS = None
_REVIEW_SHA = None
_REVIEW_SIGNATURE = None
def configure_logs():
    (WORK / "logs").mkdir(exist_ok=True)
    sys.stdout = (WORK / "logs/finalize.log").open("a", encoding="utf-8", buffering=1)
    sys.stderr = (WORK / "logs/finalize.stderr.log").open("a", encoding="utf-8", buffering=1)


def clean_shard(job):
    global _DECISIONS, _REVIEW_SHA, _REVIEW_SIGNATURE
    decision_path = WORK / "review_decisions.jsonl"
    if _DECISIONS is None:
        _DECISIONS = load_decisions();_REVIEW_SHA = sha(decision_path)
        stat = decision_path.stat();_REVIEW_SIGNATURE = (stat.st_size, stat.st_mtime_ns)
    stat = decision_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != _REVIEW_SIGNATURE:raise ValueError("Review decisions changed during cleaning")
    component, audit_path = job
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8")); source = Path(audit["output"])
    if not source.resolve().is_relative_to((RAW / component / "shards").resolve()):raise ValueError("Invalid raw source boundary")
    output = WORK / "identity_clean_shards" / component / (source.stem + "_identity_clean.jsonl")
    report_path = output.with_suffix(".report.json")
    review_sha = _REVIEW_SHA
    if report_path.exists() and output.exists():
        r = json.loads(report_path.read_text(encoding="utf-8"))
        if r["input_sha256"] == audit["sha256"] and r["review_sha256"] == review_sha and sha(output) == r["output_sha256"]:return r
        raise ValueError(f"Existing clean shard has a changed audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    source_hash = hashlib.sha256(); output_hash = hashlib.sha256(); counts = Counter(); tokens = 0; keyword_hits = Counter()
    with source.open("rb") as inp, temporary.open("wb") as out:
        for line in inp:
            source_hash.update(line); row = orjson.loads(line); counts["input_rows"] += 1
            text = row.get("text"); count = row.get("token_count")
            if not isinstance(text, str) or not text.strip() or type(count) is not int:
                counts["schema_dropped"] += 1;continue
            if not 64 <= count <= 4096:counts["length_dropped"] += 1;continue
            decision = _DECISIONS.get(text_fingerprint(text))
            if decision is not None and decision["decision"] == "drop":counts["identity_dropped"] += 1;continue
            if decision is not None:keyword_hits.update(decision["keyword_hits"])
            out.write(line);output_hash.update(line);counts["output_rows"] += 1;tokens += count
        out.flush();os.fsync(out.fileno())
    if source_hash.hexdigest() != audit["sha256"] or counts["input_rows"] != audit["rows"]:
        raise ValueError(f"Raw shard failed its collection audit: {source}")
    os.replace(temporary, output)
    result = {"component": component, "input": str(source), "input_sha256": source_hash.hexdigest(), "output": str(output),
              "output_sha256": output_hash.hexdigest(), "review_sha256": review_sha, "counts": dict(counts), "tokens": tokens,
              "identity_keyword_hit_counts": dict(keyword_hits), "rewritten_rows": 0, "source_repo": audit["upstream_repo"],
              "source_revision": audit["upstream_revision"], "upstream_file": audit["upstream_file"], "training_ready": False}
    save(report_path, result)
    print(f"[clean] {source.name}: {counts['output_rows']:,} kept, {counts['identity_dropped']} identity drops", flush=True)
    return result


def benchmark_matcher():
    import ahocorasick
    fingerprints = set(); sources = []
    for name, path, compressed in [("HumanEval", ROOT / "dataset/humaneval/HumanEval.jsonl.gz", True), ("GSM8K-test", ROOT / "dataset/gsm8k/test.jsonl", False)]:
        opener = gzip.open if compressed else open
        rows = 0
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line);rows += 1
                text = "\n".join(str(v) for v in row.values()) if name == "HumanEval" else str(row.get("question") or "")
                fingerprints.update(ngrams(text))
        sources.append({"name": name, "path": str(path), "rows": rows, "sha256": sha(path)})
    repo = "google-research-datasets/mbpp"; revision = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
    path = WORK / "benchmarks/mbpp_sanitized_test.parquet"; path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/sanitized/test-00000-of-00001.parquet?download=true", timeout=90) as response:
            data = response.read()
        temporary = path.with_suffix(".parquet.tmp");temporary.write_bytes(data);os.replace(temporary, path)
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 257:raise ValueError("Unexpected MBPP sanitized-test size")
    for row in rows:fingerprints.update(ngrams("\n".join([str(row.get("prompt") or ""), str(row.get("code") or ""), "\n".join(row.get("test_list") or [])])))
    sources.append({"name": "MBPP-sanitized-test", "repo": repo, "revision": revision, "path": str(path), "rows": len(rows), "sha256": sha(path)})
    automaton = ahocorasick.Automaton()
    for index, fingerprint in enumerate(fingerprints):automaton.add_word("\x1f" + "\x1f".join(fingerprint) + "\x1f", index)
    automaton.make_automaton()
    return automaton, sources


def benchmark_overlap(text, matcher):
    normalized = "\x1f" + "\x1f".join(words(text)) + "\x1f"
    return next(matcher.iter(normalized), None) is not None


def assemble(clean_reports, review):
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError("Final output already exists")
    prior = ExclusionIndex(RAW / "baseline_texts.blake2b128")
    decisions = load_decisions();matcher, benchmarks = benchmark_matcher()
    shuffler = ExternalRandomShuffler(OUTPUT, 20261011, 100_000, deduplicate=False, chunk_bytes=512 * 1024**2)
    counts = Counter(); totals = Counter(); rows = Counter(); keywords = Counter(); licenses = Counter(); seen = set()
    scans = 0
    for component, target in QUOTAS.items():
        for report in [r for r in clean_reports if r["component"] == component]:
            for line in Path(report["output"]).open("rb"):
                scans += 1
                if totals[component] >= target - 4096:break
                row = orjson.loads(line);text = row["text"];count = row["token_count"];fp = text_fingerprint(text)
                if fp in seen:counts["duplicate_text"] += 1;continue
                if prior.contains(text):counts["prior_corpus_text"] += 1;continue
                if benchmark_overlap(text, matcher):counts["benchmark_overlap"] += 1;continue
                if totals[component] + count > target:counts["over_budget"] += 1;continue
                row["mix_category"] = "natural" if component.startswith("natural_") else component
                row["mixture"] = "deepseek-coder-v2-60-code-10-math-30-natural"
                shuffler.add_serialized(orjson.dumps(row));seen.add(fp)
                totals[component] += count;rows[component] += 1;licenses[row.get("license") or "unspecified"] += 1
                if fp in decisions:keywords.update(decisions[fp]["keyword_hits"])
                if sum(rows.values()) % 250_000 == 0:phase("assembling", selected_rows=sum(rows.values()), tokens=dict(totals))
            if totals[component] >= target - 4096:break
        if totals[component] < target - 4096:raise ValueError(f"Insufficient cleaned data for {component}: {totals[component]}/{target}")
    phase("shuffling", selected_rows=sum(rows.values()), tokens=dict(totals))
    result = shuffler.finish(progress_label="Segment2 shuffle")
    mixing = audit_contiguous_mixing(OUTPUT)
    report = {"name": "InstinctV1Moe second segment identity_clean", "segment": 2, "output": str(OUTPUT),
              "output_sha256": result["sha256"], "rows": result["rows"], "tokens": sum(totals.values()), "component_tokens": dict(totals),
              "component_rows": dict(rows), "targets": QUOTAS, "replay_tokens": 0, "whole_records_only": True,
              "max_tokens_including_bos_eos": 4096, "text_truncated": False, "rewritten_rows": 0,
              "identity_review": review, "identity_keyword_hit_counts": dict(keywords), "input_clean_shards": clean_reports,
              "identity_dropped_rows": sum(r["counts"].get("identity_dropped", 0) for r in clean_reports),
              "schema_dropped_rows": sum(r["counts"].get("schema_dropped", 0) for r in clean_reports),
              "length_dropped_rows": sum(r["counts"].get("length_dropped", 0) for r in clean_reports),
              "rejected": dict(counts), "prior_corpus_sha256": prior.report["source_sha256"],
              "global_exact_text_deduplication": True, "benchmark_screening": {"method": "exact 13-word n-gram Aho-Corasick matching", "sources": benchmarks},
              "shuffle": {**result, "mixing_audit": mixing}, "license_row_counts": dict(licenses), "training_ready": True,
              "limitations": ["Exact text/13-word filtering cannot establish semantic novelty or eliminate paraphrased benchmark overlap.",
                              "Identity cleaning uses a reviewed preservation policy and explicit contextual decisions, not exhaustive human semantic review of every row.",
                              "Token counts are inherited from SHA-256-verified collector outputs whose texts were tokenized and are not rewritten."]}
    save(REPORT, report)
    return report


def compile_release(report):
    compiler = ROOT / "dataset_compiler/target/release/dataset_compiler.exe"
    output = OUTPUT.with_suffix(".parquet")
    phase("compiling_parquet", jsonl=str(OUTPUT))
    subprocess.run([str(compiler), "compile", "-i", str(OUTPUT), "-o", str(output), "--align-chunk-bytes", "1GiB"], cwd=ROOT, check=True, stdout=sys.stdout, stderr=sys.stderr)
    phase("verifying_parquet", parquet=str(output))
    subprocess.run([str(compiler), "verify", str(OUTPUT), str(output), "--rows", str(report["rows"])], cwd=ROOT, check=True, stdout=sys.stdout, stderr=sys.stderr)
    parquet_report = output.with_suffix(".parquet.report.json")
    compiled = json.loads(parquet_report.read_text(encoding="utf-8"))
    compiled.update(training_ready=True, identity_review=report["identity_review"], identity_keyword_hit_counts=report["identity_keyword_hit_counts"],
                    cleaned_jsonl_report=str(REPORT), cleaned_jsonl_sha256=report["output_sha256"], replay_tokens=0,
                    verification={"rows_compared": report["rows"], "full_row_value_comparison": True})
    save(parquet_report, compiled)
    phase("complete", jsonl=str(OUTPUT), parquet=str(output), rows=report["rows"], tokens=report["tokens"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-review", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-to-files", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if args.log_to_files:configure_logs()
    from scripts.data_builder.start_coder_novel_12b import acquire_run_lock
    lock = acquire_run_lock(WORK)
    if args.prepare_review:
        phase("reviewing");prepare_review();return
    phase("reviewing")
    review_path = WORK / "review.report.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
    if review is None or review["approved_decisions_sha256"] != sha(WORK / "approved_decisions.json") or review.get("review_policy_sha256") != sha(ROOT / "scripts/data_builder/coder_identity_review.py") or review["decisions_sha256"] != sha(WORK / "review_decisions.jsonl"):
        review = prepare_review()
    elif any(sha(source["path"]) != source["sha256"] for source in review["sources"]):
        raise ValueError("Triage inputs changed after contextual review")
    phase("cleaning")
    jobs = [(c, str(p)) for c in QUOTAS for p in sorted((RAW / c / "reports").glob("*.json"))]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_logs if args.log_to_files else None) as pool:
        for future in as_completed([pool.submit(clean_shard, job) for job in jobs]):
            results.append(future.result());phase("cleaning", completed_shards=len(results), total_shards=len(jobs))
    results.sort(key=lambda r:(list(QUOTAS).index(r["component"]),r["input"]))
    report = assemble(results, review)
    compile_release(report)


if __name__ == "__main__":
    try:main()
    except BaseException as error:
        phase("failed", error=str(error));raise
