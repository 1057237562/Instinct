"""Complete missing CPT source caches without rebuilding recovered components."""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import scripts.data_builder.collect_continue_pretrain_1b as base

CACHE = ROOT / "dataset/continue_pretrain_1b/filtered_cache"
PRIOR = ROOT / "dataset/pretrain_codespecialist.jsonl"
CODE_TARGET = 100_795_000
# The actually trained prior file contains only 101,923 clean, whole-record
# academic tokens at <=4096. Keep all of them and move the unavailable
# technical quota into code replay instead of truncating long papers.
REPLAY_TARGETS = {"replay_code": 119_898_077, "replay_general": 60_000_000,
                  "replay_math": 20_000_000, "replay_technical": 101_923}


def load_cache(source):
    path = CACHE / f"{source}.jsonl"; seen = set(); stats = Counter(); rows = []
    if path.exists():
        with path.open("r+b") as f:
            while True:
                line_start = f.tell(); line = f.readline()
                if not line: break
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError:
                    # An interrupted append can only be recovered safely when
                    # the malformed data is the unterminated final line.
                    if f.tell() == path.stat().st_size and not line.endswith(b"\n"):
                        f.truncate(line_start); break
                    raise
                seen.add(base.h16(row["text"])); stats["rows"] += 1
                stats["tokens"] += int(row["token_count"]); stats["bytes"] += len(line); rows.append(row)
    return path, seen, stats, rows


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True).backend_tokenizer
    he_grams, gsm_grams = base.benchmark_sets()
    benchmark_grams = he_grams | gsm_grams
    print("Indexing prior exact texts", flush=True)
    prior_hashes = set()
    with PRIOR.open("rb") as f:
        for line in f:
            row = orjson.loads(line)
            if isinstance(row.get("text"), str): prior_hashes.add(base.h16(row["text"]))

    # Finish code CoT from its recovered 39.2M cache.
    code_path, code_seen, code_stats, code_rows = load_cache("open_code_cot_novel")
    question_counts = Counter()
    for row in code_rows:
        source_id = str(row.get("source_id") or "")
        qid, _, variant = source_id.rpartition(":")
        question_counts[qid] = max(question_counts[qid], int(variant or 0) + 1)
    del code_rows
    with code_path.open("ab") as destination:
        # Rescan from shard zero on resume. Exact-text and per-question guards
        # make this idempotent and also allow a missing cache to be rebuilt
        # entirely from the downloaded parquet files.
        for shard in range(30):
            if code_stats["tokens"] >= CODE_TARGET - base.MAX_TOKENS: break
            filename = f"split_0/train-{shard:05d}-of-00030.parquet"
            path = base.download("open_code", filename, [])
            columns = ["id", "input", "output", "solution", "source", "license", "dataset", "split", "difficulty"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
                for row in batch.to_pylist():
                    code_stats["scanned_resume"] += 1
                    if str(row.get("split") or "").lower() != "train": continue
                    qid = str(row.get("id") or "")
                    if question_counts[qid] >= 16: continue
                    problem, reasoning, solution = row.get("input"), row.get("output"), row.get("solution")
                    if not base.clean(problem, 80, 30000) or not base.clean(reasoning, 200, 50000) or not base.clean(solution, 30, 30000): continue
                    try: ast.parse(solution)
                    except SyntaxError: continue
                    text = ("### Programming problem\n" + problem.strip()
                            + "\n\n### Reasoning\n" + reasoning.strip()
                            + "\n\n### Verified Python solution\n```python\n"
                            + solution.strip() + "\n```")
                    fp = base.h16(text)
                    combined = problem + "\n" + reasoning + "\n" + solution
                    if fp in code_seen or fp in prior_hashes or base.has_overlap(combined, benchmark_grams): continue
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
                    if not 64 <= tokens <= base.MAX_TOKENS or code_stats["tokens"] + tokens > CODE_TARGET: continue
                    output = {"text": text, "source": "open_code_cot_novel",
                              "source_id": f"{qid}:{question_counts[qid]}",
                              "license": str(row.get("license") or "CC-BY-4.0"),
                              "url": f"https://huggingface.co/datasets/{base.REPOS['open_code'][0]}",
                              "token_count": tokens,
                              "metadata": {"source": row.get("source"), "dataset": row.get("dataset"), "difficulty": row.get("difficulty")}}
                    line = orjson.dumps(output) + b"\n"; destination.write(line); code_seen.add(fp)
                    question_counts[qid] += 1; code_stats["rows"] += 1; code_stats["tokens"] += tokens; code_stats["bytes"] += len(line)
                    if code_stats["tokens"] >= CODE_TARGET - base.MAX_TOKENS: break
                if code_stats["tokens"] >= CODE_TARGET - base.MAX_TOKENS: break
            print(f"code cache: {code_stats['tokens']:,}/{CODE_TARGET:,}", flush=True)

    # Let math absorb every whole-record shortfall from the fixed components.
    # This also keeps the novel total at 800M if OpenCodeReasoning exhausts its
    # available shards a little below its nominal 130M target.
    recovered = json.loads((CACHE / "report.json").read_text(encoding="utf-8"))
    fixed_tokens = sum(int(recovered[name]["tokens"]) for name in (
        "python_edu_novel", "cosmopedia_novel", "verified_math_code"
    ))
    math_target = base.NEW_TARGET - fixed_tokens - int(code_stats["tokens"])
    math_path, math_seen, math_stats, math_rows = load_cache("verified_math_cot_novel")
    problem_counts = Counter()
    for cached in math_rows:
        text = cached["text"]
        problem = text.split("\n\n### Verified reasoning\n", 1)[0].removeprefix("### Math problem\n")
        problem_counts[base.h16(problem)] += 1
    del math_rows
    with math_path.open("ab") as destination:
        for shard in range(10):
            if math_stats["tokens"] >= math_target - base.MAX_TOKENS: break
            filename = f"data/train-{shard:05d}-of-00010.parquet"
            path = base.download("open_r1_math", filename, [])
            columns = ["problem", "solution", "answer", "source", "uuid", "is_reasoning_complete", "correctness_count"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
                for row in batch.to_pylist():
                    math_stats["openr1_scanned"] += 1; problem, solution = row.get("problem"), row.get("solution")
                    if int(row.get("correctness_count") or 0) < 1 or not any(row.get("is_reasoning_complete") or []): continue
                    if not base.clean(problem, 40, 20000) or not base.clean(solution, 200, 50000) or "\\boxed" not in solution: continue
                    ph = base.h16(problem)
                    if problem_counts[ph] >= 1: continue
                    combined = problem + "\n" + solution
                    if base.has_overlap(combined, benchmark_grams): continue
                    text = "### Math problem\n" + problem.strip() + "\n\n### Verified reasoning\n" + solution.strip(); fp = base.h16(text)
                    if fp in math_seen or fp in prior_hashes: continue
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
                    if not 64 <= tokens <= base.MAX_TOKENS or math_stats["tokens"] + tokens > math_target: continue
                    output = {"text": text, "source": "verified_math_cot_novel", "source_id": str(row.get("uuid")),
                              "license": "Apache-2.0", "url": f"https://huggingface.co/datasets/{base.REPOS['open_r1_math'][0]}",
                              "token_count": tokens, "metadata": {"upstream_source": row.get("source"), "verification": "Math-Verify or upstream judge"}}
                    line = orjson.dumps(output) + b"\n"; destination.write(line); math_seen.add(fp)
                    problem_counts[ph] += 1; math_stats["rows"] += 1; math_stats["tokens"] += tokens; math_stats["bytes"] += len(line)
            print(f"OpenR1 math cache: {math_stats['tokens']:,}/{math_target:,}", flush=True)

        for shard in range(144):
            if math_stats["tokens"] >= math_target - base.MAX_TOKENS: break
            filename = f"data/cot-{shard:05d}-of-00144.parquet"
            path = base.download("open_math", filename, [])
            columns = ["expected_answer", "problem_type", "problem_source", "generation_model", "problem", "generated_solution", "used_in_kaggle"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
                for row in batch.to_pylist():
                    math_stats["openmath_scanned"] += 1
                    problem, solution, answer = row.get("problem"), row.get("generated_solution"), str(row.get("expected_answer") or "").strip()
                    if not answer or not base.clean(problem, 40, 20000) or not base.clean(solution, 200, 50000): continue
                    ph = base.h16(problem)
                    if problem_counts[ph] >= 3: continue
                    boxed = base.last_boxed_answer(solution)
                    if boxed is None or not base.normalize_math_answer(answer) \
                            or base.normalize_math_answer(boxed) != base.normalize_math_answer(answer): continue
                    combined = problem + "\n" + solution
                    if base.has_overlap(combined, benchmark_grams): continue
                    text = "### Math problem\n" + problem.strip() + "\n\n### Verified reasoning\n" + solution.strip(); fp = base.h16(text)
                    if fp in math_seen or fp in prior_hashes: continue
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
                    if not 64 <= tokens <= base.MAX_TOKENS or math_stats["tokens"] + tokens > math_target: continue
                    output = {"text": text, "source": "verified_math_cot_novel", "source_id": f"openmath:{shard}:{math_stats['openmath_scanned']-1}",
                              "license": "CC-BY-4.0", "url": f"https://huggingface.co/datasets/{base.REPOS['open_math'][0]}",
                              "token_count": tokens, "metadata": {"problem_source": row.get("problem_source"), "generation_model": row.get("generation_model"), "expected_answer": answer}}
                    line = orjson.dumps(output) + b"\n"; destination.write(line); math_seen.add(fp)
                    problem_counts[ph] += 1; math_stats["rows"] += 1; math_stats["tokens"] += tokens; math_stats["bytes"] += len(line)
                    if math_stats["tokens"] >= math_target - base.MAX_TOKENS: break
                if math_stats["tokens"] >= math_target - base.MAX_TOKENS: break
            print(f"OpenMath cache: {math_stats['tokens']:,}/{math_target:,}", flush=True)

    # Build stratified replay caches once. Hash bands give deterministic coverage.
    replay = {source: load_cache(source)[:3] for source in REPLAY_TARGETS}
    for band in range(8):
        if all(replay[s][2]["tokens"] >= REPLAY_TARGETS[s] - base.MAX_TOKENS for s in replay): break
        handles = {s: replay[s][0].open("ab") for s in replay}
        try:
            with PRIOR.open("rb") as f:
                for index, line in enumerate(f):
                    row = orjson.loads(line); source = base.REPLAY_MAP.get(row.get("source"))
                    if source not in replay or replay[source][2]["tokens"] >= REPLAY_TARGETS[source] - base.MAX_TOKENS: continue
                    value = int.from_bytes(hashlib.blake2b(f"{base.SEED}/{index}".encode(), digest_size=8).digest(), "big") / 2**64
                    p = base.REPLAY_PROB[source]
                    if not band * p <= value < min(1.0, (band + 1) * p): continue
                    text = row.get("text")
                    if not base.clean(text, 80, 50000): continue
                    fp = base.h16(text)
                    if fp in replay[source][1]: continue
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
                    if not 64 <= tokens <= base.MAX_TOKENS or replay[source][2]["tokens"] + tokens > REPLAY_TARGETS[source]: continue
                    output = {"text": text, "source": source, "source_id": str(index),
                              "license": row.get("license") or "inherited; see pretrain_codespecialist report",
                              "url": row.get("url") or "", "token_count": tokens,
                              "metadata": {"prior_source": row.get("source")}}
                    out = orjson.dumps(output) + b"\n"; handles[source].write(out); replay[source][1].add(fp)
                    replay[source][2]["rows"] += 1; replay[source][2]["tokens"] += tokens; replay[source][2]["bytes"] += len(out)
        finally:
            for handle in handles.values(): handle.close()
        print("replay", {s: replay[s][2]["tokens"] for s in replay}, flush=True)

    final = {"open_code_cot_novel": dict(code_stats), "verified_math_cot_novel": dict(math_stats),
             **{source: dict(value[2]) for source, value in replay.items()}, "math_target": math_target}
    (CACHE / "completion_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
