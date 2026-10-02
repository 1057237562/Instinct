"""Collect a novel, targeted CPT corpus for the 152M-parameter checkpoint."""
from collections import Counter
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401  # before transformers/torch on Windows
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler

OUT_DIR = ROOT / "dataset" / "continue_pretrain"
RAW = OUT_DIR / "raw"
OUTPUT = ROOT / "dataset" / "pretrain_continue.jsonl"
REPORT = ROOT / "dataset" / "pretrain_continue.report.json"
PRIOR = ROOT / "dataset" / "pretrain_codespecialist.jsonl"
TOKEN_TARGETS = {"python_edu_novel": 80_000_000, "verified_math_code": 20_000_000,
                 "prior_general_replay": 20_000_000}
MAX_TOKENS = 4096
SEED = 2026091601
STACK_REPO = "Fhrozen/stack-prompts"
STACK_REVISION = "e0be91bbfa1a549eea273726108c705d38681304"
VERIFIED_REPO = "manifesta/verified-math-code-17k"
VERIFIED_REVISION = "e3e8a607e16d731f4f456a34dd696a53c189c38c"
BAD_MARKERS = ("�", "Ã", "Â", "[assistant/analysis]", "<|im_start|>", "<|im_end|>")


def h16(text): return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def lexical(text): return re.findall(r"[A-Za-z_]\w*|\d+", text.lower())
def ngrams(text, n=13):
    words = lexical(text); return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def clean_text(text, min_chars=40, max_chars=50000):
    if not isinstance(text, str) or not min_chars <= len(text) <= max_chars: return False
    if any(marker in text for marker in BAD_MARKERS) or "\x00" in text: return False
    controls = sum(ord(c) < 32 and c not in "\n\r\t" for c in text)
    return controls == 0 and len(text.strip()) >= min_chars


def python_quality(code):
    if not clean_text(code, 200, 30000): return None
    if max((len(line) for line in code.splitlines()), default=0) > 1200: return None
    try: tree = ast.parse(code)
    except (SyntaxError, ValueError): return None
    definitions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not 1 <= len(definitions) <= 40: return None
    documentation = len(ast.get_docstring(tree) or "")
    documentation += sum(len(ast.get_docstring(node) or "") for node in definitions)
    documentation += sum(len(line) for line in code.splitlines() if line.lstrip().startswith("#"))
    if documentation < 100: return None
    names = [node.name for node in definitions if len(node.name) >= 3]
    return tree, names, documentation


def description_fields(value):
    if not clean_text(value, 30, 10000): return None
    try: data = json.loads(value)
    except json.JSONDecodeError: return None
    if not isinstance(data, dict): return None
    summary = data.get("summary")
    flow = data.get("logic_flow")
    if not isinstance(summary, str) or not 20 <= len(summary) <= 1000: return None
    if not isinstance(flow, list) or not 1 <= len(flow) <= 30 or not all(isinstance(x, str) and x.strip() for x in flow): return None
    return summary.strip(), [x.strip() for x in flow]


def benchmark_ngrams():
    he, gsm = set(), set()
    with gzip.open(ROOT / "dataset/humaneval/HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line); he |= ngrams(row["prompt"] + "\n" + row["canonical_solution"])
    with (ROOT / "dataset/gsm8k/test.jsonl").open("r", encoding="utf-8") as f:
        for line in f: gsm |= ngrams(json.loads(line)["question"])
    return he, gsm


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    if not PRIOR.is_file(): raise FileNotFoundError(PRIOR)
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True).backend_tokenizer
    he_grams, gsm_grams = benchmark_ngrams()
    prior_hashes, replay_pool = set(), []
    prior_stats = Counter()
    print("Indexing all previously trained CPT texts", flush=True)
    with PRIOR.open("rb") as f:
        for index, line in enumerate(f):
            row = orjson.loads(line); text = row.get("text")
            if not isinstance(text, str): prior_stats["invalid"] += 1; continue
            prior_hashes.add(h16(text)); prior_stats["rows"] += 1
            if row.get("source") in ("general_bilingual", "academic_technical") and clean_text(text, 80, 20000):
                rank = hashlib.sha256(f"{SEED}/{index}".encode()).hexdigest()
                if rank[:2] < "20": replay_pool.append((rank, index, text, row))

    stats = {key: Counter() for key in TOKEN_TARGETS}; downloads = []
    seen_new = set(); shuffler = ExternalRandomShuffler(OUTPUT, SEED, 20000)

    def add(text, source, source_id, license_name, url, extra=None):
        if stats[source]["tokens"] >= TOKEN_TARGETS[source] - MAX_TOKENS: return False
        fingerprint = h16(text)
        if fingerprint in seen_new: stats[source]["duplicate_new"] += 1; return False
        if source != "prior_general_replay" and fingerprint in prior_hashes:
            stats[source]["duplicate_prior"] += 1; return False
        tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
        if not 128 <= tokens <= MAX_TOKENS: stats[source]["outside_token_range"] += 1; return False
        if stats[source]["tokens"] + tokens > TOKEN_TARGETS[source]: stats[source]["over_budget"] += 1; return False
        record = {"text": text, "source": source, "source_id": str(source_id),
                  "license": license_name, "url": url, "token_count": tokens}
        if extra: record["metadata"] = extra
        if not shuffler.add(record): stats[source]["duplicate_record"] += 1; return False
        seen_new.add(fingerprint); stats[source]["rows"] += 1; stats[source]["tokens"] += tokens
        stats[source]["text_bytes"] += len(text.encode("utf-8")); return True

    print("Collecting and validating novel Python-Edu files", flush=True)
    for shard_index in range(60):
        if stats["python_edu_novel"]["tokens"] >= TOKEN_TARGETS["python_edu_novel"] - MAX_TOKENS: break
        name = f"smollm-corpus/python_edu-{shard_index:05d}-of-00060.parquet"
        path = Path(hf_hub_download(STACK_REPO, name, repo_type="dataset", revision=STACK_REVISION,
                                    local_dir=RAW / "stack_prompts"))
        downloads.append({"repo": STACK_REPO, "revision": STACK_REVISION, "file": name,
                          "bytes": path.stat().st_size, "sha256": sha_file(path)})
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                s = stats["python_edu_novel"]; s["scanned"] += 1
                if not row.get("annotated") or int(row.get("int_score") or 0) < 5: s["score"] += 1; continue
                code = row.get("original_code"); quality = python_quality(code)
                if quality is None: s["code_quality"] += 1; continue
                desc = description_fields(row.get("descript"))
                prompt = row.get("prompt")
                if desc is None or not clean_text(prompt, 30, 6000): s["metadata_quality"] += 1; continue
                _, names, documentation = quality; summary, flow = desc
                metadata_text = prompt + "\n" + summary + "\n" + "\n".join(flow)
                if names and not any(re.search(rf"\b{re.escape(name)}\b", metadata_text) for name in names):
                    s["identifier_alignment"] += 1; continue
                if ngrams(prompt + "\n" + code) & he_grams: s["humaneval_overlap"] += 1; continue
                if h16(code) in prior_hashes: s["code_duplicate_prior"] += 1; continue
                text = ("### Python module description\n" + summary + "\n\n### Intended behavior\n- " +
                        "\n- ".join(flow) + "\n\n### Reference implementation\n```python\n" + code.rstrip() + "\n```")
                add(text, "python_edu_novel", row["blob_id"],
                    "generated description Apache-2.0; code retains per-file Software Heritage license",
                    f"https://huggingface.co/datasets/{STACK_REPO}/tree/{STACK_REVISION}",
                    {"educational_score": int(row["int_score"]), "documented_chars": documentation})
                if s["tokens"] >= TOKEN_TARGETS["python_edu_novel"] - MAX_TOKENS: break
            if stats["python_edu_novel"]["tokens"] >= TOKEN_TARGETS["python_edu_novel"] - MAX_TOKENS: break
        print(f"  shard {shard_index}: {stats['python_edu_novel']['tokens']:,} tokens", flush=True)

    print("Collecting fully verified math/code records", flush=True)
    verified_name = "data/train-00000-of-00001.parquet"
    verified_path = Path(hf_hub_download(VERIFIED_REPO, verified_name, repo_type="dataset", revision=VERIFIED_REVISION,
                                         local_dir=RAW / "verified_math_code"))
    downloads.append({"repo": VERIFIED_REPO, "revision": VERIFIED_REVISION, "file": verified_name,
                      "bytes": verified_path.stat().st_size, "sha256": sha_file(verified_path)})
    for batch in pq.ParquetFile(verified_path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            s = stats["verified_math_code"]; s["scanned"] += 1
            if row.get("verify_level") in (None, "", "-") or row.get("domain") not in ("math", "code"):
                s["not_verified"] += 1; continue
            problem, solution = row.get("problem"), row.get("worked_solution")
            if not clean_text(problem, 40, 20000) or not clean_text(solution, 40, 30000): s["text_quality"] += 1; continue
            overlap = ngrams(problem + "\n" + solution)
            if overlap & he_grams or overlap & gsm_grams: s["benchmark_overlap"] += 1; continue
            if row["domain"] == "code":
                blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", solution, re.S | re.I)
                if not blocks:
                    s["code_missing"] += 1; continue
                try: ast.parse(max(blocks, key=len))
                except SyntaxError: s["code_syntax"] += 1; continue
            if h16(problem) in prior_hashes or h16(solution) in prior_hashes:
                s["component_duplicate_prior"] += 1; continue
            text = "### Problem\n" + problem.strip() + "\n\n### Verified worked solution\n" + solution.strip()
            add(text, "verified_math_code", f"{row['source']}:{s['scanned']-1}", "CC-BY-SA-4.0",
                f"https://huggingface.co/datasets/{VERIFIED_REPO}",
                {"domain": row["domain"], "difficulty": row.get("difficulty"), "verify_level": row["verify_level"]})

    print("Adding prior general/technical replay", flush=True)
    for _, index, text, row in sorted(replay_pool):
        add(text, "prior_general_replay", index, row.get("license") or "mixed; inherited from pretrain_codespecialist",
            row.get("url") or "", {"prior_source": row.get("source")})
        if stats["prior_general_replay"]["tokens"] >= TOKEN_TARGETS["prior_general_replay"] - MAX_TOKENS: break

    if stats["python_edu_novel"]["tokens"] < TOKEN_TARGETS["python_edu_novel"] * 0.98:
        raise RuntimeError("Insufficient validated novel Python tokens")
    if stats["prior_general_replay"]["tokens"] < TOKEN_TARGETS["prior_general_replay"] * 0.98:
        raise RuntimeError("Insufficient general replay tokens")
    shuffler.finish()
    output_sha = sha_file(OUTPUT); total_tokens = sum(x["tokens"] for x in stats.values())
    report = {
        "model_parameters": 152_406_528, "purpose": "targeted continual pretraining before re-running SFT",
        "output": str(OUTPUT), "output_sha256": output_sha,
        "rows": sum(x["rows"] for x in stats.values()), "tokens": total_tokens,
        "max_tokens_including_bos_eos": MAX_TOKENS, "targets": TOKEN_TARGETS,
        "components": {key: dict(value) for key, value in stats.items()},
        "component_token_fraction": {key: value["tokens"] / total_tokens for key, value in stats.items()},
        "prior_index": {"path": str(PRIOR), "rows": prior_stats["rows"], "unique_exact_text_hashes": len(prior_hashes)},
        "benchmark_screening": {"HumanEval": "all prompt+canonical-solution 13-token n-grams",
                                "GSM8K_test": "all question 13-token n-grams"},
        "downloads": downloads,
        "limitations": [
            "Python files pass syntax, documentation, encoding, score, identifier-alignment and lexical decontamination checks; this is not execution of every repository file.",
            "Python source files retain heterogeneous per-file Software Heritage licenses; use is not certified for commercial distribution.",
            "Verified math/code records use upstream mechanical verification; local checks validate format, syntax and benchmark overlap but do not rerun unavailable upstream tests.",
            "Prior replay is intentional and is excluded from novelty claims.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
