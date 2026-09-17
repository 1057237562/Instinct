"""Build 800M novel + 200M replay tokens for targeted continual pretraining."""
from collections import Counter
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import warnings

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401  # before transformers/torch on Windows
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from dataset.scripts.mix_sft_datasets import ExternalRandomShuffler

OUT_DIR = ROOT / "dataset/continue_pretrain_1b"
RAW = OUT_DIR / "raw"
OUTPUT = ROOT / "dataset/pretrain_continue.jsonl"
REPORT = ROOT / "dataset/pretrain_continue.report.json"
PRIOR = ROOT / "dataset/pretrain_codespecialist.jsonl"
MAX_TOKENS = 4096
SEED = 2026091602
NEW_TARGET = 800_000_000
REPLAY_TARGET = 200_000_000
TARGETS = {
    "python_edu_novel": 250_000_000,
    "open_code_cot_novel": 100_795_000,
    "verified_math_cot_novel": 349_205_000,
    "cosmopedia_novel": 80_000_000,
    "verified_math_code": 20_000_000,
    "replay_code": 100_000_000,
    "replay_general": 60_000_000,
    "replay_math": 20_000_000,
    "replay_technical": 20_000_000,
}
REPOS = {
    "python": ("Fhrozen/stack-prompts", "e0be91bbfa1a549eea273726108c705d38681304"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f"),
    "finemath": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594"),
    "verified": ("manifesta/verified-math-code-17k", "e3e8a607e16d731f4f456a34dd696a53c189c38c"),
    "open_code": ("nvidia/OpenCodeReasoning", "20a1ca19c0d050fe9057fc08339d6b370ec1c67a"),
    "open_r1_math": ("open-r1/OpenR1-Math-220k", "e4e141ec9dea9f8326f4d347be56105859b2bd68"),
    "open_math": ("nvidia/OpenMathReasoning", "d3d08664755704f422af97d43a7ff0ded4bd95df"),
}
BAD = ("�", "Ã", "Â", "<|im_start|>", "<|im_end|>", "write a paper online",
       "online casino", "click here to buy", "[assistant/analysis]")
CODE_SOURCES = {"open_repository_code_and_docs", "competitive_problem_reasoning",
                "verified_competitive_submissions", "code_instruction",
                "text_to_sql", "exercism_software_tasks"}
REPLAY_MAP = {**{name: "replay_code" for name in CODE_SOURCES},
              "general_bilingual": "replay_general", "math_reasoning": "replay_math",
              "academic_technical": "replay_technical"}
REPLAY_PROB = {"replay_code": .16, "replay_general": .24,
               "replay_math": .16, "replay_technical": .16}


def h16(text): return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def words(text): return re.findall(r"[A-Za-z_]\w*|\d+", text.lower())
def has_overlap(text, targets, n=13):
    tokens = words(text)
    return any(tuple(tokens[i:i+n]) in targets for i in range(len(tokens)-n+1))


def normalize_math_answer(value):
    """Conservative normalization for exact expected/boxed answer matching."""
    text = re.sub(r"\s+", "", str(value or ""))
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    changed = True
    while changed and text:
        changed = False
        for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$"), ("{", "}")):
            if text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)]
                changed = True
    return text.rstrip(".,")


def last_boxed_answer(text):
    """Return the contents of the final balanced ``\\boxed{...}``, if any."""
    marker = "\\boxed{"
    for start in reversed([match.start() for match in re.finditer(re.escape(marker), text)]):
        depth = 1
        index = start + len(marker)
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(marker):index]
            index += 1
    return None


def clean(text, minimum=100, maximum=50000):
    if not isinstance(text, str) or not minimum <= len(text) <= maximum: return False
    lower = text.lower()
    if any(marker.lower() in lower for marker in BAD) or "\x00" in text: return False
    if any(ord(c) < 32 and c not in "\r\n\t" for c in text): return False
    lines = [" ".join(x.lower().split()) for x in text.splitlines() if len(" ".join(x.split())) >= 50]
    if lines and max(Counter(lines).values()) >= 3: return False
    return True


def benchmark_sets():
    he, gsm = set(), set()
    with gzip.open(ROOT / "dataset/humaneval/HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens = words("\n".join(str(value) for value in row.values()))
            he.update(tuple(tokens[i:i+13]) for i in range(len(tokens)-12))
    with (ROOT / "dataset/gsm8k/test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            tokens = words(json.loads(line)["question"])
            gsm.update(tuple(tokens[i:i+13]) for i in range(len(tokens)-12))
    return he, gsm


def download(key, filename, downloads):
    repo, revision = REPOS[key]
    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                local_dir=RAW / key))
    downloads.append({"repo": repo, "revision": revision, "file": filename,
                      "bytes": path.stat().st_size, "sha256": sha_file(path)})
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    if not PRIOR.is_file(): raise FileNotFoundError(PRIOR)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True).backend_tokenizer
    he_grams, gsm_grams = benchmark_sets()
    prior_hashes = set(); prior_rows = 0
    print("[1/6] Indexing every previously trained pretraining record", flush=True)
    with PRIOR.open("rb") as f:
        for line in f:
            row = orjson.loads(line); text = row.get("text")
            if isinstance(text, str): prior_hashes.add(h16(text)); prior_rows += 1

    stats = {name: Counter() for name in TARGETS}; downloads = []; seen = set()
    shuffler = ExternalRandomShuffler(OUTPUT, SEED, 16000)

    def full(source): return stats[source]["tokens"] >= TARGETS[source] - MAX_TOKENS

    def add(text, source, source_id, license_name, url, metadata=None, allow_prior=False):
        s = stats[source]
        if full(source): return False
        fp = h16(text)
        if fp in seen: s["duplicate_new"] += 1; return False
        if not allow_prior and fp in prior_hashes: s["duplicate_prior"] += 1; return False
        tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
        if not 64 <= tokens <= MAX_TOKENS: s["outside_token_range"] += 1; return False
        if s["tokens"] + tokens > TARGETS[source]: s["over_budget"] += 1; return False
        row = {"text": text, "source": source, "source_id": str(source_id),
               "license": license_name, "url": url, "token_count": tokens}
        if metadata: row["metadata"] = metadata
        if not shuffler.add(row): s["duplicate_record"] += 1; return False
        seen.add(fp); s["rows"] += 1; s["tokens"] += tokens
        s["text_bytes"] += len(text.encode("utf-8")); return True

    print("[2/7] Downloading educational Python and retaining score>=4, clean AST files", flush=True)
    for shard in range(60):
        if full("python_edu_novel"): break
        filename = f"smollm-corpus/python_edu-{shard:05d}-of-00060.parquet"
        path = download("python", filename, downloads)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,
                columns=["blob_id", "int_score", "annotated", "original_code"]):
            for row in batch.to_pylist():
                s = stats["python_edu_novel"]; s["scanned"] += 1
                if not row.get("annotated") or int(row.get("int_score") or 0) < 4: s["low_score"] += 1; continue
                code = row.get("original_code")
                if not clean(code, 100, 50000): s["text_quality"] += 1; continue
                try:
                    tree = ast.parse(code)
                    if not any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for x in ast.walk(tree)):
                        s["no_definition"] += 1; continue
                except (SyntaxError, ValueError): s["syntax"] += 1; continue
                if has_overlap(code, he_grams) or has_overlap(code, gsm_grams):
                    s["benchmark_overlap"] += 1; continue
                add(code.rstrip(), "python_edu_novel", row["blob_id"],
                    "per-file Software Heritage license; consult upstream", 
                    f"https://huggingface.co/datasets/{REPOS['python'][0]}",
                    {"educational_score": int(row["int_score"])})
                if full("python_edu_novel"): break
            if full("python_edu_novel"): break
        print(f"  Python shard {shard}: {stats['python_edu_novel']['tokens']:,}/250,000,000", flush=True)

    print("[3/7] Downloading Cosmopedia educational text", flush=True)
    for shard in range(104):
        if full("cosmopedia_novel"): break
        filename = f"cosmopedia-v2/train-{shard:05d}-of-00104.parquet"
        path = download("cosmopedia", filename, downloads)
        parquet = pq.ParquetFile(path)
        columns = [x for x in ("text", "prompt", "audience", "format") if x in parquet.schema.names]
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                s = stats["cosmopedia_novel"]; s["scanned"] += 1; text = row.get("text")
                if not clean(text, 300, 30000): s["text_quality"] += 1; continue
                fmt = str(row.get("format") or "").lower()
                if "story" in fmt: s["story_excluded"] += 1; continue
                if has_overlap(text, he_grams) or has_overlap(text, gsm_grams): s["benchmark_overlap"] += 1; continue
                add(text.strip(), "cosmopedia_novel", f"{shard}:{s['scanned']-1}", "ODC-By-1.0",
                    f"https://huggingface.co/datasets/{REPOS['cosmopedia'][0]}",
                    {"format": row.get("format"), "audience": row.get("audience")})
                if full("cosmopedia_novel"): break
            if full("cosmopedia_novel"): break
        print(f"  Cosmopedia shard {shard}: {stats['cosmopedia_novel']['tokens']:,}/80,000,000", flush=True)

    print("[4/7] Downloading mechanically verified math/code", flush=True)
    vpath = download("verified", "data/train-00000-of-00001.parquet", downloads)
    for batch in pq.ParquetFile(vpath).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            s = stats["verified_math_code"]; s["scanned"] += 1
            if row.get("verify_level") in (None, "", "-") or row.get("domain") not in ("math", "code"):
                s["unverified"] += 1; continue
            problem, solution = row.get("problem"), row.get("worked_solution")
            if not clean(problem, 40, 20000) or not clean(solution, 40, 40000): s["text_quality"] += 1; continue
            combined = problem + "\n" + solution
            if has_overlap(combined, he_grams) or has_overlap(combined, gsm_grams): s["benchmark_overlap"] += 1; continue
            if row["domain"] == "code":
                blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", solution, re.S | re.I)
                if not blocks: s["missing_code"] += 1; continue
                try: ast.parse(max(blocks, key=len))
                except SyntaxError: s["code_syntax"] += 1; continue
            text = "### Problem\n" + problem.strip() + "\n\n### Verified solution\n" + solution.strip()
            add(text, "verified_math_code", f"{row['source']}:{s['scanned']-1}", "CC-BY-SA-4.0",
                f"https://huggingface.co/datasets/{REPOS['verified'][0]}",
                {"domain": row["domain"], "verify_level": row["verify_level"], "difficulty": row.get("difficulty")})
    # Backfill this small mechanically verified source with the larger verified
    # math-CoT pool while preserving exactly 800M novel tokens.
    verified_shortfall = TARGETS["verified_math_code"] - stats["verified_math_code"]["tokens"]
    TARGETS["verified_math_cot_novel"] += max(0, verified_shortfall)

    print("[5/7] Downloading code reasoning with complete Python solutions", flush=True)
    code_question_counts = Counter()
    for shard in range(30):
        if full("open_code_cot_novel"): break
        filename = f"split_0/train-{shard:05d}-of-00030.parquet"
        path = download("open_code", filename, downloads)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,
                columns=["id", "input", "output", "solution", "source", "license", "dataset", "split", "difficulty"]):
            for row in batch.to_pylist():
                s = stats["open_code_cot_novel"]; s["scanned"] += 1
                if str(row.get("split") or "").lower() != "train": s["non_train_split"] += 1; continue
                qid = str(row.get("id") or "")
                if code_question_counts[qid] >= 16: s["question_cap"] += 1; continue
                problem, reasoning, solution = row.get("input"), row.get("output"), row.get("solution")
                if not clean(problem, 80, 30000) or not clean(reasoning, 200, 50000) or not clean(solution, 30, 30000):
                    s["text_quality"] += 1; continue
                try: ast.parse(solution)
                except SyntaxError: s["solution_syntax"] += 1; continue
                combined = problem + "\n" + reasoning + "\n" + solution
                if has_overlap(combined, he_grams) or has_overlap(combined, gsm_grams): s["benchmark_overlap"] += 1; continue
                text = ("### Programming problem\n" + problem.strip()
                        + "\n\n### Reasoning\n" + reasoning.strip()
                        + "\n\n### Verified Python solution\n```python\n"
                        + solution.strip() + "\n```")
                if add(text, "open_code_cot_novel", f"{qid}:{code_question_counts[qid]}",
                       str(row.get("license") or "CC-BY-4.0"),
                       f"https://huggingface.co/datasets/{REPOS['open_code'][0]}",
                       {"source": row.get("source"), "dataset": row.get("dataset"), "difficulty": row.get("difficulty")}):
                    code_question_counts[qid] += 1
                if full("open_code_cot_novel"): break
            if full("open_code_cot_novel"): break
        print(f"  Code-CoT shard {shard}: {stats['open_code_cot_novel']['tokens']:,}/{TARGETS['open_code_cot_novel']:,}", flush=True)

    print("[6/7] Downloading answer-verified math reasoning", flush=True)
    math_problem_hashes = set()
    for shard in range(10):
        if full("verified_math_cot_novel"): break
        filename = f"data/train-{shard:05d}-of-00010.parquet"
        path = download("open_r1_math", filename, downloads)
        columns = ["problem", "solution", "answer", "source", "uuid", "is_reasoning_complete", "correctness_count"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                s = stats["verified_math_cot_novel"]; s["openr1_scanned"] += 1
                problem, solution = row.get("problem"), row.get("solution")
                if int(row.get("correctness_count") or 0) < 1 or not any(row.get("is_reasoning_complete") or []):
                    s["not_verified_complete"] += 1; continue
                if not clean(problem, 40, 20000) or not clean(solution, 200, 50000) or "\\boxed" not in solution:
                    s["text_quality"] += 1; continue
                ph = h16(problem)
                if ph in math_problem_hashes: s["duplicate_problem"] += 1; continue
                combined = problem + "\n" + solution
                if has_overlap(combined, he_grams) or has_overlap(combined, gsm_grams): s["benchmark_overlap"] += 1; continue
                text = "### Math problem\n" + problem.strip() + "\n\n### Verified reasoning\n" + solution.strip()
                if add(text, "verified_math_cot_novel", row.get("uuid"), "Apache-2.0",
                       f"https://huggingface.co/datasets/{REPOS['open_r1_math'][0]}",
                       {"upstream_source": row.get("source"), "verification": "Math-Verify or upstream judge"}):
                    math_problem_hashes.add(ph)
                if full("verified_math_cot_novel"): break
            if full("verified_math_cot_novel"): break
        print(f"  OpenR1-Math shard {shard}: {stats['verified_math_cot_novel']['tokens']:,}/{TARGETS['verified_math_cot_novel']:,}", flush=True)

    # NVIDIA's released CoT pool supplies additional diverse, answer-filtered
    # problems when the shorter OpenR1 default subset cannot fill the 4K budget.
    for shard in range(144):
        if full("verified_math_cot_novel"): break
        filename = f"data/cot-{shard:05d}-of-00144.parquet"
        path = download("open_math", filename, downloads)
        columns = ["expected_answer", "problem_type", "problem_source", "generation_model",
                   "problem", "generated_solution", "inference_mode", "used_in_kaggle"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                s = stats["verified_math_cot_novel"]; s["openmath_scanned"] += 1
                problem, solution, answer = row.get("problem"), row.get("generated_solution"), str(row.get("expected_answer") or "").strip()
                if not answer or not clean(problem, 40, 20000) or not clean(solution, 200, 50000): s["openmath_text_quality"] += 1; continue
                ph = h16(problem)
                if ph in math_problem_hashes: s["duplicate_problem"] += 1; continue
                boxed = last_boxed_answer(solution)
                if boxed is None or not normalize_math_answer(answer) \
                        or normalize_math_answer(boxed) != normalize_math_answer(answer):
                    s["boxed_answer_mismatch"] += 1; continue
                combined = problem + "\n" + solution
                if has_overlap(combined, he_grams) or has_overlap(combined, gsm_grams): s["benchmark_overlap"] += 1; continue
                text = "### Math problem\n" + problem.strip() + "\n\n### Verified reasoning\n" + solution.strip()
                if add(text, "verified_math_cot_novel", f"openmath:{shard}:{s['openmath_scanned']-1}", "CC-BY-4.0",
                       f"https://huggingface.co/datasets/{REPOS['open_math'][0]}",
                       {"problem_source": row.get("problem_source"), "generation_model": row.get("generation_model"),
                        "expected_answer": answer, "used_in_kaggle": row.get("used_in_kaggle")}):
                    math_problem_hashes.add(ph)
                if full("verified_math_cot_novel"): break
            if full("verified_math_cot_novel"): break
        print(f"  OpenMath shard {shard}: {stats['verified_math_cot_novel']['tokens']:,}/{TARGETS['verified_math_cot_novel']:,}", flush=True)

    print("[7/7] Adding exactly 200M tokens of stratified prior replay", flush=True)
    for band in range(8):
        if all(full(x) for x in REPLAY_PROB): break
        with PRIOR.open("rb") as f:
            for index, line in enumerate(f):
                row = orjson.loads(line); source = REPLAY_MAP.get(row.get("source"))
                if source is None or full(source): continue
                value = int.from_bytes(hashlib.blake2b(f"{SEED}/{index}".encode(), digest_size=8).digest(), "big") / 2**64
                probability = REPLAY_PROB[source]
                if not band * probability <= value < min(1.0, (band + 1) * probability): continue
                text = row.get("text")
                if not clean(text, 80, 50000): stats[source]["text_quality"] += 1; continue
                add(text, source, index, row.get("license") or "inherited; see pretrain_codespecialist report",
                    row.get("url") or "", {"prior_source": row.get("source")}, allow_prior=True)
    short = {name: TARGETS[name] - stats[name]["tokens"] for name in TARGETS if not full(name)}
    if short: raise RuntimeError(f"Components below target: {short}")
    novel_tokens = sum(stats[x]["tokens"] for x in ("python_edu_novel", "open_code_cot_novel",
                       "verified_math_cot_novel", "cosmopedia_novel", "verified_math_code"))
    replay_tokens = sum(stats[x]["tokens"] for x in REPLAY_PROB)
    if not (NEW_TARGET - 5 * MAX_TOKENS <= novel_tokens <= NEW_TARGET): raise AssertionError(novel_tokens)
    if not (REPLAY_TARGET - 4 * MAX_TOKENS <= replay_tokens <= REPLAY_TARGET): raise AssertionError(replay_tokens)
    shuffler.finish()
    total = novel_tokens + replay_tokens
    report = {
        "model_parameters": 152_406_528, "requested_mix": {"novel_tokens": NEW_TARGET, "replay_tokens": REPLAY_TARGET},
        "output": str(OUTPUT), "output_sha256": sha_file(OUTPUT),
        "rows": sum(x["rows"] for x in stats.values()), "tokens": total,
        "novel_tokens": novel_tokens, "replay_tokens": replay_tokens,
        "tokens_per_parameter": total / 152_406_528,
        "max_tokens_including_bos_eos": MAX_TOKENS, "whole_records_only": True, "text_truncated": False,
        "targets_after_verified_backfill": TARGETS,
        "components": {name: dict(value) for name, value in stats.items()},
        "component_token_fraction": {name: value["tokens"] / total for name, value in stats.items()},
        "prior_corpus": {"path": str(PRIOR), "rows_indexed": prior_rows, "unique_text_hashes": len(prior_hashes)},
        "downloads": downloads,
        "checks": ["Every selected Python file parsed as Python AST", "all new exact texts excluded from prior corpus",
                   "all selected records contain 64..4096 tokens including BOS/EOS", "no record split or truncated",
                   "all new sources screened against HumanEval and GSM8K-test 13-token n-grams"],
        "limitations": [
            "AST parsing proves syntax, not runtime correctness of every Python repository file.",
            "OpenCodeReasoning solutions are syntax-checked locally, but upstream does not publish a per-row execution flag; reasoning correctness is not mechanically proven for every row.",
            "Cosmopedia is a filtered synthetic textbook corpus; local checks cannot prove every natural-language claim.",
            "Python source retains heterogeneous per-file Software Heritage licenses; the combined corpus is not certified for commercial redistribution.",
            "The verified-math-code aggregate is CC-BY-SA-4.0; share-alike obligations apply to redistribution of that component and potentially the mixture.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
