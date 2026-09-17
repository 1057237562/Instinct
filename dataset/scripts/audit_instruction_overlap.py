"""Full exact-overlap and HumanEval n-gram audit for instruction SFT rows."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "instruction_understanding_sft"


def normalize(text):
    # Preserve semantic punctuation such as minus signs, JSON delimiters and
    # strict/inclusive comparison wording; normalize only Unicode and whitespace.
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def digest(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def words(text):
    return re.findall(r"[a-zA-Z_]+|\d+|[\u4e00-\u9fff]", text.lower())


def ngrams(text, n=13):
    tokens = words(text)
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


def prompts(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield "\n".join(m["content"] for m in row["conversations"] if m["role"] == "user")


def main():
    current = list(prompts(DATA / "train.jsonl")) + list(prompts(DATA / "validation.jsonl"))
    current_hashes = {digest(x) for x in current}
    prior_path = ROOT / "dataset" / "sft_t2t_mini.jsonl"
    prior_exact = 0
    prior_rows = 0
    with prior_path.open("r", encoding="utf-8") as f:
        for line in f:
            prior_rows += 1
            row = json.loads(line)
            text = "\n".join(str(m.get("content", "")) for m in row.get("conversations", []) if m.get("role") == "user")
            prior_exact += digest(text) in current_hashes

    benchmark_ngrams = set()
    benchmark_tasks = 0
    with gzip.open(ROOT / "dataset" / "humaneval" / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            benchmark_tasks += 1
            benchmark_ngrams |= ngrams(json.loads(line)["prompt"])
    he_hits = []
    for i, text in enumerate(current):
        overlap = ngrams(text) & benchmark_ngrams
        if overlap:
            he_hits.append({"row_across_train_validation": i, "overlap_count": len(overlap),
                            "first_overlap": list(next(iter(overlap)))})
    report = {
        "method": "Scanned every new train/validation prompt and every prior-SFT/HumanEval prompt; no sampling.",
        "new_prompts_checked": len(current),
        "new_normalized_prompt_duplicates": len(current) - len(current_hashes),
        "prior_sft_rows_checked": prior_rows,
        "normalized_exact_prompt_matches_with_prior_sft": prior_exact,
        "humaneval_tasks_checked": benchmark_tasks,
        "new_prompts_with_any_humaneval_13_token_ngram": len(he_hits),
        "first_humaneval_ngram_hits": he_hits[:20],
        "scope": "Exact normalized prompt and lexical 13-token overlap checks do not prove absence of semantic similarity.",
    }
    (DATA / "overlap_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
