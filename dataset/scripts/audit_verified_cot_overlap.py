"""Full old-SFT exact and GSM8K-test lexical-overlap audit for verified CoT."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/verified_cot_sft"


def norm(text): return " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())
def digest(text): return hashlib.sha256(norm(text).encode()).hexdigest()
def words(text): return re.findall(r"[a-zA-Z_]+|\d+|[\u4e00-\u9fff]", str(text).lower())
def grams(text, n=13):
    values = words(text); return {tuple(values[i:i+n]) for i in range(len(values)-n+1)}


def main():
    prompts = []
    for split in ("train", "validation"):
        with (DATA / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            prompts.extend(json.loads(line)["conversations"][0]["content"] for line in f)
    hashes = {digest(x) for x in prompts}
    prior_hits = 0; prior_rows = 0
    with (ROOT / "dataset/sft_t2t_mini.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            prior_rows += 1; row = json.loads(line)
            prompt = "\n".join(str(m.get("content", "")) for m in row.get("conversations", []) if m.get("role") == "user")
            prior_hits += digest(prompt) in hashes
    test_grams = set(); test_rows = 0
    with (ROOT / "dataset/gsm8k/test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            test_rows += 1; test_grams |= grams(json.loads(line)["question"])
    hits = []
    for index, prompt in enumerate(prompts):
        overlap = grams(prompt) & test_grams
        if overlap: hits.append({"new_row": index, "overlap_count": len(overlap), "first": list(next(iter(overlap)))})
    report = {"method": "Every new CoT prompt scanned; every old-SFT and GSM8K-test prompt checked, without sampling.",
              "new_cot_prompts": len(prompts), "old_sft_rows": prior_rows,
              "old_sft_normalized_exact_prompt_hits": prior_hits, "gsm8k_test_rows": test_rows,
              "new_prompts_with_gsm8k_test_13_token_ngram": len(hits), "first_hits": hits[:20],
              "scope": "Lexical checks do not prove absence of semantic similarity."}
    (DATA / "overlap_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
