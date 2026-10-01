"""Full structural, provenance, hash and token audit of sft_continue.jsonl."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import orjson
from datasets import load_dataset  # noqa: F401
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
DATA = ROOT / "dataset/sft_continue.jsonl"
META = ROOT / "dataset/sft_continue.provenance.jsonl.gz"
REPORT = ROOT / "dataset/sft_continue.report.json"


def norm(text): return " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())
def digest(value): return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def prompt_hash(row):
    text = "\n".join(norm(m.get("content", "")) for m in row["conversations"] if m.get("role") in ("system", "user"))
    return digest(text)


def main():
    expected = json.loads(REPORT.read_text(encoding="utf-8"))
    with gzip.open(META, "rt", encoding="utf-8") as f: metadata = [json.loads(line) for line in f]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    from scripts.data_loader.lm_dataset import _create_chat_prompt
    errors, seen_prompt, seen_body, source_rows, source_tokens = [], set(), set(), {}, {}
    file_hash = hashlib.sha256(); count = 0; max_tokens = 0
    with DATA.open("rb") as f:
        for index, line in enumerate(f):
            file_hash.update(line); count += 1
            try:
                row = orjson.loads(line); meta = metadata[index]
                assert meta["output_row"] == index
                conv = row["conversations"]
                assert isinstance(conv, list) and conv
                assert all(isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool") and "content" in m for m in conv)
                assert any(m["role"] == "user" for m in conv) and any(m["role"] == "assistant" for m in conv)
                body_hash = digest(orjson.dumps(row)); ph = prompt_hash(row)
                assert body_hash == meta["body_hash"] and ph == meta["prompt_hash"]
                assert body_hash not in seen_body and ph not in seen_prompt
                seen_body.add(body_hash); seen_prompt.add(ph)
                tokens = len(tokenizer.backend_tokenizer.encode(_create_chat_prompt(tokenizer, conv), add_special_tokens=False).ids)
                assert tokens == meta["tokens"] and tokens <= 3072
                max_tokens = max(max_tokens, tokens)
                source_rows[meta["source"]] = source_rows.get(meta["source"], 0) + 1
                source_tokens[meta["source"]] = source_tokens.get(meta["source"], 0) + tokens
            except Exception as exc:
                errors.append({"row": index, "error": repr(exc)})
    result = {
        "method": "Reparsed and rerendered every final row; checked metadata correspondence, exact token count, hashes, roles and duplicates.",
        "rows_checked": count, "errors": len(errors), "first_errors": errors[:20],
        "source_rows": source_rows, "source_tokens": source_tokens, "max_tokens": max_tokens,
        "sha256": file_hash.hexdigest(), "report_rows_match": count == expected["rows"],
        "report_tokens_match": sum(source_tokens.values()) == expected["tokens"],
        "report_hash_match": file_hash.hexdigest() == expected["sha256"],
        "unique_prompt_hashes": len(seen_prompt), "unique_conversation_hashes": len(seen_body),
    }
    (ROOT / "dataset/sft_continue.audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors or not all((result["report_rows_match"], result["report_tokens_match"], result["report_hash_match"])): raise SystemExit(1)


if __name__ == "__main__": main()
