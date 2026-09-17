"""Merge reviewed code/math, instruction understanding, verified CoT and T2T replay."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

import orjson
from datasets import load_dataset  # noqa: F401  # before transformers/torch on Windows
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
OUTPUT = ROOT / "dataset" / "sft_continue.jsonl"
REPORT = ROOT / "dataset" / "sft_continue.report.json"
PROVENANCE = ROOT / "dataset" / "sft_continue.provenance.jsonl.gz"
REPLAY_ONLY = ROOT / "dataset" / "sft_continue_replay.jsonl"
REPLAY_REVIEW_MANIFEST = ROOT / "dataset" / "sft_continue_replay_reviews_manifest.json"
SEED = 2026091524
REPLAY_TOKEN_FRACTION = 0.05
BASE_SOURCES = (
    ("continue_sft_reviewed", ROOT / "dataset/quality_python_sft/continue_sft_reviewed_train.jsonl"),
    ("instruction_understanding", ROOT / "dataset/instruction_understanding_sft/train.jsonl"),
    ("verified_cot", ROOT / "dataset/verified_cot_sft/train.jsonl"),
)
REPLAY_SOURCE = ROOT / "dataset/sft_t2t_mini.jsonl"
BAD_TEXT = ("�", "<|im_start|>", "<|im_end|>", "[assistant/analysis]", "[tool/analysis]")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())


def digest(value):
    if not isinstance(value, bytes): value = str(value).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def conversation_prompt(row):
    return "\n".join(normalized(m.get("content", "")) for m in row["conversations"] if m.get("role") in ("system", "user"))


def visible_text(row):
    return "\n".join(str(m.get("content", "")) for m in row["conversations"] if m.get("content"))


def repeated_line(text):
    lines = [normalized(x) for x in text.splitlines() if len(normalized(x)) >= 30]
    return bool(lines and max(Counter(lines).values()) >= 3)


def replay_category(row):
    conv = row["conversations"]
    if any(m.get("tools") or m.get("tool_calls") or m.get("role") == "tool" for m in conv): return "tools"
    return "zh_general" if re.search(r"[\u4e00-\u9fff]", visible_text(row)) else "other_general"


def replay_quality(row):
    conv = row.get("conversations")
    if not isinstance(conv, list) or not 2 <= len(conv) <= 8: return None
    roles = [m.get("role") for m in conv if isinstance(m, dict)]
    if len(roles) != len(conv) or not any(x == "user" for x in roles) or not any(x == "assistant" for x in roles): return None
    if any(role not in ("system", "user", "assistant", "tool") for role in roles): return None
    # Previously observed hidden-reasoning replay contains verbose, low-quality
    # self-talk. Replay only clean visible-answer conversations.
    if any(str(m.get("reasoning_content", "")).strip() for m in conv): return None
    text = visible_text(row)
    if not text or any(marker in text for marker in BAD_TEXT) or repeated_line(text): return None
    user = "\n".join(str(m.get("content", "")) for m in conv if m.get("role") == "user")
    assistants = [str(m.get("content", "")) for m in conv if m.get("role") == "assistant" and m.get("content")]
    if not user.strip() or not assistants or normalized(user) == normalized(assistants[-1]): return None
    if not 8 <= len(user) <= 6000 or not 2 <= len(assistants[-1]) <= 8000: return None
    score = 0
    score += 5 if len(conv) == 2 else 3
    score += 3 if 20 <= len(user) <= 1500 else 1
    score += 3 if 20 <= len(assistants[-1]) <= 2000 else 1
    score += 2 if not re.search(r"(?i)as an ai|作为.{0,3}ai|无法访问互联网", assistants[-1]) else 0
    score += 2 if replay_category(row) == "tools" else 0
    return score


def clean_replay_row(raw):
    # Preserve tool metadata and multi-turn structure but omit empty optional
    # fields, producing the same schema expected by SFTDataset.
    conversations = []
    for message in raw["conversations"]:
        out = {"role": message.get("role"), "content": str(message.get("content") or "")}
        for key, value in message.items():
            if key not in ("role", "content", "reasoning_content") and value not in (None, "", [], {}):
                out[key] = value
        conversations.append(out)
    return {"conversations": conversations}


def token_count(tokenizer, row):
    from dataset.lm_dataset import _create_chat_prompt
    prompt = _create_chat_prompt(tokenizer, row["conversations"])
    return len(tokenizer.backend_tokenizer.encode(prompt, add_special_tokens=False).ids)


def apply_replay_reviews(selected):
    if not REPLAY_REVIEW_MANIFEST.is_file(): return selected, {"applied": False}
    manifest = json.loads(REPLAY_REVIEW_MANIFEST.read_text(encoding="utf-8"))
    payload = b"".join(orjson.dumps(item["row"]) + b"\n" for item in selected)
    actual = digest(payload)
    if actual != manifest["pre_review_sha256"] or len(selected) != manifest["pre_review_rows"]:
        raise RuntimeError("Replay review manifest does not match deterministically selected pre-review rows")
    excluded, decisions = set(), Counter()
    reviewed = 0
    for relative in manifest["review_files"]:
        review = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        reviewed += review.get("reviewed_rows", review.get("selected_rows", 0))
        for decision in ("reject", "caution"):
            for item in review.get(decision, []):
                line_numbers = item.get("lines", [item.get("line", item.get("row"))])
                for line_number in line_numbers:
                    if line_number is None: raise ValueError((relative, decision, item))
                    excluded.add(int(line_number)); decisions[decision] += 1
    if reviewed != len(selected) or any(i < 0 or i >= len(selected) for i in excluded):
        raise RuntimeError({"reviewed": reviewed, "selected": len(selected), "excluded": len(excluded)})
    kept = [item for i, item in enumerate(selected) if i not in excluded]
    return kept, {"applied": True, "pre_review_rows": len(selected), "reviewed_rows": reviewed,
                  "reject_rows": decisions["reject"], "caution_rows": decisions["caution"],
                  "excluded_rows": len(excluded), "kept_rows": len(kept), "pre_review_sha256": actual}


def main():
    for _, path in BASE_SOURCES:
        if not path.is_file(): raise FileNotFoundError(path)
    if not REPLAY_SOURCE.is_file(): raise FileNotFoundError(REPLAY_SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    entries, prompt_hashes, body_hashes = [], set(), set()
    source_stats = Counter(); source_tokens = Counter(); duplicate_stats = Counter()

    for source, path in BASE_SOURCES:
        with path.open("rb") as f:
            for source_row, line in enumerate(f):
                raw = orjson.loads(line); row = {"conversations": raw["conversations"]}
                prompt_hash = digest(conversation_prompt(row)); body = orjson.dumps(row); body_hash = digest(body)
                if prompt_hash in prompt_hashes or body_hash in body_hashes:
                    duplicate_stats[source] += 1; continue
                tokens = token_count(tokenizer, row)
                if not 1 <= tokens <= 3072: raise ValueError(f"{source}:{source_row} has {tokens} tokens")
                prompt_hashes.add(prompt_hash); body_hashes.add(body_hash)
                entries.append({"source": source, "source_row": source_row, "tokens": tokens,
                                "prompt_hash": prompt_hash, "body_hash": body_hash, "row": row})
                source_stats[source] += 1; source_tokens[source] += tokens

    base_tokens = sum(source_tokens.values())
    replay_target = round(base_tokens * REPLAY_TOKEN_FRACTION / (1 - REPLAY_TOKEN_FRACTION))
    quotas = {"zh_general": round(replay_target * 0.70),
              "other_general": round(replay_target * 0.20),
              "tools": replay_target - round(replay_target * 0.70) - round(replay_target * 0.20)}
    # Freeze an exhaustively reviewed candidate set. Changes to another source
    # must not pull previously unreviewed rows into the replay tail.
    if REPLAY_REVIEW_MANIFEST.is_file():
        frozen = json.loads(REPLAY_REVIEW_MANIFEST.read_text(encoding="utf-8"))
        replay_target = frozen["selection_token_target"]
        quotas = frozen["selection_token_quotas"]
    candidates = {key: [] for key in quotas}
    replay_scan = Counter()
    with REPLAY_SOURCE.open("rb") as f:
        for source_row, line in enumerate(f):
            replay_scan["rows"] += 1
            try: raw = orjson.loads(line)
            except orjson.JSONDecodeError: replay_scan["invalid_json"] += 1; continue
            quality = replay_quality(raw)
            if quality is None: replay_scan["quality_filter"] += 1; continue
            row = clean_replay_row(raw); prompt_hash = digest(conversation_prompt(row)); body = orjson.dumps(row); body_hash = digest(body)
            if prompt_hash in prompt_hashes or body_hash in body_hashes:
                replay_scan["base_overlap"] += 1; continue
            category = replay_category(row)
            # Quality score first, then deterministic hash. Keep a bounded pool;
            # every selected row will later receive full validation/review.
            rank = (-quality, digest(f"{SEED}/{source_row}/{body_hash}"))
            candidates[category].append((rank, source_row, row, prompt_hash, body_hash))
    for category in candidates:
        candidates[category].sort(key=lambda x: x[0]); candidates[category] = candidates[category][:12000]

    selected_replay = []
    replay_tokens_by_category = Counter()
    for category, quota in quotas.items():
        for _, source_row, row, prompt_hash, body_hash in candidates[category]:
            if prompt_hash in prompt_hashes or body_hash in body_hashes: continue
            tokens = token_count(tokenizer, row)
            if tokens > 1536: replay_scan["over_replay_token_limit"] += 1; continue
            if replay_tokens_by_category[category] and replay_tokens_by_category[category] + tokens > quota:
                continue
            item = {"source": "sft_t2t_mini_replay", "source_row": source_row,
                    "replay_category": category, "tokens": tokens,
                    "prompt_hash": prompt_hash, "body_hash": body_hash, "row": row}
            selected_replay.append(item); prompt_hashes.add(prompt_hash); body_hashes.add(body_hash)
            replay_tokens_by_category[category] += tokens
            if replay_tokens_by_category[category] >= quota: break
    if any(replay_tokens_by_category[k] < quotas[k] * 0.98 for k in quotas):
        raise RuntimeError({"quotas": quotas, "selected": replay_tokens_by_category})
    selected_replay, replay_review = apply_replay_reviews(selected_replay)
    entries.extend(selected_replay)
    source_stats["sft_t2t_mini_replay"] = len(selected_replay)
    source_tokens["sft_t2t_mini_replay"] = sum(x["tokens"] for x in selected_replay)

    # Deterministic content-independent shuffle; provenance retains source rows.
    entries.sort(key=lambda x: digest(f"{SEED}/shuffle/{x['body_hash']}"))
    tmp = OUTPUT.with_suffix(".jsonl.tmp")
    output_hash = hashlib.sha256()
    with tmp.open("wb") as f, gzip.open(PROVENANCE, "wt", encoding="utf-8", newline="\n") as meta:
        for output_row, item in enumerate(entries):
            line = orjson.dumps(item["row"]) + b"\n"; f.write(line); output_hash.update(line)
            meta.write(json.dumps({k: v for k, v in item.items() if k != "row"} |
                                  {"output_row": output_row}, ensure_ascii=False) + "\n")
    os.replace(tmp, OUTPUT)
    with REPLAY_ONLY.open("wb") as f:
        for item in selected_replay: f.write(orjson.dumps(item["row"]) + b"\n")

    total_tokens = sum(source_tokens.values())
    report = {
        "output": str(OUTPUT), "seed": SEED, "rows": len(entries), "tokens": total_tokens,
        "sha256": output_hash.hexdigest(), "source_rows": dict(source_stats),
        "source_tokens": dict(source_tokens),
        "source_token_fraction": {k: v / total_tokens for k, v in source_tokens.items()},
        "base_duplicates_removed": dict(duplicate_stats), "replay_source_rows_scanned": replay_scan["rows"],
        "replay_scan": dict(replay_scan), "replay_target_tokens": replay_target,
        "replay_token_quotas": quotas, "replay_tokens_by_category": dict(replay_tokens_by_category),
        "replay_full_review": replay_review,
        "max_tokens": max(x["tokens"] for x in entries),
        "checks": ["All source rows parsed", "all rendered rows <=3072 tokens",
                   "exact normalized system/user prompts and exact conversations deduplicated",
                   "replay rows contain no nonempty reasoning_content or repeated long lines"],
        "limitations": [
            "Verified CoT arithmetic is checked step by step, but the mixture also inherits source limitations from continue_sft_reviewed_train.",
            "T2T replay is retention data selected by strict structural filters; its open-ended semantics cannot be proven by automatic checks alone.",
            "continue_sft_reviewed_train includes CC-BY-NC-4.0 KodCode data, so this combined file is non-commercial unless that component is removed.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
