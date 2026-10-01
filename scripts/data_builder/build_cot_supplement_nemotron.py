# -*- coding: utf-8 -*-
"""Extract genuine chain-of-thought rows from Nemotron-Competitive-Programming-v2
for pretrain supplementation.

Key fact discovered 2026-09-18: the assistant message carries the CoT in a
separate `reasoning_content` field; visible `content` is usually pure code.
scripts/data_builder/build_codespecialist_mix.py render_messages() only rendered
`content`, so the existing corpus has these rows WITHOUT their reasoning.
This extractor renders per the repo chat template convention
(model/tokenizer_config.json):  <think>\n{reasoning}\n</think>\n\n{content}

Rows are kept only when reasoning_content has >= 40 words (pure-code rows are
already represented content-only in codespecialist and are NOT the target),
whole-record <= 4096 tokens (incl. bos/eos), deduped against the existing
pretrain corpora hash file and within the shard.

Run from repo root:
  PYTHONUTF8=1 python scripts/data_builder/build_cot_supplement_nemotron.py \
      --input dataset/competitive-coding/data/competitive_programming_python_00.jsonl \
      --parity 0 --out dataset/pretrain_cot_supplement/nemotron_cot_python_00_even.jsonl \
      --source nemotron_cot_python_00
"""
import argparse
import hashlib
import json
import os
import time

from tokenizers import Tokenizer

HASH_FILE = "tmp/pretrain_corpus_hashes.txt"
REPORT_DIR = "dataset/pretrain_cot_supplement/reports"
MIN_REASONING_WORDS = 40
MAX_TOKENS = 4096
LICENSE = "cc-by-4.0/odc-by/mit (Nemotron-Competitive-Programming-v2)"


def render(user: str, reasoning: str, content: str) -> str:
    return (
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n" + reasoning.strip("\n") + "\n</think>\n\n"
        + content.lstrip("\n") + "<|im_end|>\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--parity", type=int, required=True, choices=[0, 1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()

    t0 = time.time()
    with open(HASH_FILE, encoding="utf-8") as f:
        corpus_hashes = {line.strip() for line in f if line.strip()}
    tok = Tokenizer.from_file("model/tokenizer.json")

    stats = {"rows_scanned": 0, "malformed": 0, "short_or_no_reasoning": 0,
             "discarded_overlength": 0, "dup_vs_existing_corpus": 0,
             "dup_in_shard": 0, "kept": 0, "output_text_tokens": 0}
    seen = set()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(args.input, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for idx, line in enumerate(f):
            if idx % 2 != args.parity:
                continue
            stats["rows_scanned"] += 1
            try:
                row = json.loads(line)
                user = next(m["content"] for m in row["messages"] if m.get("role") == "user")
                asst = next(m for m in row["messages"] if m.get("role") == "assistant")
            except (StopIteration, KeyError, AttributeError, json.JSONDecodeError, TypeError):
                stats["malformed"] += 1
                continue
            if not isinstance(user, str) or not isinstance(asst.get("content"), str):
                stats["malformed"] += 1
                continue
            reasoning = (asst.get("reasoning_content") or "").strip()
            if len(reasoning.split()) < MIN_REASONING_WORDS:
                stats["short_or_no_reasoning"] += 1
                continue
            text = render(user, reasoning, asst["content"])
            n_tok = len(tok.encode(text).ids) + 2
            if n_tok > MAX_TOKENS:
                stats["discarded_overlength"] += 1
                continue
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            if h in corpus_hashes:
                stats["dup_vs_existing_corpus"] += 1
                continue
            if h in seen:
                stats["dup_in_shard"] += 1
                continue
            seen.add(h)
            out.write(json.dumps({"text": text, "source": args.source,
                                  "license": LICENSE, "token_count": n_tok},
                                 ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["output_text_tokens"] += n_tok - 2
            if stats["rows_scanned"] % 20000 == 0:
                print(f"{args.source} parity{args.parity}: {stats['rows_scanned']} scanned, "
                      f"{stats['kept']} kept, {time.time()-t0:.0f}s", flush=True)

    report = {"output": args.out, "input": args.input, "parity": args.parity,
              "min_reasoning_words": MIN_REASONING_WORDS, **stats,
              "elapsed_s": round(time.time() - t0, 1)}
    base = os.path.splitext(os.path.basename(args.out))[0]
    with open(f"{REPORT_DIR}/{base}.report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
