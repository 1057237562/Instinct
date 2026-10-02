# -*- coding: utf-8 -*-
"""Clean BAAI/TACO (dataset/taco/_hf_raw) into taco.clean.jsonl — RLVR/SFT-ready.

Cleaning rules:
- input_output must parse as JSON with non-empty inputs AND outputs
- rows with picture_num > 0 or <img> in the question -> excluded (text-only training)
- question HTML -> text
- solutions (python) exact-deduped, <=8 per row
- within-dataset dedup on normalized question text
- cross-dataset dedup against code_contests (canonical CF source) via
  tmp/cc_norm_hashes.txt written by clean_code_contests.py

Run from repo root:  PYTHONUTF8=1 python scripts/data_builder/clean_taco.py
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_to_text import html_to_text, norm_hash_key  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

SRC_GLOB = "dataset/taco/_hf_raw/ALL/*.parquet"
OUT = "dataset/taco/taco.clean.jsonl"
EXCLUDED = "dataset/taco/taco.excluded.jsonl"
REPORT = "dataset/taco/taco.clean.report.json"
CC_HASHES = "tmp/cc_norm_hashes.txt"
MAX_SOL = 8


def dedup_list(items, cap):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
            if len(out) >= cap:
                break
    return out


def main():
    t0 = time.time()
    cc_hashes = set()
    if os.path.exists(CC_HASHES):
        with open(CC_HASHES, encoding="utf-8") as f:
            cc_hashes = {line.strip() for line in f if line.strip()}
    files = sorted(glob.glob(SRC_GLOB))
    kept = excluded = 0
    reasons = {}
    stats = {
        "rows": 0, "fn_call": 0, "stdio": 0, "html_cleaned": 0,
        "solutions_kept": 0, "solutions_empty_rows": 0, "dup_in_dataset": 0,
        "dup_vs_code_contests": 0, "difficulty": {},
    }
    seen = set()
    with open(OUT, "w", encoding="utf-8", newline="\n") as out, \
         open(EXCLUDED, "w", encoding="utf-8", newline="\n") as ex:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=128):
                for r in batch.to_pylist():
                    stats["rows"] += 1
                    ident = {"source": r["source"], "name": r["name"], "url": r["url"]}

                    def drop(reason, **extra):
                        nonlocal excluded
                        excluded += 1
                        reasons[reason] = reasons.get(reason, 0) + 1
                        ex.write(json.dumps({"reason": reason, **extra, **ident}, ensure_ascii=False) + "\n")

                    pn = str(r["picture_num"] or "").strip()
                    if pn and pn != "0":
                        drop("image_in_statement", picture_num=pn)
                        continue
                    try:
                        io = json.loads(r["input_output"] or "{}")
                    except Exception:
                        drop("bad_input_output_json")
                        continue
                    ins, outs = io.get("inputs") or [], io.get("outputs") or []
                    if not ins or not outs or len(ins) != len(outs):
                        drop("no_or_mismatched_tests")
                        continue

                    q_text, n_images = html_to_text(r["question"])
                    if n_images:
                        drop("image_in_statement", n_images=n_images)
                        continue
                    if not q_text.strip():
                        drop("empty_question")
                        continue
                    if r["question"] and ("<" in r["question"] and ">" in r["question"]):
                        stats["html_cleaned"] += 1

                    key = norm_hash_key(q_text)
                    if key in seen:
                        stats["dup_in_dataset"] += 1
                        drop("dup_in_dataset")
                        continue
                    if key in cc_hashes:
                        stats["dup_vs_code_contests"] += 1
                        drop("dup_of_code_contests")
                        continue
                    seen.add(key)

                    try:
                        sols = dedup_list(json.loads(r["solutions"] or "[]"), MAX_SOL)
                    except Exception:
                        sols = []
                    if not sols:
                        stats["solutions_empty_rows"] += 1

                    starter = r["starter_code"] or ""
                    fn_name = io.get("fn_name")
                    row = {
                        "question": q_text,
                        "source": r["source"],
                        "difficulty": r["difficulty"],
                        "url": r["url"],
                        "tags": r["tags"],
                        "raw_tags": r["raw_tags"],
                        "skill_types": r["skill_types"],
                        "starter_code": starter,
                        "is_fn_call": bool(fn_name or starter.strip()),
                        "fn_name": fn_name,
                        "tests": {"inputs": ins, "outputs": outs},
                        "n_tests": len(ins),
                        "solutions": sols,
                    }
                    stats["fn_call" if row["is_fn_call"] else "stdio"] += 1
                    stats["solutions_kept"] += len(sols)
                    d = row["difficulty"] or "UNKNOWN"
                    stats["difficulty"][d] = stats["difficulty"].get(d, 0) + 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
            print(f"{os.path.basename(path)} done, kept={kept} excluded={excluded} {time.time()-t0:.0f}s", flush=True)

    report = {
        "output": OUT, "excluded": EXCLUDED, "source_glob": SRC_GLOB,
        "input_rows": stats["rows"], "kept_rows": kept, "excluded_rows": excluded,
        "exclusion_reasons": reasons, "stats": stats,
        "cross_dedup_against": "dataset/code_contests/code_contests.clean.jsonl (normalized statement text; CC rows are canonical)",
        "chain_of_thought_audit": "solutions field is a JSON list of raw python code; no reasoning field; prose-like rows 1/26443 -> NO CoT content",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
