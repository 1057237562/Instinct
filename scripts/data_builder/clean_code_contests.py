# -*- coding: utf-8 -*-
"""Clean deepmind/code_contests (dataset/code_contests/_hf_parquet) into
code_contests.clean.jsonl — one row per problem, RLVR/SFT-ready.

Cleaning rules:
- test groups validated (len(input) == len(output)); any mismatch -> excluded
- problems whose statement contains <img> -> excluded (text-only training)
- solutions filtered to python3 (lang 3) / cpp (lang 2), exact-deduped, <=8 each;
  python2 (lang 1) and java (lang 4) dropped (verified by syntax sampling)
- incorrect_solutions NOT carried into the clean file (counted only; stay in raw parquet)
- description HTML -> text; untranslated_description kept when the row is a
  machine translation (provenance)
- within-dataset dedup on (cf_contest_id, cf_index) else normalized title
- normalized-description hashes written to tmp/cc_norm_hashes.txt so the TACO
  cleaner can cross-dedup against this canonical source

Run from repo root:  PYTHONUTF8=1 python scripts/data_builder/clean_code_contests.py
"""
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_to_text import html_to_text, norm_hash_key  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

SRC_GLOB = "dataset/code_contests/_hf_parquet/data/*.parquet"
OUT = "dataset/code_contests/code_contests.clean.jsonl"
EXCLUDED = "dataset/code_contests/code_contests.excluded.jsonl"
REPORT = "dataset/code_contests/code_contests.clean.report.json"
HASH_OUT = "tmp/cc_norm_hashes.txt"

SOURCE_NAMES = ["unknown", "codechef", "codeforces", "hackerearth", "codejam", "atcoder", "aizu"]
DIFFICULTY_NAMES = ["unknown_difficulty", "easy", "medium", "hard", "harder", "hardest", "external"] + [chr(ord("A") + i) for i in range(22)]
LANG_PY3, LANG_CPP = 3, 2
MAX_SOL_PER_LANG = 8


def dedup_list(items, cap):
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
            if len(out) >= cap:
                break
    return out


def clean_tests(group):
    """Returns (tests_dict, n_cases) or (None, reason) on length mismatch."""
    if group is None:
        return {"input": [], "output": []}, 0
    ins, outs = group.get("input") or [], group.get("output") or []
    if len(ins) != len(outs):
        return None, "test_len_mismatch"
    return {"input": ins, "output": outs}, len(ins)


def main():
    t0 = time.time()
    files = sorted(glob.glob(SRC_GLOB))
    kept = excluded = 0
    reasons = {}
    keys_seen = {}
    desc_hashes = set()
    stats = {
        "rows": 0, "with_public": 0, "with_private": 0, "with_generated": 0,
        "file_io": 0, "translated_desc": 0, "desc_html_cleaned": 0,
        "solutions_py3": 0, "solutions_cpp": 0, "n_incorrect_total": 0,
        "dup_in_dataset": 0,
    }
    os.makedirs("tmp", exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as out, \
         open(EXCLUDED, "w", encoding="utf-8", newline="\n") as ex, \
         open(HASH_OUT, "w", encoding="utf-8", newline="\n") as hash_file:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=128):
                for r in batch.to_pylist():
                    stats["rows"] += 1
                    pid = f"{r['cf_contest_id']}{r['cf_index'] or ''}" if r["cf_contest_id"] else ""
                    ident = {"problem_id": pid, "name": r["name"], "source": SOURCE_NAMES[r["source"]]}
                    pid_key = (pid or norm_hash_key(r["name"])).lower()
                    if pid_key in keys_seen:
                        stats["dup_in_dataset"] += 1
                        excluded += 1
                        reasons["dup_in_dataset"] = reasons.get("dup_in_dataset", 0) + 1
                        ex.write(json.dumps({"reason": "dup_in_dataset", **ident}, ensure_ascii=False) + "\n")
                        continue

                    pub, n_pub = clean_tests(r["public_tests"])
                    priv, n_priv = clean_tests(r["private_tests"])
                    gen, n_gen = clean_tests(r["generated_tests"])
                    if pub is None or priv is None or gen is None:
                        excluded += 1
                        reasons["test_len_mismatch"] = reasons.get("test_len_mismatch", 0) + 1
                        ex.write(json.dumps({"reason": "test_len_mismatch", **ident}, ensure_ascii=False) + "\n")
                        continue
                    if n_pub + n_priv + n_gen == 0:
                        excluded += 1
                        reasons["no_tests"] = reasons.get("no_tests", 0) + 1
                        ex.write(json.dumps({"reason": "no_tests", **ident}, ensure_ascii=False) + "\n")
                        continue

                    desc_text, n_images = html_to_text(r["description"])
                    if n_images:
                        excluded += 1
                        reasons["image_in_statement"] = reasons.get("image_in_statement", 0) + 1
                        ex.write(json.dumps({"reason": "image_in_statement", "n_images": n_images, **ident}, ensure_ascii=False) + "\n")
                        continue
                    if not desc_text.strip():
                        excluded += 1
                        reasons["empty_statement"] = reasons.get("empty_statement", 0) + 1
                        ex.write(json.dumps({"reason": "empty_statement", **ident}, ensure_ascii=False) + "\n")
                        continue
                    if r["description"] and ("<" in r["description"] and ">" in r["description"]):
                        stats["desc_html_cleaned"] += 1

                    sols = r["solutions"] or {"language": [], "solution": []}
                    py3 = dedup_list([s for l, s in zip(sols["language"], sols["solution"]) if l == LANG_PY3], MAX_SOL_PER_LANG)
                    cpp = dedup_list([s for l, s in zip(sols["language"], sols["solution"]) if l == LANG_CPP], MAX_SOL_PER_LANG)
                    stats["solutions_py3"] += len(py3)
                    stats["solutions_cpp"] += len(cpp)
                    incorrect = r["incorrect_solutions"] or {"solution": []}
                    stats["n_incorrect_total"] += len(incorrect["solution"])

                    tl = r["time_limit"] or {}
                    time_limit_s = (tl.get("seconds", 0) + tl.get("nanos", 0) / 1e9) if tl else None
                    desc_hash = norm_hash_key(desc_text)
                    if desc_hash in desc_hashes:
                        stats["dup_in_dataset"] += 1
                        excluded += 1
                        reasons["dup_description_text"] = reasons.get("dup_description_text", 0) + 1
                        ex.write(json.dumps({"reason": "dup_description_text", **ident}, ensure_ascii=False) + "\n")
                        continue
                    desc_hashes.add(desc_hash)
                    hash_file.write(desc_hash + "\n")

                    keys_seen[pid_key] = True
                    row = {
                        "problem_id": pid or None,
                        "name": r["name"],
                        "source": SOURCE_NAMES[r["source"]],
                        "difficulty": DIFFICULTY_NAMES[r["difficulty"]] if 0 <= r["difficulty"] < len(DIFFICULTY_NAMES) else r["difficulty"],
                        "cf_rating": r["cf_rating"] or None,
                        "cf_tags": r["cf_tags"] or [],
                        "description": desc_text,
                        "description_is_translated": bool(r["is_description_translated"]),
                        "untranslated_description": r["untranslated_description"] if r["is_description_translated"] else None,
                        "time_limit_s": time_limit_s,
                        "memory_limit_mb": round(r["memory_limit_bytes"] / 1048576) if r["memory_limit_bytes"] else None,
                        "file_io": bool(r["input_file"] or r["output_file"]),
                        "tests": {"public": pub, "private": priv, "generated": gen},
                        "n_tests_public": n_pub, "n_tests_private": n_priv, "n_tests_generated": n_gen,
                        "solutions_python3": py3,
                        "solutions_cpp": cpp,
                        "n_incorrect_solutions": len(incorrect["solution"]),
                    }
                    stats["with_public"] += n_pub > 0
                    stats["with_private"] += n_priv > 0
                    stats["with_generated"] += n_gen > 0
                    stats["file_io"] += row["file_io"]
                    stats["translated_desc"] += row["description_is_translated"]
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
            print(f"{path.split('/')[-1]} done, kept={kept} excluded={excluded} {time.time()-t0:.0f}s", flush=True)

    report = {
        "output": OUT, "excluded": EXCLUDED, "source_glob": SRC_GLOB,
        "input_rows": stats["rows"], "kept_rows": kept, "excluded_rows": excluded,
        "exclusion_reasons": reasons, "stats": stats,
        "language_filter": {"python3": LANG_PY3, "cpp": LANG_CPP, "evidence": "syntax sampling: 1=python2(print 'x'), 2=cpp(#include), 3=python3, 4=java"},
        "chain_of_thought_audit": "solutions are raw submissions; no reasoning field; prose-like solutions 57/13610 rows (code comments) -> NO CoT content",
        "dedup_method": "(cf_contest_id, cf_index) else title; secondary normalized-description-text hash",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
