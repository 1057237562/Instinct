# -*- coding: utf-8 -*-
"""Dedup pretrain_continue.jsonl against the FINAL training file codespecialist.jsonl.

The existing pretrain_continue.report.json only guarantees novel rows are absent
from the PRIOR corpus (pretrain_codespecialist.jsonl, 1.46M rows). codespecialist.jsonl
(3.5M rows) additionally contains stack/general/math/academic/competitive components
that were not part of that prior corpus, so a second exact-dedup pass is required
before mixing the two files into one training run.

Methodology matches the repo convention: global exact text dedup via 128-bit hash
(blake2b-16 here, same collision-safety class as the SHA-256 used in the reports).
Rows are written out byte-identical (original line preserved).
"""
import collections
import hashlib
import json
import sys
import time

BASE = "dataset"  # run from repo root
CS = f"{BASE}/codespecialist.jsonl"
PC = f"{BASE}/pretrain_continue.jsonl"
OUT = f"{BASE}/pretrain_continue.dedup.jsonl"
REPORT = f"{BASE}/pretrain_continue.dedup.report.json"


def h(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


t0 = time.time()
seen = set()
with open(CS, encoding="utf-8") as f:
    for i, line in enumerate(f):
        seen.add(h(json.loads(line)["text"]))
        if (i + 1) % 500_000 == 0:
            print(f"pass1: {i+1} rows, {len(seen)} unique, {time.time()-t0:.0f}s", flush=True)
print(f"pass1 done: {len(seen)} unique hashes in codespecialist.jsonl ({time.time()-t0:.0f}s)", flush=True)

kept = dup = 0
kept_tok = dup_tok = 0
replay_rows = replay_dup = 0
comp = collections.defaultdict(lambda: {"kept_rows": 0, "kept_tokens": 0, "dup_rows": 0, "dup_tokens": 0})

t1 = time.time()
with open(PC, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8", newline="\n") as out:
    for i, line in enumerate(f):
        r = json.loads(line)
        is_replay = "prior_source" in (r.get("metadata") or {})
        if is_replay:
            replay_rows += 1
        src = r.get("source", "unknown")
        if h(r["text"]) in seen:
            dup += 1
            dup_tok += r["token_count"]
            comp[src]["dup_rows"] += 1
            comp[src]["dup_tokens"] += r["token_count"]
            if is_replay:
                replay_dup += 1
        else:
            kept += 1
            kept_tok += r["token_count"]
            comp[src]["kept_rows"] += 1
            comp[src]["kept_tokens"] += r["token_count"]
            out.write(line)
        if (i + 1) % 200_000 == 0:
            print(f"pass2: {i+1} rows, kept {kept}, dup {dup}, {time.time()-t1:.0f}s", flush=True)

report = {
    "output": OUT,
    "dedup_against": CS,
    "method": "global exact text dedup (blake2b-128 of text field)",
    "input_rows": kept + dup,
    "input_tokens": kept_tok + dup_tok,
    "kept_rows": kept,
    "kept_tokens": kept_tok,
    "dropped_rows": dup,
    "dropped_tokens": dup_tok,
    "replay_rows_total": replay_rows,
    "replay_rows_dropped": replay_dup,
    "novel_rows_dropped": dup - replay_dup,
    "components": dict(comp),
}
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in report.items() if k != "components"}, indent=2))
print("per-component:")
for k, v in sorted(comp.items()):
    print(f"  {k}: kept {v['kept_rows']}/{v['kept_rows']+v['dup_rows']} rows, {v['kept_tokens']/1e6:.1f}M tokens; dup {v['dup_rows']} rows / {v['dup_tokens']/1e6:.1f}M")
print(f"done in {time.time()-t0:.0f}s")
