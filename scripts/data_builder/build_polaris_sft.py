"""Assemble the Polaris editorial SFT dataset with verified subagent chain-of-thought.

Pipeline position: after build_polaris_corpus.py has produced
dataset/polaris_raw/polaris_editorial_pairs.jsonl and solving subagents have written
per-problem CoT files under dataset/polaris_raw/cot/{zh,en}/<id>.json.

Inputs
  dataset/polaris_raw/polaris_editorial_pairs.jsonl   (id, title, statement, solution)
  dataset/polaris_raw/cot/en_statements.json          {id: english statement}
  dataset/polaris_raw/cot/en_solutions.json           {id: english solution}   (optional)
  dataset/polaris_raw/cot/zh/<id>.json                {cot, solution, approach}
  dataset/polaris_raw/cot/en/<id>.json                {cot, solution, approach}
  dataset/polaris_raw/cot/verdicts.json               {id: {"zh": "keep"|"discard: reason",
                                                             "en": "keep"|"discard: reason"}}

Record format (zh sample)
  user:      你是一名信息学竞赛选手。请独立解决下面这道竞赛题...
  assistant: <think>\n<verified subagent CoT>\n</think>\n\n<official editorial>

Only problems whose verdict is "keep" in a given language are emitted for that
language; the English assistant answer uses the translated official editorial from
en_solutions.json. Outputs + identity-keyword audit + SHA-256 land in a provenance
report, per AGENTS.md dataset identity hygiene.

Usage
  python scripts/data_builder/build_polaris_sft.py                # assemble
  python scripts/data_builder/build_polaris_sft.py --list-pending # ids missing CoT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("dataset/polaris_raw")
PAIRS = RAW / "polaris_editorial_pairs.jsonl"
COT = RAW / "cot"
EN_STATEMENTS = COT / "en_statements.json"
EN_SOLUTIONS = COT / "en_solutions.json"
VERDICTS = COT / "verdicts.json"
OUT_RAW = Path("dataset/sft_polaris_cot_raw.jsonl")
OUT_REPORT = RAW / "polaris_sft_provenance.json"

IDENTITY_KEYWORDS = [
    "qwen", "通义千问", "千问", "alibaba cloud", "阿里云", "阿里巴巴",
    "通义万相", "通义听悟",
]

ZH_USER_TMPL = (
    "你是一名信息学竞赛选手。请独立解决下面这道竞赛题，"
    "给出完整的解题思路与最终解法。\n\n## 题目\n\n{statement}"
)
EN_USER_TMPL = (
    "You are a competitive programming contestant. Solve the following problem "
    "independently and give your complete reasoning and final solution."
    "\n\n## Problem\n\n{statement}"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_cot_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"invalid json: {e}"}
    for key in ("cot", "solution", "approach"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            return {"_error": f"missing/empty field: {key}"}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-pending", action="store_true",
                        help="list pair ids that have no CoT file yet and exit")
    parser.add_argument("--out", default=str(OUT_RAW))
    args = parser.parse_args()

    pairs = load_jsonl(PAIRS)
    by_id = {p["id"]: p for p in pairs}

    if args.list_pending:
        for p in pairs:
            zh_ok = (COT / "zh" / f"{p['id']}.json").exists()
            en_ok = (COT / "en" / f"{p['id']}.json").exists()
            if not (zh_ok and en_ok):
                miss = [] + ([] if zh_ok else ["zh"]) + ([] if en_ok else ["en"])
                print(f"{p['id']}  {p['title']}  missing={','.join(miss)}")
        return

    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.exists() else {}
    en_statements = json.loads(EN_STATEMENTS.read_text(encoding="utf-8")) if EN_STATEMENTS.exists() else {}
    en_solutions = json.loads(EN_SOLUTIONS.read_text(encoding="utf-8")) if EN_SOLUTIONS.exists() else {}

    records: list[dict] = []
    stats = {"zh": {"keep": 0, "discard": 0, "no_cot": 0, "bad_cot": 0},
             "en": {"keep": 0, "discard": 0, "no_cot": 0, "bad_cot": 0, "no_translation": 0}}
    discard_reasons: dict[str, list[str]] = {}

    for p in pairs:
        pid = p["id"]
        v = verdicts.get(pid, {})
        zh_state, en_state = v.get("zh"), v.get("en")

        # --- Chinese record: subagent CoT + official editorial ---
        zh_cot = load_cot_file(COT / "zh" / f"{pid}.json")
        if zh_state and str(zh_state).startswith("discard"):
            stats["zh"]["discard"] += 1
            discard_reasons.setdefault(pid, []).append(f"zh: {zh_state}")
        elif zh_cot is None:
            stats["zh"]["no_cot"] += 1
        elif "_error" in zh_cot:
            stats["zh"]["bad_cot"] += 1
            discard_reasons.setdefault(pid, []).append(f"zh cot file: {zh_cot['_error']}")
        elif zh_state != "keep":
            stats["zh"]["no_cot"] += 1  # solved but not yet verified
        else:
            assistant = f"<think>\n{zh_cot['cot'].strip()}\n</think>\n\n{p['solution'].strip()}"
            records.append({
                "lang": "zh", "id": pid, "title": p["title"], "source": "polaris",
                "conversations": [
                    {"role": "user", "content": ZH_USER_TMPL.format(statement=p["statement"])},
                    {"role": "assistant", "content": assistant},
                ],
            })
            stats["zh"]["keep"] += 1

        # --- English record: subagent CoT + translated official editorial ---
        en_cot = load_cot_file(COT / "en" / f"{pid}.json")
        if en_state and str(en_state).startswith("discard"):
            stats["en"]["discard"] += 1
            discard_reasons.setdefault(pid, []).append(f"en: {en_state}")
        elif en_cot is None:
            stats["en"]["no_cot"] += 1
        elif "_error" in en_cot:
            stats["en"]["bad_cot"] += 1
            discard_reasons.setdefault(pid, []).append(f"en cot file: {en_cot['_error']}")
        elif en_state != "keep":
            stats["en"]["no_cot"] += 1
        elif pid not in en_statements:
            stats["en"]["no_translation"] += 1
            discard_reasons.setdefault(pid, []).append("en: statement translation missing")
        elif pid not in en_solutions:
            stats["en"]["no_translation"] += 1
            discard_reasons.setdefault(pid, []).append("en: solution translation missing")
        else:
            assistant = f"<think>\n{en_cot['cot'].strip()}\n</think>\n\n{en_solutions[pid].strip()}"
            records.append({
                "lang": "en", "id": pid, "title": p["title"], "source": "polaris",
                "conversations": [
                    {"role": "user", "content": EN_USER_TMPL.format(statement=en_statements[pid])},
                    {"role": "assistant", "content": assistant},
                ],
            })
            stats["en"]["keep"] += 1

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Identity audit over the trainable text (user+assistant content)
    hits: dict[str, int] = {}
    for rec in records:
        blob = (rec["conversations"][0]["content"] + rec["conversations"][1]["content"]).lower()
        for kw in IDENTITY_KEYWORDS:
            if kw in blob:
                hits[kw] = hits.get(kw, 0) + 1

    clean_out = None
    if not hits:
        clean_out = out.with_name(out.stem.replace("_raw", "_identity_clean") + out.suffix)
        clean_out.write_text(out.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    pending = [
        p["id"] for p in pairs
        if not ((COT / "zh" / f"{p['id']}.json").exists() and (COT / "en" / f"{p['id']}.json").exists())
    ]

    report = {
        "source": {
            "site": "https://polaris.xinxingshe.cn:444",
            "base_corpus": str(PAIRS),
            "cot_generation": "subagents solved each problem from the statement only "
                              "(editorial hidden); kept only when the derived approach "
                              "matches the official editorial (verdicts.json)",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "pairs_total": len(pairs),
            "records_written": len(records),
            "zh": stats["zh"],
            "en": stats["en"],
            "pending_pairs": len(pending),
        },
        "identity_keyword_hits": hits,
        "discard_reasons": discard_reasons,
        "outputs": {
            "sft_jsonl": str(out),
            "sft_jsonl_sha256": sha256_file(out),
            "identity_clean_jsonl": str(clean_out) if clean_out else None,
            "identity_clean_jsonl_sha256": sha256_file(clean_out) if clean_out else None,
        },
        "notes": [
            "assistant content embeds <think> CoT before the editorial answer; the CoT is an "
            "independent subagent solve kept only on approach match with the official editorial.",
            "identity_clean is emitted only when the keyword audit is empty.",
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"identity hits: {hits or 'none'}")
    print(f"wrote {out}" + (f" and {clean_out}" if clean_out else ""))
    print(f"provenance: {OUT_REPORT}")


if __name__ == "__main__":
    sys.exit(main())
