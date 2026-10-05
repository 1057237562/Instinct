"""AI-rewrite pipeline for the Code92007 blog SFT set.

The raw blog SFT rows (auto-generated user turn + verbatim article body) are turned
into natural SFT pairs by an LLM rewriting pass, one article at a time:

  prepare  — pack articles into character-budgeted batch files under
             raw/sft_ai_batches/ (body <= MAX_BODY_CHARS; longer articles stay in
             the pretrain set and are only reported)
  (agents) — rewriting agents process each batch one row at a time via
             scripts/data_builder/sft_ai_append.py, writing out_batch_NNN.jsonl
  merge    — validate every rewritten row (code blocks must survive verbatim,
             turn length bounds, identity scan), assemble
             code92007_blog_sft_ai.jsonl + .meta.jsonl + sft_ai.report.json

Usage:
  python scripts/data_builder/build_code92007_sft_ai.py prepare
  python scripts/data_builder/build_code92007_sft_ai.py merge
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "dataset" / "code92007"
BATCH_DIR = OUT_DIR / "raw" / "sft_ai_batches"
ARTICLES_PATH = OUT_DIR / "code92007_blog_articles.jsonl"

MAX_BODY_CHARS = 24000
MIN_BODY_CHARS = 300
BATCH_CHAR_BUDGET = 60000
BATCH_MAX_ROWS = 25

MIN_USER_CHARS, MAX_USER_CHARS = 10, 200
MIN_ASSISTANT_CHARS = 150

IDENTITY_KEYWORDS = (
    "qwen", "通义千问", "千问", "alibaba cloud", "阿里云", "阿里巴巴", "通义万相", "通义听悟",
)

CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_articles():
    if not ARTICLES_PATH.exists():
        sys.exit("code92007_blog_articles.jsonl missing — run build_code92007_corpus.py build")
    return [json.loads(line) for line in ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def prepare():
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("*.jsonl"):
        old.unlink()
    for old in BATCH_DIR.glob("out_batch_*.jsonl"):
        old.unlink()
    for old in BATCH_DIR.glob("*.txt"):
        old.unlink()

    articles = load_articles()
    eligible, skipped_long, skipped_short = [], [], []
    for row in articles:
        body = row["body"]
        if len(body) > MAX_BODY_CHARS:
            skipped_long.append({"id": row["id"], "title": row["title"][:60],
                                 "chars": len(body)})
        elif len(body) < MIN_BODY_CHARS:
            skipped_short.append({"id": row["id"], "chars": len(body)})
        else:
            eligible.append(row)

    batches, current, used = [], [], 0
    for row in eligible:
        cost = len(row["body"])
        if current and (used + cost > BATCH_CHAR_BUDGET or len(current) >= BATCH_MAX_ROWS):
            batches.append(current)
            current, used = [], 0
        current.append(row)
        used += cost
    if current:
        batches.append(current)

    manifest = []
    for number, rows in enumerate(batches):
        batch_path = BATCH_DIR / f"batch_{number:03d}.jsonl"
        with open(batch_path, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(
                    {"id": row["id"], "title": row["title"], "tags": row["tags"],
                     "url": row["url"], "body": row["body"]}, ensure_ascii=False) + "\n")
        manifest.append({"batch": number, "ids": [r["id"] for r in rows],
                         "rows": len(rows),
                         "chars": sum(len(r["body"]) for r in rows)})

    (BATCH_DIR / "manifest.json").write_text(json.dumps({
        "batches": manifest,
        "skipped_long": skipped_long,
        "skipped_short": skipped_short,
        "eligible_rows": len(eligible),
        "total_chars": sum(m["chars"] for m in manifest),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(batches)} batches, {len(eligible)} rows, "
          f"{len(skipped_long)} skipped long, {len(skipped_short)} skipped short")
    print(f"batches + manifest under {BATCH_DIR}")


def merge():
    manifest_path = BATCH_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit("manifest.json missing — run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    articles = {row["id"]: row for row in load_articles()}

    final_rows, meta_rows, failed_rows = [], [], []
    per_batch = Counter()
    scanned_units = []
    for entry in manifest["batches"]:
        number = entry["batch"]
        out_path = BATCH_DIR / f"out_batch_{number:03d}.jsonl"
        written = {}
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    failed_rows.append({"batch": number, "reason": "unparseable line",
                                        "line": line[:120]})
                    continue
                written[row["id"]] = row
        for article_id in entry["ids"]:
            row = written.get(article_id)
            source = articles[article_id]
            if row is None:
                failed_rows.append({"batch": number, "id": article_id,
                                    "reason": "missing — agent did not write it"})
                continue
            if row.get("skip_reason"):
                failed_rows.append({"batch": number, "id": article_id,
                                    "reason": f"agent skip: {row['skip_reason'][:100]}"})
                continue
            user, assistant = row.get("user", ""), row.get("assistant", "")
            reasons = []
            if not (MIN_USER_CHARS <= len(user) <= MAX_USER_CHARS):
                reasons.append(f"user length {len(user)}")
            if len(assistant) < MIN_ASSISTANT_CHARS:
                reasons.append(f"assistant length {len(assistant)}")
            body = source["body"]
            blocks = [m for m in CODE_FENCE.findall(body)]
            missing_code = [b[:60] for b in blocks if b not in assistant]
            if blocks and missing_code:
                reasons.append(f"{len(missing_code)}/{len(blocks)} code blocks altered")
            for keyword in IDENTITY_KEYWORDS:
                if keyword.lower() in (user + assistant).lower():
                    reasons.append(f"identity keyword: {keyword}")
                    break
            if reasons:
                failed_rows.append({"batch": number, "id": article_id,
                                    "reason": "; ".join(reasons)})
                continue
            final_rows.append({"conversations": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]})
            scanned_units.append((user + assistant))
            meta_rows.append({"id": article_id, "batch": number,
                              "url": source["url"], "title": source["title"],
                              "user_chars": len(user), "assistant_chars": len(assistant),
                              "code_blocks": len(blocks)})
            per_batch[number] += 1

    final_path = OUT_DIR / "code92007_blog_sft_ai.jsonl"
    meta_path = OUT_DIR / "code92007_blog_sft_ai.meta.jsonl"
    with open(final_path, "w", encoding="utf-8", newline="\n") as stream:
        for row in final_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as stream:
        for row in meta_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (BATCH_DIR / "failed.jsonl").write_text(
        json.dumps(failed_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    identity_hits = sorted({u for u in scanned_units
                            for k in IDENTITY_KEYWORDS if k.lower() in u.lower()})
    report = {
        "stage": "sft_ai",
        "batches": len(manifest["batches"]),
        "input_rows": sum(m["rows"] for m in manifest["batches"]),
        "kept_rows": len(final_rows),
        "failed_rows": len(failed_rows),
        "failed_detail_file": "raw/sft_ai_batches/failed.jsonl",
        "skipped_long": manifest["skipped_long"],
        "skipped_short_count": len(manifest["skipped_short"]),
        "per_batch_kept": dict(sorted(per_batch.items())),
        "identity_scan": {"keywords": list(IDENTITY_KEYWORDS),
                          "units_scanned": len(scanned_units),
                          "units_with_hits": len(identity_hits)},
        "outputs": {
            final_path.name: {"rows": len(final_rows), "sha256": sha256_file(final_path)},
            meta_path.name: {"rows": len(meta_rows), "sha256": sha256_file(meta_path)},
        },
    }
    (OUT_DIR / "sft_ai.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("batches", "input_rows", "kept_rows",
                                             "failed_rows")}, indent=1))
    print(f"report: {OUT_DIR / 'sft_ai.report.json'}")


if __name__ == "__main__":
    {"prepare": prepare, "merge": merge}[sys.argv[1]]()
