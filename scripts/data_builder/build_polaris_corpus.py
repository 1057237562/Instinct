"""Build a pretraining corpus from the Polaris (polaris.xinxingshe.cn) article knowledge base.

Logs in with the account given via --email/--password (or an existing cookie jar),
paginates GET /api/v1/articles?limit=32&offset=N, and writes:

  - dataset/polaris_raw/polaris_articles_full.jsonl  raw rows, all fields (source of truth)
  - dataset/pretrain_polaris_raw.jsonl               {"text": ...} pretrain rows (audited)
  - dataset/pretrain_polaris_identity_clean.jsonl    training artifact, emitted only when the
                                                     identity audit flags nothing
  - dataset/polaris_raw/polaris_provenance.json      counts, keyword audit hits, SHA-256

Identity hygiene (AGENTS.md): the keyword audit runs here and its hit counts land in
the provenance report, but hits are flagged for review -- rows are never auto-deleted.
Name the final training artifact with an explicit clean/identity_clean suffix only
after the flagged rows have been reviewed.

Usage:
  python scripts/data_builder/build_polaris_corpus.py \
    --email you@example.com --password '...' [--cookie-jar dataset/polaris_raw/cookies.txt]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://polaris.xinxingshe.cn:444"
PAGE_LIMIT = 32  # the API rejects any other limit
MAX_PAGES = 500  # safety cap

# AGENTS.md identity-contamination keywords (trainable-content check, not vocab checks)
IDENTITY_KEYWORDS = [
    "qwen", "通义千问", "千问", "alibaba cloud", "阿里云", "阿里巴巴",
    "通义万相", "通义听悟",
]


def load_cookies(cookie_jar: Path) -> str:
    cookie = ""
    for line in cookie_jar.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        if "\t" in line and not line.startswith("#"):
            parts = line.split("\t")
            cookie += f"{parts[5]}={parts[6]}; "
    return cookie


def login(email: str, password: str, cookie_jar: Path) -> str:
    """Fresh login; persists Netscape cookies to cookie_jar and returns the Cookie header.

    The server requires an Origin/Referer check (403 invalid_origin otherwise) and sets
    both __Host-oi_session and __Host-oi_csrf on the login response itself, so no prior
    GET is needed. Each successful login appears to revoke earlier sessions of the user.
    """
    body = json.dumps({"email": email, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/auth/login",
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/login",
            "User-Agent": "polaris-corpus-builder/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            set_cookies = resp.headers.get_all("Set-Cookie") or []
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        raise SystemExit(f"login failed: HTTP {e.code} {detail}") from e
    if "userId" not in payload.get("data", {}):
        raise SystemExit(f"login rejected: {payload}")

    host = BASE_URL.split("//", 1)[1]
    lines = []
    for sc in set_cookies:
        pair = sc.split(";", 1)[0]
        name, value = pair.split("=", 1)
        httponly = "HttpOnly" in sc
        lines.append(
            f"{'#HttpOnly_' if httponly else ''}{host}\tFALSE\t/\tTRUE\t0\t{name}\t{value}"
        )
    cookie_jar.parent.mkdir(parents=True, exist_ok=True)
    cookie_jar.write_text(
        "# Netscape HTTP Cookie File\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    return load_cookies(cookie_jar)


def fetch_page(cookie: str, offset: int) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/articles?limit={PAGE_LIMIT}&offset={offset}",
        headers={"Cookie": cookie, "Origin": BASE_URL, "Referer": f"{BASE_URL}/articles"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--cookie-jar", default="dataset/polaris_raw/cookies.txt")
    parser.add_argument("--raw-out", default="dataset/polaris_raw/polaris_articles_full.jsonl")
    parser.add_argument("--text-out", default="dataset/pretrain_polaris_raw.jsonl")
    parser.add_argument("--report-out", default="dataset/polaris_raw/polaris_provenance.json")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds between page fetches")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="rebuild outputs from an existing --raw-out instead of refetching (no login)",
    )
    args = parser.parse_args()

    cookie_jar = Path(args.cookie_jar)
    raw_out = Path(args.raw_out)
    if args.skip_fetch:
        rows = {}
        for line in raw_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
        page = 0
        print(f"loaded {len(rows)} rows from {raw_out}")
    else:
        cookie = login(args.email, args.password, cookie_jar)
        print(f"login ok (cookie jar: {cookie_jar})")

        rows: dict[str, dict] = {}
        offset, page = 0, 0
        while True:
            payload = fetch_page(cookie, offset)
            data = payload.get("data", [])
            meta = payload.get("meta", {})
            for row in data:
                rows[row["id"]] = row
            page += 1
            print(f"page {page}: offset={offset} rows={len(data)} total={len(rows)} hasNext={meta.get('hasNext')}")
            if not meta.get("hasNext"):
                break
            offset += PAGE_LIMIT
            if page >= MAX_PAGES:
                raise SystemExit(f"hit MAX_PAGES={MAX_PAGES}; aborting to avoid a runaway loop")
            time.sleep(args.delay)

    published = [r for r in rows.values() if r.get("status") == "published"]
    skipped = [r for r in rows.values() if r.get("status") != "published"]

    # Deduplicate identical title+markdown bodies (same body under a second sourceHash)
    seen_body: set[str] = set()
    deduped: list[dict] = []
    for row in sorted(published, key=lambda r: r["id"]):
        digest = hashlib.sha256(
            (row["title"] + "\n" + row["markdown"]).encode("utf-8")
        ).hexdigest()
        if digest in seen_body:
            continue
        seen_body.add(digest)
        deduped.append(row)
    duplicate_bodies_removed = len(published) - len(deduped)

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w", encoding="utf-8", newline="\n") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    text_out = Path(args.text_out)
    with text_out.open("w", encoding="utf-8", newline="\n") as f:
        for row in deduped:
            text = f"{row['title']}\n\n{row['markdown']}".strip() + "\n"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    # Identity audit (keyword scan over trainable text; hits are review flags, not deletions)
    hits: dict[str, list[dict]] = {}
    for row in deduped:
        blob = f"{row['title']}\n{row['summary'] or ''}\n{row['markdown']}".lower()
        for kw in IDENTITY_KEYWORDS:
            if kw in blob:
                hits.setdefault(kw, []).append({"id": row["id"], "title": row["title"], "slug": row["slug"]})
    flagged_ids = {r["id"] for rows_ in hits.values() for r in rows_}

    # With zero flagged rows the audited corpus doubles as the identity_clean artifact;
    # otherwise stop for review instead of emitting a training artifact.
    clean_out: Path | None = None
    if not flagged_ids:
        clean_out = text_out.with_name(text_out.stem.replace("_raw", "_identity_clean") + text_out.suffix)
        clean_out.write_text(text_out.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    dup_by_source_hash: dict[str, int] = {}
    for row in deduped:
        key = row.get("sourceHash") or ""
        dup_by_source_hash[key] = dup_by_source_hash.get(key, 0) + 1
    duplicate_source_hashes = {k: v for k, v in dup_by_source_hash.items() if v > 1}

    report = {
        "source": {
            "site": BASE_URL,
            "endpoint": "/api/v1/articles?limit=32&offset=N",
            "license": "site content; review before redistribution",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "counts": {
            "unique_ids": len(rows),
            "published": len(published),
            "non_published_skipped": len(skipped),
            "duplicate_bodies_removed": duplicate_bodies_removed,
            "duplicate_source_hash_groups": len(duplicate_source_hashes),
            "pages": page,
        },
        "identity_keyword_hits": {k: v for k, v in sorted(hits.items())},
        "identity_hit_rows": sorted(flagged_ids),
        "outputs": {
            "raw_jsonl": str(raw_out),
            "raw_jsonl_sha256": sha256_file(raw_out),
            "text_jsonl": str(text_out),
            "text_jsonl_sha256": sha256_file(text_out),
            "text_chars": sum(len(json.loads(l)["text"]) for l in text_out.open(encoding="utf-8")),
            "identity_clean_jsonl": str(clean_out) if clean_out else None,
            "identity_clean_jsonl_sha256": sha256_file(clean_out) if clean_out else None,
        },
        "notes": [
            "pretrain_polaris_raw.jsonl is the audited conversion; identity_clean is emitted "
            "only when identity_hit_rows is empty. Flagged rows must be reviewed (dropped or "
            "rewritten) before any clean artifact is named/trained on.",
        ],
    }
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"identity keyword groups with hits: {len(hits)} -> see {args.report_out}")
    print(f"wrote {raw_out}, {text_out}, {args.report_out}")


if __name__ == "__main__":
    sys.exit(main())
