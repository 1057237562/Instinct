"""Build a training corpus from blogger Code92007 (CSDN "小衣同学") blog + GitHub code.

Sources:
  - CSDN blog  https://blog.csdn.net/Code92007  (~1013 articles, competitive-programming
    editorials, data-structure/DP/graph notes). Articles are fetched through the public
    home-api list endpoint + article pages; raw HTML is cached gzipped under
    dataset/code92007/raw/articles/ so the build is resumable and re-runnable offline.
  - GitHub non-fork repos of user Code92007 (cpc, cf-tuxun, tijing, ...), shallow-cloned
    into dataset/code92007/raw/repos/.

Outputs (dataset/code92007/):
  code92007_blog_pretrain.jsonl  {"text": title + markdown body} per original article
  code92007_blog_sft.jsonl       {"conversations": [user=topic question, assistant=body]}
  code92007_code_pretrain.jsonl  {"text": repo source files, path-comment header}
  build.report.json              provenance, per-stage counts, identity scan, SHA-256

Stages (each resumable):
  python scripts/data_builder/build_code92007_corpus.py fetch-index
  python scripts/data_builder/build_code92007_corpus.py fetch-articles [--workers 4]
  python scripts/data_builder/build_code92007_corpus.py fetch-repos
  python scripts/data_builder/build_code92007_corpus.py build

Identity hygiene (AGENTS.md): the corpus is scanned for model-identity keywords; hits are
recorded as review candidates in the report, never auto-rewritten. Blog/repo content is
copyrighted by the author — research use only, no redistribution.
"""
import argparse
import gzip
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "dataset" / "code92007"
RAW_DIR = OUT_DIR / "raw"
ARTICLE_DIR = RAW_DIR / "articles"
REPO_DIR = RAW_DIR / "repos"

CSDN_USER = "Code92007"
LIST_API = (
    "https://blog.csdn.net/community/home-api/v1/get-business-list"
    "?page={page}&size=100&businessType=blog&orderby=&noMore=false&year=&month="
    "&username=" + CSDN_USER
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": f"https://blog.csdn.net/{CSDN_USER}",
}

GITHUB_USER = "Code92007"
# Forks and the empty Mytest repo are excluded — only the author's own code.
REPO_EXCLUDE = {"Mytest"}
MIN_CODE_FILE_BYTES = 40
MAX_CODE_FILE_BYTES = 80 * 1024
CODE_EXTENSIONS = {
    ".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hh", ".hpp",
    ".js", ".jsx", ".ts", ".tsx", ".vue",
    ".html", ".css", ".scss", ".less",
    ".md", ".sh", ".bat", ".ps1", ".sql", ".yml", ".yaml", ".toml",
}
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", "dist", "build", "vendor",
    "target", ".next", "coverage", ".idea", ".vscode",
}

IDENTITY_KEYWORDS = (
    "qwen", "通义千问", "千问", "alibaba cloud", "阿里云", "阿里巴巴", "通义万相", "通义听悟",
)

MIN_SFT_BODY_CHARS = 300
MIN_PRETRAIN_CHARS = 200

C_LIKE_COMMENT = {".cpp", ".cc", ".cxx", ".c", ".h", ".hh", ".hpp", ".js", ".jsx",
                  ".ts", ".tsx", ".css", ".scss", ".less", ".sql"}
HASH_COMMENT = {".py", ".sh", ".yml", ".yaml", ".toml"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: article index
# ---------------------------------------------------------------------------

def fetch_index():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    articles = []
    page = 1
    session = requests.Session()
    session.headers.update(HEADERS)
    while True:
        response = session.get(LIST_API.format(page=page), timeout=25)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 200:
            raise RuntimeError(f"list api error: {payload.get('message')}")
        items = payload["data"]["list"]
        if not items:
            break
        articles.extend(items)
        total = payload["data"]["total"]
        print(f"page {page}: {len(items)} items (total {total})")
        if len(articles) >= total or page > 30:
            break
        page += 1
        time.sleep(0.5)
    index_path = RAW_DIR / "articles_index.json"
    index_path.write_text(json.dumps(articles, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {index_path} with {len(articles)} entries")


# ---------------------------------------------------------------------------
# Stage 2: article pages
# ---------------------------------------------------------------------------

def fetch_article(session, article_id, url):
    path = ARTICLE_DIR / f"{article_id}.html.gz"
    if path.exists() and path.stat().st_size > 0:
        return "cached"
    last_status = None
    # CSDN throttles bursts with 521 / TLS resets; back off progressively and
    # let the caller's cooldown handle an active block
    for attempt, wait in enumerate((0, 8, 20, 40), start=1):
        if wait:
            time.sleep(wait)
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200 and "content_views" in response.text:
                path.write_bytes(gzip.compress(response.text.encode("utf-8")))
                return "fetched"
            last_status = response.status_code
        except requests.RequestException as exc:
            last_status = repr(exc)[:120]
    return f"failed:{last_status}"


def fetch_articles(workers):
    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / "articles_index.json"
    if not index_path.exists():
        sys.exit("articles_index.json missing — run fetch-index first")
    articles = json.loads(index_path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update(HEADERS)

    # CSDN bans by IP at the TLS level when it sees a burst. Probe until the
    # ban lifts (checks every 5 min, gives up after 8 h) before crawling.
    probe_url = next(
        (a["url"] for a in articles
         if not (ARTICLE_DIR / f"{a['articleId']}.html.gz").exists()),
        None)
    if probe_url is None:
        print("all articles already cached — nothing to fetch")
        return
    for attempt in range(96):
        try:
            response = session.get(probe_url, timeout=30)
            ok = response.status_code == 200 and "content_views" in response.text
        except requests.RequestException as exc:
            ok = False
            print(f"probe: {type(exc).__name__}", flush=True)
        if ok:
            print("probe OK — unblocked, starting crawl", flush=True)
            break
        if attempt == 95:
            print("still blocked after 8 h of probes — giving up; "
                  "re-run fetch-articles later (resumable)", flush=True)
            return
        time.sleep(300)
    # prime cookies once (the list API works without them, article pages are pickier)
    try:
        session.get(f"https://blog.csdn.net/{CSDN_USER}", timeout=30)
    except requests.RequestException:
        pass

    results = Counter()
    failures = []
    consecutive_fails = 0
    pending = [a for a in articles
               if not (ARTICLE_DIR / f"{a['articleId']}.html.gz").exists()]
    for done, article in enumerate(pending, start=1):
        outcome = fetch_article(session, article["articleId"], article["url"])
        results[outcome] += 1
        if outcome.startswith("failed"):
            failures.append({"articleId": article["articleId"], "url": article["url"],
                             "reason": outcome})
            consecutive_fails += 1
            cooldown = min(90, 10 * consecutive_fails)
            print(f"{done}/{len(pending)} FAILED {outcome} — cooldown {cooldown}s",
                  flush=True)
            time.sleep(cooldown)
            if consecutive_fails >= 12:
                print("aborting run: still blocked after repeated cooldowns; "
                      "wait and re-run fetch-articles (resumable)", flush=True)
                break
        else:
            consecutive_fails = 0
            time.sleep(2.0)
        if done % 25 == 0:
            print(f"{done}/{len(pending)} done, {dict(results)}", flush=True)
    (RAW_DIR / "fetch_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"finished: {dict(results)}; failures listed in fetch_failures.json")


# ---------------------------------------------------------------------------
# Stage 3: GitHub repos
# ---------------------------------------------------------------------------

def fetch_repos():
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://api.github.com/users/{GITHUB_USER}/repos?per_page=100",
                            timeout=25, headers={"User-Agent": HEADERS["User-Agent"]})
    response.raise_for_status()
    repos = [r["name"] for r in response.json()
             if not r["fork"] and r["name"] not in REPO_EXCLUDE and r["size"] > 0]
    print(f"repos to clone: {repos}")
    for name in repos:
        dest = REPO_DIR / name
        if dest.exists():
            print(f"{name}: already present")
            continue
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "-q",
             f"https://github.com/{GITHUB_USER}/{name}.git", str(dest)],
            capture_output=True, text=True)
        if proc.returncode == 0:
            shutil.rmtree(dest / ".git", ignore_errors=True)
            print(f"{name}: cloned")
        else:
            print(f"{name}: CLONE FAILED {proc.stderr.strip()[:200]}")


# ---------------------------------------------------------------------------
# HTML -> markdown-ish text (stdlib only; CSDN wraps content in div#content_views,
# where code blocks are <pre><ol><li><code>line</code></li></ol></pre>)
# ---------------------------------------------------------------------------

class ContentExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "svg", "iframe", "button", "input", "select", "video"}
    BLOCK_TAGS = {"p", "div", "section", "blockquote", "table", "tr", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth_in_target = 0
        self.target_depth = None
        self.open_tags = []
        self.out = []
        self.img_count = 0
        self._in_pre = False
        self._pre_lines = False  # <pre><ol><li>: each li is one code line
        self._in_li = False
        self._pending = []

    @staticmethod
    def _is_content_views(attrs):
        as_dict = dict(attrs)
        return as_dict.get("id") == "content_views"

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            if self.depth_in_target:
                self.depth_in_target += 1
            elif self._is_content_views(attrs):
                self.depth_in_target = 1
            return
        if not self.depth_in_target:
            return
        if tag in self.SKIP_TAGS:
            self.open_tags.append(tag)
            return
        if tag == "pre":
            self._in_pre = True
            self._pre_lines = False
            self.out.append("\n```")
            return
        if tag == "ol" and self._in_pre:
            self._pre_lines = True
            return
        if tag == "li":
            if self._in_pre:
                self._in_li = True
                self._pending = []  # each li is one code line
            else:
                self.out.append("\n- ")
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self.out.append(f"\n\n{'#' * level} ")
            return
        if tag == "br":
            if self._in_pre:
                if self._in_li:
                    self._pending.append("\n")
            else:
                self.out.append("\n")
            return
        if tag == "img":
            self.img_count += 1
            return
        if tag == "code" and not self._in_pre:
            self.out.append("`")
            return
        if tag == "table" or tag == "tr":
            self.out.append("\n")
        if tag == "td" or tag == "th":
            self.out.append(" | ")
        # a/span/strong/em/... fall through: keep their text inline

    def handle_endtag(self, tag):
        if tag == "div" and self.depth_in_target:
            self.depth_in_target -= 1
            return
        if not self.depth_in_target:
            return
        if tag in self.SKIP_TAGS:
            if self.open_tags and self.open_tags[-1] == tag:
                self.open_tags.pop()
            return
        if tag == "pre":
            self._in_pre = False
            self._pre_lines = False
            self.out.append("\n```\n")
            return
        if tag == "li" and self._in_pre:
            self._in_li = False
            self.out.append("".join(self._pending))
            self._pending = []
            return
        if tag == "code" and not self._in_pre:
            self.out.append("`")
            return
        if tag in self.BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.depth_in_target or not data:
            return
        if self._in_pre:
            if self._pre_lines:
                if self._in_li:  # stray text between li elements is dropped
                    self._pending.append(data)
                return
            self.out.append(data)
            return
        self.out.append(data)

    def text(self):
        raw = "".join(self.out)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def extract_article(html_text):
    """Return (markdown_body, img_count) using the longest #content_views div."""
    parser = ContentExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return "", 0
    return parser.text(), parser.img_count


# ---------------------------------------------------------------------------
# Stage 4: build the dataset
# ---------------------------------------------------------------------------

def identity_scan(texts_by_source):
    """texts_by_source: list of (kind, id, text). Returns (hits, scanned_count)."""
    pattern = re.compile("|".join(re.escape(k) for k in IDENTITY_KEYWORDS), re.IGNORECASE)
    hits = []
    scanned = 0
    for kind, key, text in texts_by_source:
        scanned += 1
        found = pattern.findall(text)
        if found:
            hits.append({"kind": kind, "key": str(key),
                         "keywords": sorted({k.lower() for k in found})})
    return hits, scanned


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_body(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"source_blog": f"https://blog.csdn.net/{CSDN_USER}",
              "source_github": f"https://github.com/{GITHUB_USER}",
              "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "outputs": {}}

    # ---- blog articles -----------------------------------------------------
    index_path = RAW_DIR / "articles_index.json"
    articles = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    meta_by_id = {a["articleId"]: a for a in articles}

    type_counts = Counter(str(a.get("type")) for a in articles)
    original_ids = {a["articleId"] for a in articles if str(a.get("type")) in ("1", "None", "0", "")}

    pretrain_rows, sft_rows = [], []
    articles_rows = []
    seen_body_hashes = set()
    stats = Counter()
    scan_pool = []

    html_files = sorted(ARTICLE_DIR.glob("*.html.gz"))
    for path in html_files:
        article_id = int(path.stem.split(".")[0])
        meta = meta_by_id.get(article_id)
        if meta is None:
            stats["not_in_index"] += 1
            continue
        if article_id not in original_ids:
            stats["dropped_repost"] += 1
            continue
        body, img_count = extract_article(
            gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace"))
        body = clean_body(body)
        if len(body) < MIN_PRETRAIN_CHARS:
            stats["dropped_too_short"] += 1
            continue
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest in seen_body_hashes:
            stats["dropped_duplicate"] += 1
            continue
        seen_body_hashes.add(digest)

        title = clean_body(html.unescape(meta.get("title", "")))
        pretrain_rows.append({"text": f"{title}\n\n{body}"})
        scan_pool.append(("blog_pretrain", article_id, pretrain_rows[-1]["text"]))
        articles_rows.append({
            "id": article_id,
            "title": title,
            "tags": meta.get("tags") or [],
            "url": meta.get("url", ""),
            "body": body,
        })
        if len(body) >= MIN_SFT_BODY_CHARS:
            sft_rows.append({"conversations": [
                {"role": "user",
                 "content": f"请讲解这道题或这个知识点的思路与解法：{title}"},
                {"role": "assistant", "content": body},
            ]})
            scan_pool.append(("blog_sft", article_id, body))
        stats["kept"] += 1
        stats["images_dropped"] += img_count

    blog_pretrain_path = OUT_DIR / "code92007_blog_pretrain.jsonl"
    blog_sft_path = OUT_DIR / "code92007_blog_sft.jsonl"
    blog_articles_path = OUT_DIR / "code92007_blog_articles.jsonl"
    write_jsonl(blog_pretrain_path, pretrain_rows)
    write_jsonl(blog_sft_path, sft_rows)
    write_jsonl(blog_articles_path, articles_rows)
    report["blog"] = {
        "index_entries": len(articles),
        "html_files": len(html_files),
        "type_counts": dict(type_counts),
        "pretrain_rows": len(pretrain_rows),
        "sft_rows": len(sft_rows),
        "articles_rows": len(articles_rows),
        "row_stats": dict(stats),
    }

    # ---- github code --------------------------------------------------------
    code_rows = []
    code_stats = Counter()
    if REPO_DIR.exists():
        for repo_dir in sorted(p for p in REPO_DIR.iterdir() if p.is_dir()):
            repo = repo_dir.name
            for file_path in repo_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(repo_dir)
                if any(part in SKIP_DIR_NAMES for part in rel.parts):
                    code_stats["skipped_dir"] += 1
                    continue
                if file_path.suffix.lower() not in CODE_EXTENSIONS:
                    code_stats["skipped_ext"] += 1
                    continue
                data = file_path.read_bytes()
                if not (MIN_CODE_FILE_BYTES <= len(data) <= MAX_CODE_FILE_BYTES):
                    code_stats["skipped_size"] += 1
                    continue
                if b"\x00" in data[:8192]:
                    code_stats["skipped_binary"] += 1
                    continue
                text = data.decode("utf-8", errors="replace")
                if text.count("\ufffd") > len(text) * 0.05:
                    code_stats["skipped_mojibake"] += 1
                    continue
                suffix = file_path.suffix.lower()
                if suffix in C_LIKE_COMMENT:
                    header = f"// repo: {repo} path: {rel.as_posix()}\n"
                elif suffix in HASH_COMMENT:
                    header = f"# repo: {repo} path: {rel.as_posix()}\n"
                else:
                    header = f"repo: {repo} path: {rel.as_posix()}\n"
                code_rows.append({"text": header + text})
                scan_pool.append(("code", f"{repo}/{rel.as_posix()}", text))
                code_stats["kept"] += 1
            code_stats[f"repo:{repo}"] = sum(
                1 for r in code_rows if r["text"].startswith(f"// repo: {repo} ")
                or r["text"].startswith(f"# repo: {repo} ")
                or r["text"].startswith(f"repo: {repo} "))

    code_path = OUT_DIR / "code92007_code_pretrain.jsonl"
    write_jsonl(code_path, code_rows)
    report["code"] = {"rows": len(code_rows), "row_stats": dict(code_stats)}

    # ---- identity scan + checksums ------------------------------------------
    hits, scanned = identity_scan(scan_pool)
    (RAW_DIR / "identity_hits.json").write_text(
        json.dumps(hits, ensure_ascii=False, indent=1), encoding="utf-8")
    report["identity_scan"] = {
        "keywords": list(IDENTITY_KEYWORDS),
        "units_scanned": scanned,
        "units_with_hits": len(hits),
        "hits_file": "raw/identity_hits.json",
        "note": "hits are review candidates per AGENTS.md, not auto-deleted",
    }
    report["outputs"] = {
        path.name: {"rows": n, "sha256": sha256_file(path)}
        for path, n in ((blog_pretrain_path, len(pretrain_rows)),
                        (blog_sft_path, len(sft_rows)),
                        (blog_articles_path, len(articles_rows)),
                        (code_path, len(code_rows)))
    }

    report_path = OUT_DIR / "build.report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("blog", "code", "identity_scan")},
                     ensure_ascii=False, indent=2))
    print(f"report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["fetch-index", "fetch-articles", "fetch-repos", "build"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "fetch-index":
        fetch_index()
    elif args.stage == "fetch-articles":
        fetch_articles(args.workers)
    elif args.stage == "fetch-repos":
        fetch_repos()
    else:
        build()


if __name__ == "__main__":
    main()
