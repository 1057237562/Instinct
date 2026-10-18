"""Build a 12B-token Instinct CPT corpus from openbmb/UltraData-Code-L3.

UltraData-Code (arXiv 2602.09003) ships the L3 refined tier of the UltraData
L0-L4 code framework: programming exercises synthesized from L2 ALGO seed
files, serialized per row as ``content`` (task + solution) and
``full_content`` (task + analysis + solution + test in several orders).  The
L3 experiments train a 1B model with continual pre-training, which is exactly
the Instinct CPT use case, so this builder samples whole ``full_content``
records from the eleven language splits proportionally to their upstream size
and assembles a globally shuffled 12B project-tokenizer-token corpus.

Phases (share ``dataset/ultradata_code_l3_12b`` for plan and downloads):

    python scripts/data_builder/build_ultradata_code_l3_12b.py plan
    python scripts/data_builder/build_ultradata_code_l3_12b.py collect
    python scripts/data_builder/build_ultradata_code_l3_12b.py build

``build`` may extend the plan and download more shards itself when a language
quota cannot be met, so the phases only need to be rerun for manual control.

Parallelism: each parquet row group is one unit of work dispatched to a pool
of worker processes (tokenization, 13-word n-gram screening, and identity
screening all run inside the worker); the main process only merges global
deduplication state, enforces per-language token quotas, and feeds the
external shuffle.  ``TOKENIZERS_PARALLELISM`` is deliberately disabled so
each worker tokenizes single-threaded and the pool width is the only
parallelism knob.

Record policy matches pretrain_coder_12b: whole records only (no split, no
truncation), at most 4096 project-tokenizer tokens including BOS/EOS, global
exact text deduplication, cross-corpus deduplication against the corpora the
model already consumed, 13-word n-gram screening against HumanEval, sanitized
MBPP test, and GSM8K test, plus vendor self-identification screening so no
record teaches the assistant to claim another model's identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

# Windows workaround: import datasets before transformers (and torch).
from datasets import load_dataset  # noqa: F401  # before transformers on Windows
import orjson
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_builder.audit_identity_contamination import (  # noqa: E402
    CN_SELF_CUE_RE, EN_SELF_CUE_RE, OTHER_FOREIGN_ALT, OTHER_FOREIGN_TERMS,
    QWEN_HARD_ALT, QWEN_TERMS_HARD, QWEN_TERMS_WEAK,
    _build_claim_patterns, _build_en_claim_patterns, _is_negated,
    _is_non_assertion, _is_role_mention, _iter_sentences, _snippet,
)
from scripts.data_builder.coder_pretrain_common import (  # noqa: E402
    MAX_TOKENS, benchmark_ngrams, has_benchmark_overlap, sha256_file,
)
from scripts.data_builder.collect_coder_pretrain_12b import (  # noqa: E402
    direct_dataset_download,
)
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler  # noqa: E402


REPO = "openbmb/UltraData-Code"
REVISION = "525c9ac8639c5032633392d096e54db8c84a005e"
TREE_URL = f"https://hf-mirror.com/api/datasets/{REPO}/tree/main/data/UltraData-Code-L3"
HTTP_USER_AGENT = "Instinct-ultradata-code-l3-builder/1.0"

WORK = ROOT / "dataset/ultradata_code_l3_12b"
PLAN_PATH = WORK / "plan.json"
RAW = WORK / "raw"

OUTPUT = ROOT / "dataset/pretrain_ultradata_code_l3_12b_identity_clean.jsonl"
REPORT = ROOT / "dataset/pretrain_ultradata_code_l3_12b_identity_clean.report.json"
CANDIDATES = ROOT / "dataset/review_candidates/ultradata_code_l3_12b_identity_candidates.jsonl"

LANGS = ("cpp", "cs", "go", "java", "js", "php", "py", "r", "rb", "rs", "sh")
TARGET_TOKENS = 12_000_000_000
# Probe-measured yield of one ~1.08 GB shard of full_content records after the
# <=4096-token whole-record filter: 237M (py) and 247M (cpp) Instinct tokens.
EST_TOKENS_PER_BYTE = 220e6 / 1.08e9
PLAN_MARGIN = 1.25
TEXT_FIELD = "full_content"
SEED = 20261014
SHUFFLE_CHUNK_ROWS = 100_000
SHUFFLE_CHUNK_BYTES = 512 * 1024 ** 2
CANDIDATE_ROW_CAP = 200_000

SOURCE_NAME = "ultradata_code_l3"
MIXTURE = "ultradata-code-l3-language-proportional"
LICENSE = "apache-2.0"
# Corpora whose texts the model already consumed; L3 must not repeat them.
EXISTING_CORPORA = (
    "dataset/pretrain_coder_12b.jsonl",
    "dataset/pretrain_coder_12b_seg2_identity_clean.jsonl",
    "dataset/pretrain_arxiv_abstracts_160m_tokens_4096.jsonl",
    "dataset/pretrain_arxiv_open_500mb.jsonl",
)

# The upstream auditor does not track the vendors most likely to appear in
# MiniCPM-produced data, so this build extends the vendor set.
EXTRA_VENDORS = r"minicpm[a-z0-9\-]*|openbmb|modelbest|面壁智能|面壁"
VENDOR_ALT = f"(?:{QWEN_HARD_ALT}|{OTHER_FOREIGN_ALT}|{EXTRA_VENDORS})"
VENDOR_RE = re.compile(VENDOR_ALT, re.IGNORECASE)
CLAIM_PATTERNS = _build_claim_patterns(VENDOR_ALT) + _build_en_claim_patterns(VENDOR_ALT)


def _bounded_term(term):
    """ASCII-boundary form for keyword counting (bare alternations otherwise
    match inside ordinary words: "bing" ⊂ "debugging", "gpt" ⊂ "gptnee")."""
    if re.fullmatch(r"[a-z0-9][a-z0-9 .+#\-]*", term, re.IGNORECASE):
        return rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.escape(term)


_VENDOR_TERMS = sorted(
    set(QWEN_TERMS_HARD) | set(QWEN_TERMS_WEAK) | set(OTHER_FOREIGN_TERMS)
    | {"minicpm", "openbmb", "modelbest", "面壁智能", "面壁"},
    key=len, reverse=True,
)
VENDOR_COUNT_RE = re.compile(
    "|".join(_bounded_term(term) for term in _VENDOR_TERMS), re.IGNORECASE
)
# Two-stage prefilter: substring triggers are ~10x cheaper than the bounded
# alternation, and rows that mention no vendor at all skip the sentence scan.
_ASCII_TRIGGERS = tuple(
    term.lower() for term in _VENDOR_TERMS
    if term.isascii() and re.fullmatch(r"[a-z0-9][a-z0-9 .+#\-]*", term, re.IGNORECASE)
)
_CJK_TRIGGER_RE = re.compile(
    "|".join(re.escape(term) for term in _VENDOR_TERMS if not term.isascii())
)


def list_language_files(lang):
    url = f"{TREE_URL}/{lang}"
    request = Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    with urlopen(request, timeout=60) as response:
        items = json.load(response)
    files = [
        {"path": item["path"], "size": int(item.get("size") or 0)}
        for item in items if item.get("type") == "file" and item["path"].endswith(".parquet")
    ]
    files.sort(key=lambda item: item["path"])
    return files


def choose_shards(files, quota):
    """Evenly spaced deterministic shard selection sized to quota * margin."""
    if not files:
        return []
    mean_bytes = sum(item["size"] for item in files) / len(files)
    estimated = quota * PLAN_MARGIN / (EST_TOKENS_PER_BYTE * mean_bytes)
    count = max(1, min(len(files), int(estimated) + 1))
    if count == len(files):
        return list(range(len(files)))
    if count == 1:
        return [len(files) // 2]
    return sorted({round(index * (len(files) - 1) / (count - 1)) for index in range(count)})


def cmd_plan(args):
    WORK.mkdir(parents=True, exist_ok=True)
    languages = {}
    for lang in LANGS:
        files = list_language_files(lang)
        languages[lang] = {
            "upstream_files": len(files),
            "upstream_bytes": sum(item["size"] for item in files),
            "files": files,
        }
        print(f"[plan] {lang}: {len(files)} shards, "
              f"{languages[lang]['upstream_bytes'] / 1e9:.2f} GB", flush=True)
    total_bytes = sum(value["upstream_bytes"] for value in languages.values())

    for lang, value in languages.items():
        quota = round(TARGET_TOKENS * value["upstream_bytes"] / total_bytes)
        chosen = choose_shards(value["files"], quota)
        value["quota"] = quota
        value["chosen"] = chosen
        value["chosen_bytes"] = sum(value["files"][index]["size"] for index in chosen)

    plan = {
        "repo": REPO,
        "revision": REVISION,
        "text_field": TEXT_FIELD,
        "target_tokens": TARGET_TOKENS,
        "est_tokens_per_byte": EST_TOKENS_PER_BYTE,
        "plan_margin": PLAN_MARGIN,
        "upstream_bytes": total_bytes,
        "languages": {
            lang: {
                "quota": value["quota"],
                "upstream_files": value["upstream_files"],
                "upstream_bytes": value["upstream_bytes"],
                "chosen_shards": [
                    {"index": index, "path": value["files"][index]["path"],
                     "size": value["files"][index]["size"]}
                    for index in value["chosen"]
                ],
                "chosen_bytes": value["chosen_bytes"],
            }
            for lang, value in languages.items()
        },
    }
    temporary = PLAN_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, PLAN_PATH)
    planned_bytes = sum(value["chosen_bytes"] for value in languages.values())
    planned_shards = sum(len(value["chosen"]) for value in languages.values())
    print(f"[plan] wrote {PLAN_PATH}; {planned_shards} shards, {planned_bytes / 1e9:.1f} GB", flush=True)


def extend_plan(plan, lang, extra_count):
    """Append the next evenly spaced not-yet-chosen shards for one language."""
    value = plan["languages"][lang]
    have = {item["index"] for item in value["chosen_shards"]}
    files = list_language_files(lang)
    remaining = [index for index in range(len(files)) if index not in have]
    if not remaining:
        return []
    extra_count = min(extra_count, len(remaining))
    step = len(remaining) / extra_count
    picked = sorted({
        remaining[min(len(remaining) - 1, round((offset + 0.5) * step))]
        for offset in range(extra_count)
    })
    for index in picked:
        value["chosen_shards"].append(
            {"index": index, "path": files[index]["path"], "size": files[index]["size"]}
        )
    value["chosen_shards"].sort(key=lambda item: item["index"])
    value.setdefault("extended_builds", []).append(
        {"added_shards": len(picked), "paths": [files[index]["path"] for index in picked]}
    )
    return [files[index]["path"] for index in picked]


def shard_destination(upstream_path):
    return RAW / upstream_path


def download_paths(paths, workers):
    pending = [path for path in paths if not shard_destination(path).is_file()]
    if not pending:
        return
    print(f"[collect] downloading {len(pending)} shards with {workers} workers", flush=True)

    def fetch(path):
        direct_dataset_download(REPO, REVISION, path, RAW, timeout=180)
        return path

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch, pending):
            completed += 1
            if completed % 5 == 0 or completed == len(pending):
                print(f"[collect] {completed}/{len(pending)} shards ready", flush=True)
    print(f"[collect] all requested shards present", flush=True)


def cmd_collect(args):
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    paths = [item["path"] for value in plan["languages"].values() for item in value["chosen_shards"]]
    download_paths(paths, args.workers)


# -------------------------------------------------------- group workers ----

_SHARD_TOKENIZER = None
_SHARD_BENCHMARK = None


def shard_tokenizer():
    global _SHARD_TOKENIZER
    if _SHARD_TOKENIZER is None:
        _SHARD_TOKENIZER = AutoTokenizer.from_pretrained(
            ROOT / "model", local_files_only=True
        ).backend_tokenizer
    return _SHARD_TOKENIZER


def shard_benchmark():
    global _SHARD_BENCHMARK
    if _SHARD_BENCHMARK is None:
        _SHARD_BENCHMARK, _sources = benchmark_ngrams()
    return _SHARD_BENCHMARK


def identity_scan(text):
    """Return (verdict, hits, keyword_counts) for one record.

    verdict is "clean", "claim" (explicit foreign self-identification), or
    "review" (self-cue next to a vendor name; keep but audit).
    """
    hits = []
    keywords = Counter()
    lowered = text.lower()
    if not (any(trigger in lowered for trigger in _ASCII_TRIGGERS)
            or _CJK_TRIGGER_RE.search(text)):
        return "clean", hits, keywords
    for match in VENDOR_COUNT_RE.finditer(text):
        keywords[match.group(0).lower()] += 1
    for sentence in _iter_sentences(text):
        term_match = VENDOR_RE.search(sentence)
        if not term_match:
            continue
        claimed = False
        for pattern in CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            term = VENDOR_RE.search(sentence, match.start(), match.end())
            if term is None:
                term = VENDOR_RE.search(sentence[: match.end()])
            if term and _is_negated(sentence, term.start(), term.end()):
                continue
            if term and _is_role_mention(sentence, match, term):
                continue
            anchor = term.start() if term else match.start()
            tier = "review" if _is_non_assertion(sentence, anchor) else "claim"
            hits.append({"tier": tier, "match": match.group(0).strip()[:200],
                         "snippet": _snippet(sentence, match.start(), match.end())})
            claimed = True
            break
        if claimed:
            continue
        cue = CN_SELF_CUE_RE.search(sentence) or EN_SELF_CUE_RE.search(sentence)
        if cue and not _is_negated(sentence, term_match.start(), term_match.end()):
            hits.append({"tier": "review", "match": term_match.group(0),
                         "snippet": _snippet(sentence, term_match.start(), term_match.end())})
    if any(hit["tier"] == "claim" for hit in hits):
        return "claim", hits, keywords
    return ("review", hits, keywords) if hits else ("clean", hits, keywords)


def build_group(task):
    """Process one parquet row group inside a worker process.

    Returns accepted-candidate rows (already serialized, with text
    fingerprints) plus per-group counters; the main process owns the global
    dedup sets, token quotas, and the shuffle feed.
    """
    lang, shard_path, group_index = task
    tokenizer = shard_tokenizer()
    benchmark = shard_benchmark()
    counters = Counter()
    keywords = Counter()
    rows = []
    candidates = []
    local_seen = set()
    parquet = pq.ParquetFile(shard_destination(shard_path))
    batch = parquet.read_row_group(group_index, columns=["uuid", TEXT_FIELD]).to_pylist()
    texts = []
    keys = []
    for row in batch:
        counters["scanned"] += 1
        text = row.get(TEXT_FIELD)
        if not isinstance(text, str) or not text.strip():
            counters["invalid_text"] += 1
            continue
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if digest in local_seen:
            counters["duplicate_text_local"] += 1
            continue
        local_seen.add(digest)
        texts.append(text)
        keys.append((row.get("uuid") or "", text, digest))
    if not texts:
        return {"lang": lang, "rows": rows, "candidates": candidates,
                "counters": dict(counters), "keywords": dict(keywords)}
    encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
    for (uuid, text, digest), encoding in zip(keys, encoded):
        token_count = len(encoding.ids) + 2
        if not 1 <= token_count <= MAX_TOKENS:
            counters["outside_token_range"] += 1
            continue
        if has_benchmark_overlap(text, benchmark):
            counters["benchmark_overlap"] += 1
            continue
        verdict, hits, vendor_keywords = identity_scan(text)
        if vendor_keywords:
            keywords.update(vendor_keywords)
        if verdict == "claim":
            counters["identity_claim"] += 1
            candidates.append({
                "language": lang, "source_shard": shard_path, "uuid": uuid,
                "hits": hits, "text": text,
            })
            continue
        if hits:
            counters["identity_review"] += 1
            candidates.append({
                "language": lang, "source_shard": shard_path, "uuid": uuid,
                "hits": hits, "text": text,
            })
        narrow = int.from_bytes(
            hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
        )
        row_bytes = orjson.dumps({
            "text": text,
            "token_count": token_count,
            "source": f"{SOURCE_NAME}:{lang}",
            "license": LICENSE,
            "mix_category": lang,
            "mixture": MIXTURE,
            "uuid": uuid,
        })
        rows.append((row_bytes, token_count, digest, narrow))
    return {"lang": lang, "rows": rows, "candidates": candidates,
            "counters": dict(counters), "keywords": dict(keywords)}


def fingerprint_existing_corpora():
    """BLAKE2b-64bit fingerprints of every text the model already consumed."""
    fingerprints = set()
    sources = {}
    for relative in EXISTING_CORPORA:
        path = ROOT / relative
        if not path.is_file():
            sources[relative] = "missing; skipped"
            continue
        rows = 0
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = orjson.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text:
                    fingerprints.add(
                        int.from_bytes(
                            hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
                        )
                    )
                rows += 1
                if rows % 2_000_000 == 0:
                    print(f"[cross-dedup] {path.name}: {rows:,} rows, "
                          f"{stream.tell() / 1e9:.1f} GB", flush=True)
        sources[str(path)] = {"rows": rows, "bytes": path.stat().st_size}
        print(f"[cross-dedup] {path.name}: {rows:,} rows done", flush=True)
    return fingerprints, sources


def audit_contiguous_mixing(path, languages):
    """Byte-spaced window audit: every window must expose many languages."""
    windows = 64
    window_rows = 2048
    size = path.stat().st_size
    expected = {lang: value["quota"] / TARGET_TOKENS for lang, value in languages.items()}
    minimum_languages = None
    maximum_error = Counter()
    sampled_rows = 0
    big_three = ("py", "cpp", "java")
    with path.open("rb") as stream:
        for number in range(windows):
            offset = size * number // windows
            stream.seek(offset)
            if offset:
                stream.readline()
            category_tokens = Counter()
            rows = 0
            for _ in range(window_rows):
                line = stream.readline()
                if not line:
                    break
                row = orjson.loads(line)
                category_tokens[str(row.get("mix_category"))] += int(row.get("token_count") or 0)
                rows += 1
            total = sum(category_tokens.values())
            if rows == 0 or total == 0:
                raise ValueError(f"Empty audit window near byte {offset}")
            sampled_rows += rows
            minimum_languages = (
                len(category_tokens) if minimum_languages is None
                else min(minimum_languages, len(category_tokens))
            )
            for lang, share in expected.items():
                error = abs(category_tokens[lang] / total - share)
                maximum_error[lang] = max(maximum_error[lang], error)
    failures = []
    for lang in big_three:
        if maximum_error[lang] > 0.05:
            failures.append(f"{lang} window token-fraction error {maximum_error[lang]:.1%} > 5.0%")
    aggregate_error = sum(maximum_error[lang] for lang in expected if lang not in big_three)
    if aggregate_error > 0.15:
        failures.append(f"small-language aggregate window error {aggregate_error:.1%} > 15%")
    if (minimum_languages or 0) < 5:
        failures.append(f"only {minimum_languages} languages in the least diverse window")
    result = {
        "method": "deterministic byte-spaced contiguous-window audit",
        "windows": windows, "rows_per_window": window_rows, "sampled_rows": sampled_rows,
        "minimum_languages_per_window": minimum_languages or 0,
        "maximum_absolute_token_fraction_error": dict(maximum_error),
        "limits": {"big_three_each": 0.05, "small_languages_aggregate": 0.15,
                   "minimum_languages_per_window": 5},
        "passed": not failures,
    }
    if failures:
        raise RuntimeError("Dataset is not sufficiently mixed: " + "; ".join(failures))
    return result


def merge_group(result, plan, stats, rejected, keyword_totals, seen_texts,
                cross_fingerprints, shuffler, candidate_writer):
    """Fold one worker result into global state; returns rows accepted."""
    lang = result["lang"]
    quota = plan["languages"][lang]["quota"]
    for key, value in result["counters"].items():
        if key in ("benchmark_overlap", "identity_claim"):
            rejected[f"{lang}:{key}"] += value
        elif key == "identity_review":
            stats[lang]["identity_review_kept"] += value
        else:
            rejected[f"{lang}:{key}"] += value
    if result["keywords"]:
        keyword_totals.update(result["keywords"])
    for payload in result["candidates"]:
        candidate_writer(payload)
    accepted = 0
    for row_bytes, token_count, digest, narrow in result["rows"]:
        if narrow in cross_fingerprints:
            rejected[f"{lang}:existing_corpus_duplicate"] += 1
            continue
        if digest in seen_texts:
            rejected[f"{lang}:duplicate_text_global"] += 1
            continue
        seen_texts.add(digest)
        if stats[lang]["tokens"] + token_count > quota:
            rejected[f"{lang}:over_budget"] += 1
            continue
        if not shuffler.add_serialized(row_bytes):
            rejected[f"{lang}:duplicate_record"] += 1
            continue
        stats[lang]["rows"] += 1
        stats[lang]["tokens"] += token_count
        accepted += 1
    return accepted


def consume_language(lang, shard_paths, pool, plan, stats, rejected, keyword_totals,
                     seen_texts, cross_fingerprints, shuffler, candidate_writer):
    """Dispatch every row group of the given shards and merge the results."""
    tasks = []
    for shard_path in shard_paths:
        parquet = pq.ParquetFile(shard_destination(shard_path))
        for group in range(parquet.metadata.num_row_groups):
            tasks.append((lang, shard_path, group))
    print(f"[build] {lang}: dispatching {len(tasks)} row groups "
          f"from {len(shard_paths)} shards", flush=True)
    completed = 0
    accepted_total = 0
    for future in as_completed([pool.submit(build_group, task) for task in tasks]):
        accepted_total += merge_group(
            future.result(), plan, stats, rejected, keyword_totals,
            seen_texts, cross_fingerprints, shuffler, candidate_writer,
        )
        completed += 1
        if completed % 40 == 0 or completed == len(tasks):
            print(f"[build] {lang}: {completed}/{len(tasks)} groups, "
                  f"accepted_rows={stats[lang]['rows']:,} "
                  f"tokens={stats[lang]['tokens']:,}", flush=True)
    return accepted_total


def cmd_build(args):
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if OUTPUT.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}; pass --overwrite")
    if REPORT.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {REPORT}; pass --overwrite")

    cross_fingerprints, cross_sources = fingerprint_existing_corpora()
    print(f"[cross-dedup] {len(cross_fingerprints):,} known-text fingerprints", flush=True)

    stats = {lang: Counter() for lang in plan["languages"]}
    rejected = Counter()
    keyword_totals = Counter()
    state = {"candidates_written": 0}
    candidates_hash = hashlib.sha256()
    seen_texts = set()
    shuffler = ExternalRandomShuffler(
        OUTPUT, args.shuffle_seed, SHUFFLE_CHUNK_ROWS, chunk_bytes=SHUFFLE_CHUNK_BYTES,
    )
    candidate_temporary = CANDIDATES.with_suffix(".jsonl.tmp")
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    candidate_stream = candidate_temporary.open("wb")

    def candidate_writer(payload):
        state["candidates_written"] += 1
        if state["candidates_written"] > CANDIDATE_ROW_CAP:
            return
        serialized = orjson.dumps(payload)
        candidates_hash.update(hashlib.blake2b(serialized, digest_size=8).digest())
        candidate_stream.write(serialized + b"\n")

    try:
        with ProcessPoolExecutor(max_workers=args.group_workers) as pool:
            for lang in sorted(plan["languages"]):
                shards = [item["path"] for item in plan["languages"][lang]["chosen_shards"]]
                consume_language(lang, shards, pool, plan, stats, rejected,
                                 keyword_totals, seen_texts, cross_fingerprints,
                                 shuffler, candidate_writer)
                shortfall = plan["languages"][lang]["quota"] - stats[lang]["tokens"]
                if shortfall > MAX_TOKENS:
                    if args.no_extend:
                        raise RuntimeError(
                            f"{lang} short by {shortfall:,} tokens and --no-extend set"
                        )
                    print(f"[build] {lang} short by {shortfall:,} tokens; extending plan",
                          flush=True)
                    extra = max(2, int(shortfall / (EST_TOKENS_PER_BYTE * 1.08e9)) + 2)
                    new_paths = extend_plan(plan, lang, extra)
                    if not new_paths:
                        raise RuntimeError(f"{lang} has no upstream shards left to extend with")
                    download_paths(new_paths, args.collect_workers)
                    consume_language(lang, new_paths, pool, plan, stats, rejected,
                                     keyword_totals, seen_texts, cross_fingerprints,
                                     shuffler, candidate_writer)
                    shortfall = plan["languages"][lang]["quota"] - stats[lang]["tokens"]
                    if shortfall > MAX_TOKENS:
                        raise RuntimeError(
                            f"{lang} still short by {shortfall:,} tokens after extension"
                        )
                stats[lang]["shards_consumed"] = len(plan["languages"][lang]["chosen_shards"])

        candidate_stream.flush()
        os.replace(candidate_temporary, CANDIDATES)
        shuffle_result = shuffler.finish(progress_label="L3 Build Shuffle")
    except BaseException:
        candidate_stream.close()
        shuffler.temporary_dir.cleanup()
        raise
    candidate_stream.close()

    mixing_audit = audit_contiguous_mixing(OUTPUT, plan["languages"])
    total_tokens = sum(value["tokens"] for value in stats.values())
    total_rows = sum(value["rows"] for value in stats.values())
    output_sha = sha256_file(OUTPUT)
    candidates_sha = sha256_file(CANDIDATES) if CANDIDATES.is_file() else None

    report = {
        "name": "instinct_ultradata_code_l3_12b",
        "created_for": "Instinct V1 MoE: 678,726,144 total / 187,992,576 active parameters",
        "reference": {
            "dataset": "openbmb/UltraData-Code (config UltraData-Code-L3)",
            "revision": REVISION,
            "framework": "UltraData L0-L4 tiered data management (arXiv 2602.09003)",
            "license": ("Apache-2.0 dataset release; synthesized from public repository "
                        "code whose per-repo licenses still apply"),
            "l3_evidence": (
                "Under controlled 10B-token continual pre-training of a 1B model, replacing "
                "half of the L2 tokens with L3 improved EvalPlus by 8.42pp and MultiPL-E by "
                "8.07pp over L2-only training."
            ),
        },
        "output": str(OUTPUT),
        "output_sha256": output_sha,
        "rows": total_rows,
        "tokens": total_tokens,
        "unique_tokens_per_active_parameter": total_tokens / 187_992_576,
        "unique_tokens_per_total_parameter": total_tokens / 678_726_144,
        "text_field": TEXT_FIELD,
        "whole_records_only": True,
        "text_split": False,
        "text_truncated": False,
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "language_quotas": {lang: value["quota"] for lang, value in plan["languages"].items()},
        "language_alloc_rule": "quota proportional to upstream L3 shard bytes per language",
        "components": {lang: dict(value) for lang, value in stats.items()},
        "component_token_fraction": {
            lang: value["tokens"] / total_tokens for lang, value in stats.items() if total_tokens
        },
        "global_exact_text_deduplication": True,
        "cross_corpus_deduplication": {
            "method": "BLAKE2b-64 of exact text against corpora the model already consumed",
            "sources": cross_sources,
            "fingerprints": len(cross_fingerprints),
        },
        "benchmark_screening": {
            "method": "reject any record sharing a 13-word n-gram with held-out benchmark text",
            "sources": "HumanEval, sanitized MBPP test, GSM8K test (coder_pretrain_common)",
        },
        "identity_clean": {
            "target_identity": "InstinctV1Moe; must not self-identify as any other model/provider",
            "method": (
                "audit_identity_contamination claim patterns extended with MiniCPM/OpenBMB/"
                "ModelBest vendors; whole records dropped on an explicit self-identification "
                "claim; speculative/role/disavowed sentences kept and emitted as review candidates"
            ),
            "vendor_keyword_hits": dict(keyword_totals),
            "claim_rows_dropped": sum(
                value for key, value in rejected.items() if key.endswith(":identity_claim")
            ),
            "candidates_path": str(CANDIDATES),
            "candidates_rows": state["candidates_written"],
            "candidates_sha256": candidates_sha,
            "candidate_row_cap": CANDIDATE_ROW_CAP,
            "decision": ("Training-ready; claim-tier records were dropped, review-tier "
                         "records kept for audit."),
        },
        "shuffle": {
            "method": "128-bit random-key external merge sort",
            "seed": args.shuffle_seed,
            "chunk_rows": SHUFFLE_CHUNK_ROWS,
            "chunk_bytes": SHUFFLE_CHUNK_BYTES,
            "temporary_chunks": shuffle_result["temporary_chunks"],
            "mixing_audit": mixing_audit,
        },
        "build_parallelism": {
            "unit": "one parquet row group per task",
            "group_workers": args.group_workers,
            "tokenizer_parallelism": "disabled in workers; pool width is the only knob",
        },
        "inputs": {
            lang: {
                "planned_shards": [item["path"] for item in value["chosen_shards"]],
                "extended_builds": value.get("extended_builds", []),
            }
            for lang, value in plan["languages"].items()
        },
        "rejected": dict(rejected),
        "checks": [
            "all output texts are globally exact-deduplicated and absent from previously consumed corpora",
            "all records contain at most 4096 project-tokenizer tokens including BOS/EOS",
            "all records are complete and were neither split nor truncated",
            "all rows were screened against HumanEval, sanitized MBPP-test, and GSM8K-test",
            "no record carries a foreign-model self-identification claim; audit candidates retained",
            "language mixture mirrors the upstream UltraData-Code-L3 split sizes",
        ],
        "limitations": [
            "13-word n-gram screening cannot prove absence of semantic benchmark contamination.",
            ("Upstream L3 records are model-synthesized; factual correctness of analysis "
             "fields is not independently verified."),
            "Apache-2.0 covers the dataset release; the seed repositories keep their own licenses.",
            ("Token ratios were probed on one py and one cpp shard; other languages are "
             "assumed similar."),
        ],
    }
    temporary_report = REPORT.with_suffix(".report.json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, REPORT)
    updated_plan = PLAN_PATH.with_suffix(".json.tmp")
    updated_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(updated_plan, PLAN_PATH)
    print(json.dumps({key: report[key] for key in (
        "output", "rows", "tokens", "component_token_fraction",
        "identity_clean")}, ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="List upstream shards and select an even sample per language.")
    collect = sub.add_parser("collect", help="Download planned shards (resumable).")
    collect.add_argument("--workers", type=int, default=6)
    build = sub.add_parser("build", help="Filter, tokenize, and assemble the corpus.")
    build.add_argument("--group-workers", type=int, default=max(1, (os.cpu_count() or 8) - 6))
    build.add_argument("--collect-workers", type=int, default=6)
    build.add_argument("--shuffle-seed", type=int, default=SEED)
    build.add_argument("--no-extend", action="store_true",
                       help="Fail instead of downloading more shards when a quota is short.")
    build.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_FALLBACK_ENDPOINT", "https://huggingface.co")
    # Worker processes inherit this; each worker tokenizes single-threaded so
    # the pool width is the only parallelism knob (no rayon oversubscription).
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "build":
        cmd_build(args)


if __name__ == "__main__":
    main()
