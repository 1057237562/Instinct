"""Clean / rewrite repetitive reasoning ("think") in UltraData-style SFT jsonl.

Background (see docs/reports/2026-11-22-dataset-repetitive-reasoning.md): 37% of
the raw UltraData code-think pool contains looping reasoning (template-sentence
repeats, "But wait" ping-pong, copied lines, numeric enumeration tables,
degenerate token loops, repeated code blocks) and the repetition is concentrated
in very long traces. v4 "fixed" it by squashing lengths to ~2k tokens, which also
removed every long, properly-closed trace, so the model loops as soon as it tries
to think long.

This tool does the opposite: keep the natural length distribution, but make every
kept row *clean and closed*. Policy is "repair if cheap, otherwise drop the whole
row" - never blind truncation.

Sub-commands
------------
scan          Detection only. Reproduces the report tables (length buckets,
              T1..T6 signal counts, per-lane rates) as JSON.
clean         Stream jsonl -> jsonl. Per row: keep / rewrite / drop.
rebuild-lane  Two-pass selection of a think lane from a (raw) pool: clean every
              row, drop unclosed / too-long rows, then sample with a forced
              short-think quota while preserving the natural length shape.

Rewrite stages (only run on rows flagged dirty)
-----------------------------------------------
 1. identical fenced code blocks inside the reasoning  -> keep first   (T6)
 2. identical lines (>=12 chars)                       -> <= max_occ   (T3, T4-ish)
 3. identical sentences (>=20 chars)                   -> <= max_occ   (T1/T2)
 4. repeated token / short phrase (>=5 consecutive)    -> 1 copy      (T5)
A rewritten row is accepted only if, afterwards: the 5-gram unique ratio is at
least --min-unique-5gram, no T3/T5/T6 signal is left, the ping-pong density is
bounded, at most --max-removed-ratio of the words were removed, and the row still
has a final answer (reasoning is closed).

Schema handling is automatic (override with --reasoning-key / --lane-key):
  * top-level str field  reasoning | reasoning_content | thinking | think
  * messages[-1 assistant] with one of those fields
  * "<think>...</think>" inside messages[-1].content / response / output / text
  * unclosed "<think>" (no "</think>") == loop that hit the length cap -> drop

Usage
-----
  python experiments/clean_think_repetition.py scan raw/Code_think_part-*.jsonl --report scan.json
  python experiments/clean_think_repetition.py clean train/*.jsonl --out-dir train_clean --workers 8
  python experiments/clean_think_repetition.py rebuild-lane raw/Code_think_part-*.jsonl \
      --out lane_code_think_v5.jsonl --target-rows 7390 --max-tokens 8192 --short-quota 0.25
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
FENCE = re.compile(r"```[^\n`]*\n.*?```", re.S)
SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
SELF_DOUBT = re.compile(r"(but wait|wait,|let me check|let's check|let's verify|let me verify|"
                        r"actually|hmm|i think|maybe we|let's test|so we need|we can)", re.I)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

REASONING_KEYS = ("reasoning", "reasoning_content", "thinking", "think")
INLINE_KEYS = ("content", "response", "output", "text")
MESSAGE_KEYS = ("messages", "conversations")

# Natural length shape buckets (reasoning words); the report uses the same ones.
LENGTH_BUCKETS = ((0, 256), (257, 768), (769, 1536), (1537, 2500), (2501, 4096), (4097, 8192))

DEFAULT_MAX_OCC = 1
DEFAULT_MIN_REPEATS = 5


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def open_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


@dataclass
class Reasoning:
    """Location of the reasoning text inside a row."""

    text: str
    kind: str  # field | message_field | inline
    key: str
    message_index: int = -1
    unclosed: bool = False


def messages_of(row):
    for key in MESSAGE_KEYS:
        value = row.get(key) if isinstance(row, dict) else None
        if isinstance(value, list):
            return key, value
    return None, None


def assistant_index(messages):
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], dict) and messages[index].get("role") == "assistant":
            return index
    return -1


def locate_reasoning(row, reasoning_key=None):
    """Find the reasoning text, or None. Unclosed inline thinking is flagged."""
    keys = (reasoning_key,) if reasoning_key else REASONING_KEYS
    if isinstance(row, dict):
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return Reasoning(value, "field", key)
    message_key, messages = messages_of(row)
    if isinstance(messages, list) and messages:
        index = assistant_index(messages)
        if index >= 0:
            message = messages[index]
            for key in keys:
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    return Reasoning(value, "message_field", key, index)
            for key in INLINE_KEYS:
                content = message.get(key)
                if isinstance(content, str) and THINK_OPEN in content:
                    body, tail = content.split(THINK_OPEN, 1)
                    if THINK_CLOSE not in tail:
                        return Reasoning(tail, "inline", key, index, unclosed=True)
                    return Reasoning(tail.split(THINK_CLOSE, 1)[0], "inline", key, index)
    if isinstance(row, dict):
        for key in INLINE_KEYS:
            content = row.get(key)
            if isinstance(content, str) and THINK_OPEN in content:
                body, tail = content.split(THINK_OPEN, 1)
                if THINK_CLOSE not in tail:
                    return Reasoning(tail, "inline", key, -1, unclosed=True)
                return Reasoning(tail.split(THINK_CLOSE, 1)[0], "inline", key, -1)
    return None


def set_reasoning(row, reasoning, new_text):
    if reasoning.kind == "field":
        row[reasoning.key] = new_text
        return
    if reasoning.kind == "message_field":
        _, messages = messages_of(row)
        messages[reasoning.message_index][reasoning.key] = new_text
        return
    _, messages = messages_of(row)
    container = messages[reasoning.message_index] if reasoning.message_index >= 0 else row
    content = container[reasoning.key]
    head, tail = content.split(THINK_OPEN, 1)
    body, rest = tail.split(THINK_CLOSE, 1)
    container[reasoning.key] = head + THINK_OPEN + new_text + THINK_CLOSE + rest


def answer_text(row):
    message_key, messages = messages_of(row)
    if isinstance(messages, list) and messages:
        index = assistant_index(messages)
        if index >= 0:
            content = messages[index].get("content")
            if isinstance(content, str):
                return content
    return ""


def words_of(text):
    return WORD.findall(text.lower())


def five_gram_ratio(words):
    if len(words) < 5:
        return 1.0
    grams = {tuple(words[i:i + 5]) for i in range(len(words) - 4)}
    return len(grams) / (len(words) - 4)


def signals(text):
    """Structural repetition signals, matching the dataset report heuristics."""
    words = words_of(text)
    n = len(words)
    ratio = five_gram_ratio(words)
    lines = Counter(re.sub(r"\s+", " ", line).strip() for line in text.splitlines()
                    if len(line.strip()) >= 12)
    max_line = max(lines.values()) if lines else 0
    blocks = Counter(re.sub(r"\s+", " ", body).strip() for body in
                     re.findall(r"```[^\n`]*\n(.*?)```", text, re.S))
    dup_code = max(blocks.values()) if blocks else 0
    doubt = len(SELF_DOUBT.findall(text))
    degenerate = 0
    if n >= 5:
        grams = Counter(tuple(words[i:i + 5]) for i in range(n - 4))
        top, count = grams.most_common(1)[0]
        degenerate = count if len(set(top)) == 1 else 0
    flags = []
    if max_line >= 5:
        flags.append("T3_repeated_line")
    if degenerate >= 5:
        flags.append("T5_degenerate_tokens")
    if dup_code >= 2:
        flags.append("T6_repeated_code_block")
    if ratio < 0.88 and n >= 30:
        flags.append("ngram_repetition")
    if doubt and n and doubt * 100 / n >= 2.0:
        flags.append("T2_self_correction")
    return dict(words=n, ratio=round(ratio, 4), max_line=max_line, dup_code=dup_code,
                doubt=doubt, degenerate=degenerate, flags=flags)


def dense(text, min_chars):
    """True when the strongest repeated normalized line exceeds max_occ."""
    counts = Counter(re.sub(r"\s+", " ", line).strip() for line in text.splitlines()
                     if len(line.strip()) >= min_chars)
    return max(counts.values()) if counts else 0


def split_fences(text):
    """Split text into (is_code, segment) preserving fenced blocks."""
    parts = []
    position = 0
    for match in FENCE.finditer(text):
        if match.start() > position:
            parts.append((False, text[position:match.start()]))
        parts.append((True, match.group(0)))
        position = match.end()
    if position < len(text):
        parts.append((False, text[position:]))
    return parts


def dedupe_lines(segment, max_occ, min_chars=12):
    trailing = "\n" if segment.endswith("\n") else ""
    seen = Counter()
    kept = []
    for line in segment.splitlines():
        key = re.sub(r"\s+", " ", line).strip()
        if len(key) >= min_chars:
            if seen[key] >= max_occ:
                continue
            seen[key] += 1
        kept.append(line)
    out = "\n".join(kept)
    return out + trailing if trailing and not out.endswith("\n") else out


def dedupe_sentences(segment, max_occ, min_chars=20):
    """Drop duplicate sentences per line so newlines / list structure survive."""
    output = []
    for line in segment.splitlines(keepends=True):
        ending = line[len(line.rstrip("\r\n")):]
        body = line[:len(line) - len(ending)] if ending else line
        sentences = SENTENCE.findall(body)
        if len(sentences) < 4:
            output.append(line)
            continue
        seen = Counter()
        kept = []
        for sentence in sentences:
            key = re.sub(r"\s+", " ", sentence).strip()
            if len(key) >= min_chars:
                if seen[key] >= max_occ:
                    continue
                seen[key] += 1
            kept.append(sentence)
        output.append("".join(kept) + ending)
    return "".join(output)


def collapse_repeated_units(segment, min_repeats=DEFAULT_MIN_REPEATS, max_unit=6):
    """Collapse token / short-phrase runs that repeat >= min_repeats consecutively."""
    leading = segment[:len(segment) - len(segment.lstrip())]
    tokens = re.findall(r"\S+\s*", segment)
    if not any(WORD.search(token) for token in tokens):
        return segment
    changed = True
    guard = 0
    while changed and guard < 5000:
        guard += 1
        changed = False
        for unit_len in range(1, max_unit + 1):
            index = 0
            while index + unit_len * min_repeats <= len(tokens):
                unit = tokens[index:index + unit_len]
                if not any(WORD.search(token) for token in unit):
                    index += 1
                    continue
                repeats = 1
                while tokens[index + repeats * unit_len:index + (repeats + 1) * unit_len] == unit:
                    repeats += 1
                if repeats >= min_repeats:
                    del tokens[index + unit_len:index + repeats * unit_len]
                    changed = True
                    break
                index += 1
            if changed:
                break
    return leading + "".join(tokens)


def rewrite(text, max_occ=DEFAULT_MAX_OCC):
    """Apply the four repair stages; code blocks keep their content except duplicates."""
    blocks = []
    seen_blocks = set()
    for is_code, segment in split_fences(text):
        if is_code:
            key = re.sub(r"\s+", " ", segment)
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            blocks.append(segment)
        else:
            segment = dedupe_lines(segment, max_occ)
            segment = dedupe_sentences(segment, max_occ)
            segment = collapse_repeated_units(segment)
            blocks.append(segment)
    return "".join(blocks)


def judge_after(text, before_words, min_unique=0.88, max_removed_ratio=0.35,
                max_pingpong=2.0, max_occ=DEFAULT_MAX_OCC):
    """Accept a rewrite only when all structural and conservation checks pass."""
    after = signals(text)
    words = after["words"]
    if words < 8:
        return None, "too_short_after"
    if after["ratio"] < min_unique:
        return None, "still_ngram_repetition"
    if after["max_line"] > max_occ:
        return None, "still_repeated_line"
    if after["degenerate"] >= 5:
        return None, "still_degenerate_tokens"
    if after["dup_code"] >= 2:
        return None, "still_duplicate_code"
    if after["doubt"] * 100 / max(words, 1) > max_pingpong:
        return None, "still_pingpong"
    removed = before_words - words
    if before_words and removed > max_removed_ratio * before_words:
        return None, "removed_too_much"
    return after, None


def clean_row(row, *, min_unique=0.88, max_occ=DEFAULT_MAX_OCC,
              max_removed_ratio=0.35, max_pingpong=2.0, reasoning_key=None):
    """Return (status, row, detail). status is keep_clean / rewritten / dropped."""
    reasoning = locate_reasoning(row, reasoning_key)
    if reasoning is None:
        return "dropped", None, "no_reasoning"
    if reasoning.unclosed:
        return "dropped", None, "unclosed_think"
    if not answer_text(row).strip() and reasoning.kind != "field":
        return "dropped", None, "empty_answer"
    before = signals(reasoning.text)
    dirty = (before["ratio"] < min_unique and before["words"] >= 30) or \
            before["max_line"] >= 5 or before["degenerate"] >= 5 or before["dup_code"] >= 2
    if not dirty:
        return "keep_clean", row, None
    repaired = rewrite(reasoning.text, max_occ=max_occ)
    if repaired.strip() == reasoning.text.strip():
        return "dropped", None, "repair_no_change"
    after, reason = judge_after(repaired, before["words"], min_unique=min_unique,
                                max_removed_ratio=max_removed_ratio, max_pingpong=max_pingpong,
                                max_occ=max_occ)
    if after is None:
        return "dropped", None, reason
    set_reasoning(row, reasoning, repaired)
    return "rewritten", row, None


def iter_inputs(paths):
    for pattern in paths:
        path = Path(pattern)
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                yield child
        elif "*" in pattern or "?" in pattern:
            for child in sorted(Path().glob(pattern)):
                yield child
        else:
            yield path


def scan_file(path, reasoning_key=None, lane_key=None):
    path = Path(path)

    def lane_of(row):
        if lane_key:
            value = row
            for part in lane_key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return str(value) if value is not None else "unknown"
        for key in ("source", "domain", "category", "think_type"):
            if isinstance(row, dict) and row.get(key) is not None:
                return str(row[key])
        return "unknown"

    stats = dict(rows=0, with_reasoning=0, repetitive=0, buckets=Counter(),
                 bucket_rep=Counter(), flags=Counter(), lanes=defaultdict(lambda: dict(rows=0, rep=0, words=0)))
    for row in read_jsonl(path):
        stats["rows"] += 1
        reasoning = locate_reasoning(row, reasoning_key)
        lane = lane_of(row)
        stats["lanes"][lane]["rows"] += 1
        if reasoning is None:
            continue
        info = signals(reasoning.text)
        stats["with_reasoning"] += 1
        stats["lanes"][lane]["words"] += info["words"]
        repetitive = info["ratio"] < 0.88 and info["words"] >= 30
        if repetitive:
            stats["repetitive"] += 1
            stats["lanes"][lane]["rep"] += 1
            for flag in info["flags"]:
                stats["flags"][flag] += 1
        index = bucket_index(info["words"])
        stats["buckets"][len(LENGTH_BUCKETS) if index is None else index] += 1
        if repetitive and index is not None:
            stats["bucket_rep"][index] += 1
    return path.name, {**stats,
                       "buckets": dict(stats["buckets"]),
                       "bucket_rep": dict(stats["bucket_rep"]),
                       "flags": dict(stats["flags"]),
                       "lanes": {lane: dict(entry) for lane, entry in stats["lanes"].items()}}


def merge_scan(results):
    merged = dict(rows=0, with_reasoning=0, repetitive=0, buckets=Counter(),
                  bucket_rep=Counter(), flags=Counter(), lanes={})
    for _, stats in results:
        for key in ("rows", "with_reasoning", "repetitive"):
            merged[key] += stats[key]
        merged["buckets"].update(stats["buckets"])
        merged["bucket_rep"].update(stats["bucket_rep"])
        merged["flags"].update(stats["flags"])
        for lane, entry in stats["lanes"].items():
            target = merged["lanes"].setdefault(lane, dict(rows=0, rep=0, words=0))
            for key in target:
                target[key] += entry[key]
    buckets = []
    for index, (lo, hi) in enumerate(LENGTH_BUCKETS):
        n = merged["buckets"][index]
        rep = merged["bucket_rep"][index]
        buckets.append(dict(bucket=f"{lo}-{hi}", rows=n, repetitive=rep,
                            rate=round(rep / n, 4) if n else 0.0))
    overflow = merged["buckets"][len(LENGTH_BUCKETS)]
    overflow_rep = merged["repetitive"] - sum(merged["bucket_rep"].values())
    buckets.append(dict(bucket=">8192", rows=overflow, repetitive=overflow_rep,
                        rate=round(overflow_rep / overflow, 4) if overflow else 0.0))
    lanes = [dict(lane=lane, rows=entry["rows"], repetitive=entry["rep"],
                 rate=round(entry["rep"] / max(entry["rows"], 1), 4),
                 mean_words=round(entry["words"] / max(entry["rows"], 1)))
             for lane, entry in sorted(merged["lanes"].items(), key=lambda item: -item[1]["rows"])]
    payload = dict(rows=merged["rows"], with_reasoning=merged["with_reasoning"],
                   repetitive=merged["repetitive"],
                   rate=round(merged["repetitive"] / max(merged["with_reasoning"], 1), 4),
                   length_buckets=buckets, flags=dict(merged["flags"]), lanes=lanes)
    return payload


def scan_one(args):
    path, reasoning_key, lane_key = args
    return scan_file(path, reasoning_key, lane_key)


def command_scan(args):
    files = list(iter_inputs(args.inputs))
    if not files:
        raise SystemExit("no input files")
    with Pool(args.workers) as pool:
        results = pool.map(scan_one, [(str(path), args.reasoning_key, args.lane_key) for path in files])
    payload = merge_scan(results)
    if args.report:
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(dict(rows=payload["rows"], with_reasoning=payload["with_reasoning"],
                          repetitive=payload["repetitive"], rate=payload["rate"]), indent=1))
    for bucket in payload["length_buckets"]:
        print(f"  {bucket['bucket']:>10}: n={bucket['rows']} rep={bucket['repetitive']} rate={bucket['rate']}")
    print("  flags:", payload["flags"])
    return payload


def clean_file(args):
    path, options = args
    stats = Counter()
    statuses = Counter()
    with open_output(options["tmp"]) as handle:
        for row in read_jsonl(path):
            status, cleaned, reason = clean_row(row, **options["row_options"])
            statuses[status] += 1
            if status == "dropped":
                stats[reason] += 1
                continue
            handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    return dict(file=str(path), statuses=dict(statuses), dropped=dict(stats))


def command_clean(args):
    files = list(iter_inputs(args.inputs))
    if not files:
        raise SystemExit("no input files")
    tmp_dir = Path(args.tmp_dir or ".clean_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    row_options = dict(min_unique=args.min_unique_5gram, max_occ=args.max_occ,
                       max_removed_ratio=args.max_removed_ratio,
                       max_pingpong=args.max_pingpong, reasoning_key=args.reasoning_key)
    jobs = [(str(path), dict(tmp=str(tmp_dir / f"{index:04d}.jsonl"), row_options=row_options))
            for index, path in enumerate(files)]
    with Pool(args.workers) as pool:
        results = pool.map(clean_file, jobs)
    totals = Counter()
    dropped = Counter()
    for result in results:
        totals.update(result["statuses"])
        dropped.update(result["dropped"])
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, job in zip(files, jobs):
            target = out_dir / (Path(path).name + ".clean.jsonl")
            target.write_bytes(Path(job[1]["tmp"]).read_bytes())
            Path(job[1]["tmp"]).unlink()
    elif args.out:
        with open_output(args.out) as handle:
            for job in jobs:
                tmp = Path(job[1]["tmp"])
                handle.write(tmp.read_text(encoding="utf-8"))
                tmp.unlink()
    payload = dict(rows=sum(totals.values()), statuses=dict(totals), dropped=dict(dropped),
                   outputs=args.out_dir or args.out)
    if args.report:
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return payload


def priority(seed, *parts):
    body = "\x1f".join(map(str, (seed,) + parts)).encode("utf-8", "replace")
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "big")


def allocate_quotas(target, available, short_index=0, short_quota=0.25):
    """Return per-bucket quotas: short floor first, then natural shape, capped by supply."""
    quotas = [0] * len(available)
    total = sum(available)
    if total == 0 or target <= 0:
        return quotas
    short_target = min(int(round(target * short_quota)), available[short_index])
    quotas[short_index] = short_target
    remaining = target - short_target
    other_total = total - available[short_index]
    for index in range(len(available)):
        if index == short_index:
            continue
        share = available[index] / other_total if other_total else 0
        quotas[index] = min(available[index], int(math.floor(share * remaining)))
    # Fill rounding shortfall from buckets with spare supply, largest first.
    guard = 0
    while sum(quotas) < target and guard < 10000:
        guard += 1
        spare = [i for i in range(len(available)) if quotas[i] < available[i]]
        if not spare:
            break
        moved = 0
        for index in sorted(spare, key=lambda i: -available[i]):
            if sum(quotas) >= target:
                break
            quotas[index] += 1
            moved += 1
        if not moved:
            break
    return quotas


def bucket_index(words):
    for index, (lo, hi) in enumerate(LENGTH_BUCKETS):
        if lo <= words <= hi:
            return index
    return None


def rebuild_one(args):
    path, options, tmp = args
    entries = []
    with open(tmp, "wb") as handle:
        for row in read_jsonl(path):
            status, cleaned, reason = clean_row(row, **options["row_options"])
            if status == "dropped":
                entries.append(dict(status="dropped", reason=reason))
                continue
            words = len(words_of(locate_reasoning(cleaned, options["reasoning_key"]).text))
            if words > options["max_words"]:
                entries.append(dict(status="dropped", reason="too_long"))
                continue
            index = bucket_index(words)
            if index is None:
                entries.append(dict(status="dropped", reason="out_of_buckets"))
                continue
            payload = (json.dumps(cleaned, ensure_ascii=False) + "\n").encode("utf-8")
            offset = handle.tell()
            handle.write(payload)
            question = ""
            message_key, messages = messages_of(cleaned)
            for message in messages or []:
                if message.get("role") == "user":
                    question = message.get("content") or ""
                    break
            entries.append(dict(status="kept", offset=offset, bucket=index, words=words,
                                question=hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]))
    return dict(file=str(path), tmp=str(tmp), entries=entries,
                max_words=options["max_words"])


def command_rebuild(args):
    files = list(iter_inputs(args.inputs))
    if not files:
        raise SystemExit("no input files")
    tmp_dir = Path(args.tmp_dir or ".rebuild_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    row_options = dict(min_unique=args.min_unique_5gram, max_occ=args.max_occ,
                       max_removed_ratio=args.max_removed_ratio,
                       max_pingpong=args.max_pingpong, reasoning_key=args.reasoning_key)
    # --max-tokens is a token budget; the cleaner works in words, assume ~1.3 tokens/word.
    max_words = max(1, int(args.max_tokens / 1.3))
    jobs = [(str(path), dict(row_options=row_options, max_words=max_words,
                             reasoning_key=args.reasoning_key),
             str(tmp_dir / f"{index:04d}.jsonl")) for index, path in enumerate(files)]
    with Pool(args.workers) as pool:
        results = pool.map(rebuild_one, jobs)
    kept = []
    dropped = Counter()
    for result in results:
        for entry in result["entries"]:
            if entry["status"] == "kept":
                entry["tmp"] = result["tmp"]
                entry["source"] = result["file"]
                kept.append(entry)
            else:
                dropped[entry["reason"]] += 1
    available = [0] * len(LENGTH_BUCKETS)
    for entry in kept:
        available[entry["bucket"]] += 1
    quotas = allocate_quotas(args.target_rows, available, 0, args.short_quota)
    rng_priority = lambda entry: priority(args.seed, entry["source"], entry["offset"])
    ordered = defaultdict(list)
    for entry in kept:
        ordered[entry["bucket"]].append(entry)
    selected = []
    for index, quota in enumerate(quotas):
        group = sorted(ordered[index], key=rng_priority)
        selected.extend(group[:quota])
    selected.sort(key=rng_priority)
    handles = {}
    with open_output(args.out) as handle:
        for entry in selected:
            source = handles.get(entry["tmp"])
            if source is None:
                source = Path(entry["tmp"]).open("rb")
                handles[entry["tmp"]] = source
            source.seek(entry["offset"])
            handle.write(source.readline().decode("utf-8"))
    for source in handles.values():
        source.close()
    for job in jobs:
        Path(job[2]).unlink(missing_ok=True)
    profile = [dict(bucket=f"{lo}-{hi}", available=available[index], selected=quotas[index])
               for index, (lo, hi) in enumerate(LENGTH_BUCKETS)]
    payload = dict(inputs=len(files), kept_rows=len(kept), dropped=dict(dropped),
                   target_rows=args.target_rows, selected_rows=len(selected),
                   short_quota=args.short_quota, max_tokens=args.max_tokens,
                   max_words_estimate=max_words, buckets=profile, output=str(args.out))
    if args.report:
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="detection only; reproduce the report tables")
    scan.add_argument("inputs", nargs="+")
    scan.add_argument("--report", default=None)
    scan.add_argument("--reasoning-key", default=None)
    scan.add_argument("--lane-key", default=None, help="dotted field, e.g. source")
    scan.add_argument("--workers", type=int, default=1)
    scan.set_defaults(func=command_scan)

    clean = sub.add_parser("clean", help="stream jsonl -> jsonl with keep/rewrite/drop")
    clean.add_argument("inputs", nargs="+")
    clean.add_argument("--out", default=None)
    clean.add_argument("--out-dir", default=None)
    clean.add_argument("--report", default=None)
    clean.add_argument("--tmp-dir", default=None)
    clean.add_argument("--reasoning-key", default=None)
    clean.add_argument("--min-unique-5gram", type=float, default=0.88)
    clean.add_argument("--max-occ", type=int, default=DEFAULT_MAX_OCC)
    clean.add_argument("--max-removed-ratio", type=float, default=0.35)
    clean.add_argument("--max-pingpong", type=float, default=2.0)
    clean.add_argument("--workers", type=int, default=1)
    clean.set_defaults(func=command_clean)

    rebuild = sub.add_parser("rebuild-lane", help="two-pass clean + natural-shape selection")
    rebuild.add_argument("inputs", nargs="+")
    rebuild.add_argument("--out", required=True)
    rebuild.add_argument("--report", default=None)
    rebuild.add_argument("--target-rows", type=int, default=7390)
    rebuild.add_argument("--max-tokens", type=int, default=8192)
    rebuild.add_argument("--short-quota", type=float, default=0.25)
    rebuild.add_argument("--seed", type=int, default=42)
    rebuild.add_argument("--tmp-dir", default=None)
    rebuild.add_argument("--reasoning-key", default=None)
    rebuild.add_argument("--min-unique-5gram", type=float, default=0.88)
    rebuild.add_argument("--max-occ", type=int, default=DEFAULT_MAX_OCC)
    rebuild.add_argument("--max-removed-ratio", type=float, default=0.35)
    rebuild.add_argument("--max-pingpong", type=float, default=2.0)
    rebuild.add_argument("--workers", type=int, default=1)
    rebuild.set_defaults(func=command_rebuild)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
