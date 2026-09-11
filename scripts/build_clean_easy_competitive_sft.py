"""Build a low-memory SFT mixture led by cleaned Easy CompetitiveCoding rows.

The raw shards are indexed by byte offset in one streaming pass.  Only clearly
Easy problems are selected first (Codeforces rating <= 1200, or an explicit
introductory/easy label); a limited 1300-1600/Medium tier fills the byte target.
A question seen in any non-train split is excluded.

Competitive reasoning is shortened structurally: code blocks, answer-writing
meta talk, uncertain branches, and repeated sentences are removed.  The final
answer is normalized to one code block and statically checked.  Every passing
generation is retained, including repeated solutions for the same question.
UltraChat is used as clean
non-reasoning support data.  All final conversations are measured with the
project tokenizer and must fit the configured token limit.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "2")
import orjson
import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mix_continued_sft import LengthChecker, valid
from scripts.mix_large_continued_sft import fingerprints, has_reasoning
from scripts.mix_sft_datasets import ExternalRandomShuffler, iter_json_array


RAW_DIR = ROOT / "dataset/competitive-coding/data"
DEFAULT_INDEX = ROOT / ".cache/easy_competitive/offsets.sqlite"
DEFAULT_OUTPUT = ROOT / "dataset/sft_easy_medium_competitive_clean_4k.jsonl"
CODEALPACA = ROOT / "dataset/sft_codealpaca_20k.jsonl"
MAGICODER = ROOT / "dataset/magicoder-110k/data-evol_instruct-decontaminated.jsonl"
MATH_INSTRUCT = ROOT / "dataset/math-instruct/MathInstruct.json"
T2T_MINI = ROOT / "dataset/sft_t2t_mini.jsonl"


def log(message: str) -> None:
    rss = psutil.Process().memory_info().rss / 1024**2
    print(f"{time.strftime('%H:%M:%S')} {message}; RAM {rss:.0f} MiB", flush=True)


def normalized_splits(value) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(x).strip().lower() for x in values if x is not None]


def easy_difficulty(value) -> bool:
    text = str(value or "").strip().lower()
    if text in {"introductory", "easy", "beginner", "basic", "1", "1.0"}:
        return True
    if re.fullmatch(r"\d+", text):
        return int(text) <= 1200
    return False


def eligible_difficulty(value) -> bool:
    text = str(value or "").strip().lower()
    if easy_difficulty(value) or text in {"medium", "intermediate"}:
        return True
    return bool(re.fullmatch(r"\d+", text) and int(text) <= 1600)


def priority(seed: int, *parts: object) -> int:
    body = "\x1f".join(map(str, (seed,) + parts)).encode("utf-8", "replace")
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "big") & ((1 << 63) - 1)


def raw_shards() -> list[Path]:
    result = sorted(RAW_DIR.glob("competitive_programming_*.jsonl"))
    if len(result) != 4:
        raise FileNotFoundError(f"Expected four CompetitiveCoding shards, found {len(result)}")
    return result


def offset_manifest() -> list[dict]:
    paths = raw_shards() + sorted((ROOT / "dataset/ultrachat-200k/data").glob("train_sft-*.parquet"))
    paths += [ROOT / "model/tokenizer.json", ROOT / "model/tokenizer_config.json"]
    return [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths]


def manifest() -> list[dict]:
    indexed = offset_manifest()
    extras = [CODEALPACA, MAGICODER, MATH_INSTRUCT, T2T_MINI]
    return indexed + [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in extras]


def build_offset_index(path: Path, seed: int, rebuild: bool) -> dict:
    current_manifest = offset_manifest()
    if path.exists() and not rebuild:
        db = sqlite3.connect(path)
        try:
            settings = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM settings")}
            if (settings.get("complete") and settings.get("manifest") == current_manifest
                    and settings.get("difficulty_policy_version") == 2):
                log(f"Reusing complete offset index {path}")
                return settings
        except sqlite3.Error:
            pass
        finally:
            db.close()
        raise RuntimeError(f"Stale/incomplete index exists: {path}; rerun with --rebuild-index")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript("""
        CREATE TABLE candidates (
            file TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL,
            question TEXT NOT NULL, language TEXT NOT NULL, uuid TEXT,
            difficulty TEXT, source_name TEXT, priority INTEGER NOT NULL);
        CREATE TABLE heldout (question TEXT PRIMARY KEY);
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    stats = Counter()
    difficulty_counts = Counter()
    for shard in raw_shards():
        language = "python" if "python" in shard.name else "cpp"
        log(f"Scanning {shard.name}")
        with shard.open("rb", buffering=8 * 1024 * 1024) as handle:
            row_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row_number += 1
                if not line.strip():
                    continue
                try:
                    raw = orjson.loads(line)
                except orjson.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue
                stats["raw_rows"] += 1
                qid = raw.get("question_id") or hashlib.sha256(orjson.dumps(
                    [raw.get("dataset"), raw.get("source"), raw.get("index")])).hexdigest()
                splits = normalized_splits(raw.get("split"))
                is_train = bool(splits) and all(x == "train" for x in splits)
                if not is_train:
                    db.execute("INSERT OR IGNORE INTO heldout VALUES (?)", (qid,))
                    stats["non_train_rows"] += 1
                difficulty = str(raw.get("difficulty") or "")
                difficulty_counts[difficulty.lower()] += 1
                if is_train and eligible_difficulty(difficulty):
                    db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)", (
                        str(shard), offset, len(line), qid, language, raw.get("uuid"),
                        difficulty, str(raw.get("source") or ""),
                        priority(seed, qid, language, raw.get("uuid"), shard.name, row_number)))
                    tier = "easy" if easy_difficulty(difficulty) else "medium"
                    stats[tier + "_train_candidates"] += 1
                if row_number % 25000 == 0:
                    db.commit()
                    log(f"{shard.name}: {row_number:,} rows; easy/medium candidates "
                        f"{stats['easy_train_candidates'] + stats['medium_train_candidates']:,}")
        db.commit()
        log(f"Finished {shard.name}: {row_number:,} rows")
    db.execute("CREATE INDEX candidates_order ON candidates(language, priority)")
    db.execute("CREATE INDEX candidates_question ON candidates(question, language)")
    db.execute("CREATE INDEX heldout_question ON heldout(question)")
    eligible = db.execute("""SELECT count(*) FROM candidates c
        WHERE NOT EXISTS (SELECT 1 FROM heldout h WHERE h.question=c.question)""").fetchone()[0]
    unique_groups = db.execute("""SELECT count(*) FROM (
        SELECT question,language FROM candidates c
        WHERE NOT EXISTS (SELECT 1 FROM heldout h WHERE h.question=c.question)
        GROUP BY question,language)""").fetchone()[0]
    settings = {"complete": True, "manifest": current_manifest, "seed": seed,
                "difficulty_policy_version": 2,
                "stats": dict(stats), "difficulty_counts": dict(difficulty_counts),
                "eligible_rows": eligible, "unique_question_language": unique_groups}
    db.executemany("INSERT INTO settings VALUES (?,?)",
                   [(k, json.dumps(v, ensure_ascii=False)) for k, v in settings.items()])
    db.commit()
    db.close()
    log(f"Offset index complete: {eligible:,} eligible rows, {unique_groups:,} unique question/language groups")
    return settings


PROMPT_PREFIX = re.compile(
    r"\AYou are a helpful and harmless assistant\..*?You must use ```(?:python|cpp).*?"
    r"```(?:python|cpp)?\s*\n.*?\n```\s*\n+", re.I | re.S)
FENCE = re.compile(r"```\s*([A-Za-z0-9_+.-]*)\s*\n(.*?)```", re.S)
UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|possibly|not sure|unclear|wait|hold on|could be|might be|I think)\b", re.I)
META = re.compile(
    r"\b(?:the user|as chatgpt|final answer|final response|final code|code block|"
    r"must output|need to output|allowed methods|instruction below|let'?s craft|now craft)\b", re.I)
CODE_LINE = re.compile(
    r"^\s*(?:#include|using namespace|int main|def main|if __name__|import [A-Za-z_.]+|"
    r"from [A-Za-z_.]+ import|class\s+\w+|(?:for|while|if)\s*\(.*\)\s*\{|return\s+[^.;]+[;]?)")


def clean_prompt(text: str, language: str) -> str | None:
    problem = PROMPT_PREFIX.sub("", text.strip(), count=1).strip()
    if problem == text.strip():
        # The four raw shards currently use a fixed preamble.  Do not guess if it changes.
        return None
    if not 120 <= len(problem) <= 12000:
        return None
    language_name = "Python" if language == "python" else "C++"
    return (f"Solve the following programming problem in {language_name}. "
            f"Return the final solution as one `{language}` code block.\n\n{problem}")


def extract_final_code(content: str, language: str) -> tuple[str, str] | None:
    matches = FENCE.findall(content.strip())
    compatible = [(tag.lower(), code.strip()) for tag, code in matches
                  if tag.lower() in ({"python", "py", ""} if language == "python" else {"cpp", "c++", "cc", ""})]
    if len(matches) != 1 or len(compatible) != 1:
        return None
    _, code = compatible[0]
    if not 40 <= len(code) <= 10000:
        return None
    lowered = code.lower()
    if any(x in lowered for x in ("your code here", "todo", "solution goes here")) or "..." in code:
        return None
    if language == "python":
        try:
            ast.parse(code)
        except SyntaxError:
            return None
    else:
        if not re.search(r"\b(?:int|signed)\s+main\s*\(", code):
            return None
        if code.count("{") != code.count("}"):
            return None
    return code, f"```{language}\n{code}\n```"


def sentence_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def clean_reasoning(text: str, final_code: str) -> tuple[str | None, str]:
    if not isinstance(text, str) or not text.strip():
        return None, "missing_reasoning"
    text = text.replace("\x00", " ").replace("\ufffd", " ")
    text = FENCE.sub("\n", text)
    text = re.sub(r"<think>|</think>", "", text, flags=re.I)
    # Work at sentence/line boundaries so the result never ends mid-thought.
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept, seen = [], set()
    for piece in pieces:
        piece = re.sub(r"\s+", " ", piece).strip(" -*#\t")
        if len(piece) < 18:
            continue
        if META.search(piece) or UNCERTAIN.search(piece) or CODE_LINE.search(piece):
            continue
        if piece.count("{") + piece.count("}") + piece.count(";") >= 3:
            continue
        piece = re.sub(r"^(?:So,?\s+)?We need to\s+", "", piece, flags=re.I)
        piece = re.sub(r"^We can\s+", "", piece, flags=re.I)
        piece = re.sub(r"^Let'?s\s+", "", piece, flags=re.I)
        if piece:
            piece = piece[0].upper() + piece[1:]
        key = sentence_key(piece)
        if len(key) < 15 or key in seen:
            continue
        seen.add(key)
        kept.append(piece)
        if sum(len(x) + 1 for x in kept) >= 3200:
            break
    cleaned = " ".join(kept).strip()
    if not 160 <= len(cleaned) <= 3600:
        return None, "reasoning_length"
    words = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", cleaned.lower())
    if len(words) >= 30:
        grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
        if len(set(grams)) / len(grams) < 0.88:
            return None, "reasoning_repetition"
    code_tokens = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", final_code.lower()))
    prose_tokens = set(words)
    if len(prose_tokens) > 20 and len(prose_tokens & code_tokens) / len(prose_tokens) > 0.72:
        return None, "reasoning_code_overlap"
    return cleaned, "ok"


FINAL_MATH = re.compile(
    r"(?is)^(.*?)(?:\n|^)((?:the answer is|answer:|therefore,? the answer is|thus,? the answer is).{1,160})\s*$")


def normalize_math(row: dict) -> dict | None:
    instruction = str(row.get("instruction") or "").strip()
    output = str(row.get("output") or "").strip()
    if not instruction or not output or len(instruction) > 10000 or len(output) > 5000:
        return None
    if "print(" in output and ("\n#" in output or output.lstrip().startswith("#")):
        try:
            ast.parse(output)
        except SyntaxError:
            pass
        else:
            comments = [re.sub(r"^\s*#\s?", "", line).strip()
                        for line in output.splitlines() if re.match(r"^\s*#", line)]
            reasoning = ". ".join(x.rstrip(".") for x in comments if len(x) >= 8).strip()
            reasoning = re.sub(r"\s+", " ", reasoning)
            if 50 <= len(reasoning) <= 2400 and not UNCERTAIN.search(reasoning) and not META.search(reasoning):
                return {"conversations": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": f"```python\n{output}\n```",
                     "reasoning_content": reasoning}]}
    match = FINAL_MATH.match(output)
    if match:
        reasoning, answer = match.group(1).strip(), match.group(2).strip()
    else:
        # MathInstruct combines many sources with different answer markers.  Use
        # only a complete final line/sentence as the visible answer; never cut at
        # an arbitrary character limit.
        parts = [x.strip() for x in re.split(r"\n+", output) if x.strip()]
        if len(parts) < 2:
            parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", output) if x.strip()]
        if len(parts) < 2 or not 1 <= len(parts[-1]) <= 300:
            return None
        reasoning, answer = " ".join(parts[:-1]), parts[-1]
        if not re.search(r"\d|[A-E]\b|\\boxed|=|true|false|yes|no", answer, re.I):
            return None
    reasoning = re.sub(r"(?i)^let'?s (?:think about|solve) (?:the )?(?:multi-choice )?question(?: step by step)?\.\s*",
                       "", reasoning)
    reasoning = re.sub(r"\s+", " ", reasoning).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    if not 30 <= len(reasoning) <= 3000 or UNCERTAIN.search(reasoning) or META.search(reasoning):
        return None
    return {"conversations": [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning}]}


def normalize_magicoder(row: dict) -> dict | None:
    instruction = str(row.get("instruction") or "").strip()
    response = str(row.get("response") or "").strip()
    if not instruction or not response or len(instruction) > 12000 or len(response) > 12000:
        return None
    fences = FENCE.findall(response)
    outside = FENCE.sub("\n", response)
    outside = re.sub(r"\s+", " ", outside).strip()
    # When there is one code block plus a concise explanation, make the explanation
    # hidden reasoning and leave the code as the final response.  Otherwise preserve
    # the original response without inventing a rationale.
    if len(fences) == 1 and 80 <= len(outside) <= 2400 and not UNCERTAIN.search(outside) and not META.search(outside):
        tag, code = fences[0]
        tag = tag.strip().lower()
        code = code.strip()
        if code and "..." not in code and "your code here" not in code.lower():
            content = f"```{tag}\n{code}\n```" if tag else f"```\n{code}\n```"
            return {"conversations": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": content, "reasoning_content": outside}]}
    return {"conversations": [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}]}


T2T_META = re.compile(
    r"(?:用户(?:问|想|希望|需要|要求)|分析用户|这位用户|the user (?:asks|wants|needs)|"
    r"作为(?:一个)?(?:AI|助手)|as (?:an )?ai|I need to (?:answer|respond)|"
    r"首先.{0,8}分析.{0,8}需求|接下来.{0,12}考虑|最后.{0,8}推荐)", re.I)
MIXED_GARBAGE = re.compile(r"(?:[A-Za-z]{4,}[\u3400-\u9fff]|[\u3400-\u9fff][A-Za-z]{4,})")
MIXED_ALLOW = re.compile(
    r"(?:Python|JavaScript|TypeScript|Java|Linux|Windows|Android|iOS|API|JSON|HTML|CSS|SQL|"
    r"GitHub|App|ChatGPT|OpenAI|Transformer|Token|Web|HTTP|HTTPS|VPN|JVM|GUI|Hadoop|Spark)[\u3400-\u9fff]",
    re.I)


def bad_text_quality(text: str) -> bool:
    if not isinstance(text, str) or not text.strip() or "\ufffd" in text or "\x00" in text:
        return True
    lowered = text.lower()
    if any(x in lowered for x in ("usuario", "uptake外", "郁strong", "天numer", "<unk>")):
        return True
    mixed = MIXED_GARBAGE.findall(text)
    allowed = MIXED_ALLOW.findall(text)
    if len(mixed) - len(allowed) >= 2:
        return True
    words = re.findall(r"[A-Za-z\u3400-\u9fff]+", lowered)
    if len(words) >= 60:
        grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and len(set(grams)) / len(grams) < 0.82:
            return True
    return False


def normalize_t2t(row: dict) -> tuple[dict | None, str]:
    if not valid(row):
        return None, "invalid_structure"
    # Copy because rejected reasoning is removed in place while the source remains read-only.
    cleaned = orjson.loads(orjson.dumps(row))
    messages = cleaned.get("conversations", [])
    if len(messages) > 12:
        return None, "too_many_turns"
    total_chars = 0
    last_user = ""
    retained_reasoning = False
    removed_reasoning = False
    for message in messages:
        content = str(message.get("content") or "").strip()
        message["content"] = content
        total_chars += len(content)
        if bad_text_quality(content):
            return None, "bad_visible_text"
        if message.get("role") == "user":
            last_user = content
        if message.get("role") != "assistant":
            continue
        thought = str(message.get("reasoning_content") or "").strip()
        if not thought:
            message.pop("reasoning_content", None)
            continue
        reject_reasoning = (
            not 60 <= len(thought) <= 1600 or bad_text_quality(thought) or
            bool(T2T_META.search(thought)) or bool(UNCERTAIN.search(thought)))
        if "```" in thought and not re.search(r"代码|程序|编程|code|program|script", last_user, re.I):
            reject_reasoning = True
        compact_thought = re.sub(r"\s+", " ", thought)
        compact_answer = re.sub(r"\s+", " ", content)
        if len(compact_answer) >= 80 and compact_answer[:80] in compact_thought:
            reject_reasoning = True
        if reject_reasoning:
            message.pop("reasoning_content", None)
            removed_reasoning = True
        else:
            message["reasoning_content"] = thought
            retained_reasoning = True
    if total_chars > 24000:
        return None, "visible_text_too_long"
    if retained_reasoning:
        return cleaned, "cot_retained"
    return cleaned, "bad_cot_removed" if removed_reasoning else "plain_retained"


def read_raw(file_handles: dict[str, object], file: str, offset: int, length: int) -> dict:
    handle = file_handles.get(file)
    if handle is None:
        handle = open(file, "rb", buffering=1024 * 1024)
        file_handles[file] = handle
    handle.seek(offset)
    line = handle.read(length)
    return orjson.loads(line)


def build_dataset(index_path: Path, output: Path, seed: int, max_tokens: int,
                  target_bytes: int, competitive_target_bytes: int) -> dict:
    for path in (output, output.with_suffix(".sources.jsonl"), output.with_suffix(".report.json")):
        if path.exists():
            raise FileExistsError(path)
    payload = output.with_suffix(".payload.jsonl")
    if payload.exists():
        raise FileExistsError(payload)
    db = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    checker = LengthChecker(ROOT / "model")
    shuffler = ExternalRandomShuffler(payload, seed + 17, 96)
    handles: dict[str, object] = {}
    seen_fp = set()
    stats = defaultdict(Counter)
    selected_bytes = 0

    def emit(source: str, row: dict, meta: dict, tokens: int, allow_duplicate: bool = False) -> bool:
        nonlocal selected_bytes
        fp, _ = fingerprints(row)
        if (fp in seen_fp and not allow_duplicate) or not valid(row) or not 0 < tokens <= max_tokens:
            return False
        body = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        size = len(body.encode("utf-8")) + 1
        meta = dict(meta, source=source, tokens=tokens, bytes=size,
                    fingerprint=fp.hex(), has_reasoning=bool(has_reasoning(row)))
        shuffler.add({"row": row, "meta": meta})
        seen_fp.add(fp)
        selected_bytes += size
        stats[source]["selected"] += 1
        stats[source]["bytes"] += size
        stats[source]["tokens"] += tokens
        stats[source]["reasoning_selected"] += int(meta["has_reasoning"])
        stats[source]["max_tokens"] = max(stats[source]["max_tokens"], tokens)
        return True

    def emit_stream(source: str, iterable, byte_cap: int | None = None) -> None:
        """Token-check and emit a normalized source with small bounded batches."""
        batch = []
        for row, meta in iterable:
            stats[source]["considered"] += 1
            if row is None or not valid(row):
                stats[source]["invalid"] += 1
                continue
            batch.append((row, meta))
            if len(batch) < 32:
                continue
            lengths = checker.count_batch([x[0] for x in batch])
            for (candidate, candidate_meta), tokens in zip(batch, lengths):
                if byte_cap is not None and stats[source]["bytes"] >= byte_cap:
                    return
                if tokens <= max_tokens:
                    if not emit(source, candidate, candidate_meta, tokens):
                        stats[source]["duplicate"] += 1
                else:
                    stats[source]["over_token_limit"] += 1
            batch.clear()
        if batch:
            lengths = checker.count_batch([x[0] for x in batch])
            for (candidate, candidate_meta), tokens in zip(batch, lengths):
                if byte_cap is not None and stats[source]["bytes"] >= byte_cap:
                    return
                if tokens <= max_tokens:
                    if not emit(source, candidate, candidate_meta, tokens):
                        stats[source]["duplicate"] += 1
                else:
                    stats[source]["over_token_limit"] += 1

    log("Selecting and cleaning Easy-first CompetitiveCoding candidates")
    cursors = {}
    for language in ("python", "cpp"):
        cursors[language] = db.execute("""SELECT file,offset,length,question,uuid,difficulty,source_name
            FROM candidates c WHERE language=?
            AND NOT EXISTS (SELECT 1 FROM heldout h WHERE h.question=c.question)
            ORDER BY CASE
                WHEN lower(difficulty) IN ('introductory','easy','beginner','basic','1','1.0') THEN 0
                WHEN CAST(difficulty AS INTEGER) <= 900 THEN 1
                WHEN CAST(difficulty AS INTEGER) = 1000 THEN 2
                WHEN CAST(difficulty AS INTEGER) = 1100 THEN 3
                ELSE 4 END, priority""", (language,))
    exhausted = set()
    last_log = time.monotonic()
    while selected_bytes < competitive_target_bytes and len(exhausted) < 2:
        for language in ("python", "cpp"):
            if language in exhausted or selected_bytes >= competitive_target_bytes:
                continue
            batch = cursors[language].fetchmany(32)
            if not batch:
                exhausted.add(language)
                continue
            pending = []
            for file, offset, length, question, uuid, difficulty, source_name in batch:
                stats["competitive_scan"]["considered"] += 1
                try:
                    raw = read_raw(handles, file, offset, length)
                    messages = raw.get("messages")
                    if not isinstance(messages, list) or len(messages) != 2:
                        stats["competitive_scan"]["invalid_messages"] += 1
                        continue
                    prompt = clean_prompt(str(messages[0].get("content") or ""), language)
                    final = extract_final_code(str(messages[1].get("content") or ""), language)
                    if not prompt or not final:
                        stats["competitive_scan"]["prompt_or_code_rejected"] += 1
                        continue
                    code, content = final
                    normalized_code = re.sub(r"\s+", "", code)
                    solution_hash = hashlib.sha256(normalized_code.encode("utf-8", "replace")).hexdigest()
                    reasoning, reason = clean_reasoning(messages[1].get("reasoning_content"), code)
                    if not reasoning:
                        stats["competitive_scan"][reason] += 1
                        continue
                    row = {"conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": content, "reasoning_content": reasoning}]}
                    meta = {"file": file, "offset": offset, "uuid": uuid,
                            "question_id": question, "language": language,
                            "difficulty": difficulty, "original_source": source_name,
                            "solution_code_sha256": solution_hash,
                            "reasoning_cleaning": "remove code/meta/uncertainty/repeated sentences; complete-boundary cap"}
                    pending.append((row, meta))
                except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                    stats["competitive_scan"]["read_or_parse_error"] += 1
            lengths = checker.count_batch([x[0] for x in pending]) if pending else []
            for (row, meta), tokens in zip(pending, lengths):
                if tokens > max_tokens:
                    stats["competitive_scan"]["over_token_limit"] += 1
                    continue
                if emit("competitive_easy_medium_clean", row, meta, tokens, allow_duplicate=True):
                    tier = "easy" if easy_difficulty(meta["difficulty"]) else "medium"
                    stats["competitive_difficulty"][tier] += 1
            if time.monotonic() - last_log > 20:
                log(f"Competitive selected {stats['competitive_easy_medium_clean']['selected']:,}, "
                    f"{stats['competitive_easy_medium_clean']['bytes']/1e9:.3f}/{competitive_target_bytes/1e9:.3f} GB")
                last_log = time.monotonic()
    for handle in handles.values():
        handle.close()
    competitive_rows = stats["competitive_easy_medium_clean"]["selected"]
    log(f"Clean Competitive ready: {competitive_rows:,} rows, {stats['competitive_easy_medium_clean']['bytes']/1e9:.3f} GB")

    log("Adding CodeAlpaca replay")
    def codealpaca_rows():
        with CODEALPACA.open("rb") as source:
            for row_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield orjson.loads(line), {"file": str(CODEALPACA), "row": row_number}
                    except orjson.JSONDecodeError:
                        stats["codealpaca_replay"]["invalid_json"] += 1
    emit_stream("codealpaca_replay", codealpaca_rows(), 8_000_000)

    log("Adding Magicoder replay")
    def magicoder_rows():
        with MAGICODER.open("rb") as source:
            for row_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield normalize_magicoder(orjson.loads(line)), {"file": str(MAGICODER), "row": row_number}
                    except orjson.JSONDecodeError:
                        stats["magicoder_replay"]["invalid_json"] += 1
    emit_stream("magicoder_replay", magicoder_rows(), 230_000_000)

    log("Adding MathInstruct reasoning replay")
    def math_rows():
        for row_number, raw in enumerate(iter_json_array(MATH_INSTRUCT), 1):
            yield normalize_math(raw), {"file": str(MATH_INSTRUCT), "row": row_number,
                                        "original_source": raw.get("source")}
    emit_stream("math_replay_clean_cot", math_rows(), 210_000_000)

    log("Adding a cleaned T2T-mini replay sample")
    t2t_cap = 300_000_000
    t2t_batch = []
    with T2T_MINI.open("rb", buffering=8 * 1024 * 1024) as source:
        for row_number, line in enumerate(source, 1):
            if stats["t2t_clean_cot"]["bytes"] + stats["t2t_clean_plain"]["bytes"] >= t2t_cap:
                break
            if not line.strip() or priority(seed + 91, row_number) % 3:
                continue
            try:
                row, status = normalize_t2t(orjson.loads(line))
            except orjson.JSONDecodeError:
                stats["t2t_scan"]["invalid_json"] += 1
                continue
            stats["t2t_scan"][status] += 1
            if row is None:
                continue
            source_name = "t2t_clean_cot" if has_reasoning(row) else "t2t_clean_plain"
            t2t_batch.append((source_name, row, {"file": str(T2T_MINI), "row": row_number,
                                                 "cleaning_status": status}))
            if len(t2t_batch) < 32:
                continue
            lengths = checker.count_batch([x[1] for x in t2t_batch])
            for (source_name, candidate, meta), tokens in zip(t2t_batch, lengths):
                if tokens <= max_tokens:
                    emit(source_name, candidate, meta, tokens)
                else:
                    stats["t2t_scan"]["over_token_limit"] += 1
            t2t_batch.clear()
    if t2t_batch:
        lengths = checker.count_batch([x[1] for x in t2t_batch])
        for (source_name, candidate, meta), tokens in zip(t2t_batch, lengths):
            if tokens <= max_tokens:
                emit(source_name, candidate, meta, tokens)
            else:
                stats["t2t_scan"]["over_token_limit"] += 1

    base_rows = sum(x["selected"] for x in stats.values())
    base_reasoning = sum(x["reasoning_selected"] for x in stats.values())
    log(f"Replay ready: {base_rows:,} rows, reasoning {base_reasoning/base_rows:.2%}, total {selected_bytes/1e9:.3f} GB")

    # Prefer longer clean UltraChat conversations.  Match CompetitiveCoding by
    # serialized bytes while preserving the requested strict reasoning majority.
    # with fewer non-reasoning rows and therefore preserves the requested strict
    # reasoning majority without duplicating CoT samples.
    log("Adding UltraChat support conversations, longer rows first")
    ultra_target_bytes = stats["competitive_easy_medium_clean"]["bytes"]
    ultra_paths = sorted((ROOT / "dataset/ultrachat-200k/data").glob("train_sft-*.parquet"))
    size_bands = [(12000, None), (9000, 12000), (7000, 9000), (0, 7000)]
    stop_ultra = False
    for minimum_bytes, maximum_bytes in size_bands:
        if stop_ultra:
            break
        for parquet_path in ultra_paths:
            row_number = 0
            for batch in pq.ParquetFile(parquet_path).iter_batches(columns=["messages"], batch_size=64):
                candidates = []
                for raw in batch.to_pylist():
                    row_number += 1
                    if stats["ultrachat_clean"]["bytes"] >= ultra_target_bytes:
                        stop_ultra = True
                        break
                    current_rows = sum(x["selected"] for x in stats.values())
                    current_reasoning = sum(x["reasoning_selected"] for x in stats.values())
                    if 2 * current_reasoning <= current_rows + 1:
                        stop_ultra = True
                        break
                    row = {"conversations": [{"role": m["role"], "content": (m.get("content") or "").strip()}
                                             for m in raw["messages"]]}
                    if not valid(row) or has_reasoning(row):
                        continue
                    serialized_size = len(orjson.dumps(row)) + 1
                    if serialized_size < minimum_bytes or (maximum_bytes is not None and serialized_size >= maximum_bytes):
                        continue
                    candidates.append((row, {"file": str(parquet_path), "row": row_number,
                                             "length_priority_band": minimum_bytes}))
                lengths = checker.count_batch([x[0] for x in candidates]) if candidates else []
                for (row, meta), tokens in zip(candidates, lengths):
                    if stats["ultrachat_clean"]["bytes"] >= ultra_target_bytes:
                        stop_ultra = True
                        break
                    if tokens <= max_tokens:
                        emit("ultrachat_clean", row, meta, tokens)
                if stop_ultra:
                    break
                if time.monotonic() - last_log > 20:
                    log(f"UltraChat selected {stats['ultrachat_clean']['selected']:,}; total {selected_bytes/1e9:.3f} GB")
                    last_log = time.monotonic()
            if stop_ultra:
                break
    db.close()
    if selected_bytes < 1_600_000_000:
        shuffler.finish()
        raise RuntimeError(
            f"Strict clean pool reached only {selected_bytes/1e9:.3f} GB; "
            "do not weaken filters silently. Inspect the reportable selection counters.")
    shuffler.finish()

    log("Writing globally shuffled JSONL and aligned provenance")
    temporary = output.with_suffix(".jsonl.tmp")
    sources_tmp = output.with_suffix(".sources.jsonl.tmp")
    counts, thoughts, token_values, language_counts = Counter(), Counter(), defaultdict(list), Counter()
    digest = hashlib.sha256()
    with payload.open("rb") as source, temporary.open("wb") as target, sources_tmp.open("wb") as provenance:
        for line in source:
            item = orjson.loads(line)
            body = orjson.dumps(item["row"], option=orjson.OPT_APPEND_NEWLINE)
            target.write(body)
            digest.update(body)
            provenance.write(orjson.dumps(item["meta"], option=orjson.OPT_APPEND_NEWLINE))
            meta = item["meta"]
            counts[meta["source"]] += 1
            thoughts[meta["source"]] += int(meta["has_reasoning"])
            token_values[meta["source"]].append(meta["tokens"])
            if meta["source"] == "competitive_easy_medium_clean":
                language_counts[meta["language"]] += 1
    final_bytes = temporary.stat().st_size
    total_rows = sum(counts.values())
    reasoning_rows = sum(thoughts.values())
    assert 1_600_000_000 <= final_bytes <= 5_000_000_000
    assert reasoning_rows / total_rows > 0.5
    token_stats = {}
    for source, values in token_values.items():
        values.sort()
        token_stats[source] = {"total": sum(values), "mean": sum(values)/len(values),
                               "p50": values[len(values)//2], "p95": values[int(len(values)*.95)],
                               "max": max(values)}
    report = {
        "output": str(output), "bytes": final_bytes, "gb": final_bytes/1e9,
        "gib": final_bytes/1024**3, "sha256": digest.hexdigest(), "seed": seed,
        "max_tokens": max_tokens, "total_rows": total_rows, "counts": dict(counts),
        "nonempty_reasoning_rows": dict(thoughts),
        "reasoning_fraction": reasoning_rows/total_rows,
        "token_stats": token_stats, "competitive_languages": dict(language_counts),
        "selection_stats": {k: dict(v) for k, v in stats.items()},
        "policy": {
            "difficulty": "Easy/introductory and Codeforces <=1200 first; Medium/intermediate and ratings 1300-1600 only fill the 460 MB Competitive target",
            "splits": "exclude any question ID observed in any non-train split",
            "deduplication": "retain every passing CompetitiveCoding generation, including repeated solutions under the same question ID; other replay sources use exact-conversation dedupe",
            "reasoning": "remove code blocks, answer-writing meta talk, uncertainty, repeated/code-like sentences; cap at complete sentence boundary",
            "answer_checks": "source answers assumed correct by user; one language-matching code fence; Python AST parse; C++ main/braces static check; no execution judge",
            "prompt": "remove dataset boilerplate and replace it with a short language-specific instruction",
            "length": f"full rendered conversation <= {max_tokens} tokens; no truncation",
            "t2t": "one-third deterministic hash sample until 300 MB; reject corrupt visible text; retain only short non-meta CoT, otherwise remove bad reasoning and keep a clean final answer as plain SFT",
            "source_balance": "UltraChat serialized JSONL bytes matched to CompetitiveCoding serialized JSONL bytes; Easy first, Medium only fills the Competitive target",
            "mix": "Easy-first CompetitiveCoding with limited Medium, cleaned T2T-mini, CodeAlpaca/Magicoder/MathInstruct replay, and byte-matched UltraChat; strictly >50% nonempty-reasoning rows"
        },
        "inputs": manifest(), "index": str(index_path),
        "peak_process_memory_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
    }
    temporary.replace(output)
    sources_tmp.replace(output.with_suffix(".sources.jsonl"))
    output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    payload.unlink()
    log(json.dumps({k: report[k] for k in ("output", "gb", "total_rows", "counts", "reasoning_fraction")}, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--target-gb", type=float, default=1.6)
    parser.add_argument("--competitive-target-gb", type=float, default=0.46)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    settings = build_offset_index(args.index, args.seed, args.rebuild_index)
    print(json.dumps({k: settings[k] for k in ("stats", "eligible_rows", "unique_question_language")},
                     ensure_ascii=False, indent=2), flush=True)
    if not args.index_only:
        build_dataset(args.index, args.output, args.seed, args.max_tokens,
                      int(args.target_gb * 1e9), int(args.competitive_target_gb * 1e9))


if __name__ == "__main__":
    main()
