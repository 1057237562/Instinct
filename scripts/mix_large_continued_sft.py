"""Build a size-targeted continued SFT dataset, with a reusable SQLite candidate cache.

Default: 1.6 GiB, 50/30/20 by serialized UTF-8 JSONL bytes for competitive
coding / T2T replay / unseen UltraChat. Competitive reasoning takes 70% of
the competitive byte budget; final-answer-only rows take the remainder.
More than 50% of output conversations contain nonempty reasoning. First
reach at least 55% reasoning using replay, then add balanced replay pairs
to meet the byte target while staying strictly above 50%.
Every full rendered conversation must fit 16,384 tokens. Random selection
is without replacement; at most three different answers per problem/language.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time

os.environ.setdefault('RAYON_NUM_THREADS', '8')
import orjson
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mix_continued_sft import LengthChecker, valid


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def fingerprints(row):
    plain = [{'role': m['role'], 'content': m.get('content', '').strip()}
             for m in row['conversations']]
    prompt = next((m['content'] for m in plain if m['role'] == 'user'), '')
    return (hashlib.sha256(orjson.dumps(plain, option=orjson.OPT_SORT_KEYS)).digest(),
            hashlib.sha256(prompt.encode()).digest())


def has_reasoning(row):
    return any(m.get('reasoning_content', '').strip() or
               any(x.strip() for x in re.findall(r'<think>(.*?)</think>', m.get('content', ''), re.S))
               for m in row['conversations'] if m['role'] == 'assistant')


def input_manifest():
    paths = [ROOT / 'dataset/sft_magicoder110k_mathinstruct_ultrachat20.jsonl',
             ROOT / 'dataset/sft_t2t_mini.jsonl']
    code = sorted((ROOT / 'dataset/competitive-coding/data').glob('competitive_programming_*.jsonl'))
    ultra = sorted((ROOT / 'dataset/ultrachat-200k/data').glob('train_sft-*.parquet'))
    if not code or not ultra:
        raise FileNotFoundError('Missing competitive coding or UltraChat train_sft shards')
    paths += code + ultra + [ROOT / 'model/tokenizer.json', ROOT / 'model/tokenizer_config.json']
    return [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]


def build_cache(db, manifest, seed, max_tokens):
    db.executescript('''
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE candidates (
            source TEXT, priority INTEGER, fp BLOB, body TEXT, meta TEXT,
            question TEXT, language TEXT, tokens INTEGER);
        CREATE TABLE heldout (question TEXT PRIMARY KEY);
    ''')
    rng = random.Random(seed)
    known, prompts = set(), set()
    stats = Counter()
    def stage(source, row, meta, question=None, language=None):
        fp, _ = fingerprints(row)
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?,NULL)',
                   (source, rng.getrandbits(63), fp, dumps(row), dumps(meta), question, language))
        stats[source + '_candidates'] += 1

    previous, replay = (Path(x['path']) for x in manifest[:2])
    log('Indexing previously trained mixture')
    with previous.open('rb') as f:
        for line in f:
            if not line.strip():
                continue
            fp, prompt = fingerprints(orjson.loads(line))
            known.add(fp)
            prompts.add(prompt)
    log(f'Previously trained mixture indexed: {len(known):,} unique conversations')
    with replay.open('rb') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = orjson.loads(line)
            fp, prompt = fingerprints(row)
            known.add(fp)
            prompts.add(prompt)
            if valid(row):
                stage('t2t_replay', row, {'file': str(replay), 'row': i})
            else:
                stats['t2t_invalid'] += 1
            if i % 100000 == 0:
                db.commit()
                log(f'T2T indexed {i:,} rows')
    db.commit()

    for item in manifest:
        path = Path(item['path'])
        if not path.name.startswith('competitive_programming_'):
            continue
        language = 'python' if 'python' in path.name else 'cpp'
        with path.open('rb') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                raw = orjson.loads(line)
                stats['competitive_raw'] += 1
                question = raw.get('question_id') or hashlib.sha256(
                    orjson.dumps([raw.get('dataset'), raw.get('index')])).hexdigest()
                splits = raw.get('split')
                splits = splits if isinstance(splits, list) else [splits]
                if not splits or any(s != 'train' for s in splits):
                    db.execute('INSERT OR IGNORE INTO heldout VALUES (?)', (question,))
                    stats['competitive_non_train'] += 1
                else:
                    row = {'conversations': [{'role': m['role'], 'content': m['content'].strip()}
                                             for m in raw['messages']]}
                    if not valid(row):
                        stats['competitive_invalid'] += 1
                        continue
                    fp, prompt = fingerprints(row)
                    if fp in known or prompt in prompts:
                        stats['competitive_previous_overlap'] += 1
                        continue
                    meta = {'file': str(path), 'row': i, 'uuid': raw.get('uuid'),
                            'question_id': question, 'language': language,
                            'split': raw.get('split'), 'dataset': raw.get('dataset'),
                            'difficulty': raw.get('difficulty'), 'license': raw.get('license')}
                    stage('competitive_final', row, meta, question, language)
                    reasoning = raw['messages'][-1].get('reasoning_content')
                    if isinstance(reasoning, str) and reasoning.strip():
                        # A conservative candidate filter; exact tokens are measured later.
                        chars = sum(len(m['content']) for m in row['conversations']) + len(reasoning.strip())
                        if chars <= max_tokens * 4:
                            row['conversations'][-1]['reasoning_content'] = reasoning.strip()
                            stage('competitive_cot', row, meta, question, language)
                        else:
                            stats['competitive_cot_character_prefilter'] += 1
                if i % 25000 == 0:
                    db.commit()
                    log(f'{path.name}: indexed {i:,}; short CoT candidates {stats["competitive_cot_candidates"]:,}')
        db.commit()
        log(f'Finished indexing {path.name}')

    for item in manifest:
        path = Path(item['path'])
        if not path.name.startswith('train_sft-'):
            continue
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=['messages'], batch_size=1024):
            for raw in batch.to_pylist():
                index += 1
                row = {'conversations': [{'role': m['role'], 'content': m['content'].strip()}
                                         for m in raw['messages']]}
                if not valid(row):
                    stats['ultrachat_invalid'] += 1
                    continue
                fp, prompt = fingerprints(row)
                if fp in known or prompt in prompts:
                    stats['ultrachat_previous_overlap'] += 1
                    continue
                stage('ultrachat_new', row, {'file': str(path), 'row': index})
        db.commit()
        log(f'Finished indexing {path.name}')
    log('Creating random-order candidate index')
    db.execute('CREATE INDEX candidate_order ON candidates(source, priority)')
    settings = {'manifest': manifest, 'seed': seed, 'max_tokens': max_tokens, 'stats': dict(stats), 'complete': True}
    db.executemany('INSERT INTO settings VALUES (?,?)', [(k, json.dumps(v)) for k, v in settings.items()])
    db.commit()
    log('Reusable candidate cache complete')


def select_source(db, source, target_bytes, checker, max_tokens, seen, groups, group_cap, rng):
    stats = Counter()
    selected_bytes = 0
    cursor = db.execute('''SELECT c.rowid,c.fp,c.body,c.meta,c.question,c.language,c.tokens
        FROM candidates c WHERE source=?
        AND (c.question IS NULL OR NOT EXISTS (SELECT 1 FROM heldout h WHERE h.question=c.question))
        ORDER BY priority''', (source,))
    last_log = time.monotonic()
    done = False
    while not done:
        records = cursor.fetchmany(128)
        if not records:
            break
        eligible, batch_seen = [], set()
        for record in records:
            rowid, fp, body, meta, question, language, tokens = record
            stats['considered'] += 1
            if fp in seen or fp in batch_seen:
                stats['duplicate'] += 1
                continue
            if question is not None and groups[(question, language)] >= group_cap:
                stats['question_cap'] += 1
                continue
            batch_seen.add(fp)
            eligible.append(record)
        unknown = [e for e in eligible if e[6] is None]
        lengths = checker.count_batch([orjson.loads(e[2]) for e in unknown]) if unknown else []
        measured = {e[0]: n for e, n in zip(unknown, lengths)}
        db.executemany('UPDATE candidates SET tokens=? WHERE rowid=?', [(n, k) for k, n in measured.items()])
        for rowid, fp, body, meta, question, language, tokens in eligible:
            tokens = measured.get(rowid, tokens)
            if tokens > max_tokens:
                stats['over_limit'] += 1
                continue
            if question is not None and groups[(question, language)] >= group_cap:
                stats['question_cap'] += 1
                continue
            size = len(body.encode('utf-8')) + 1
            if selected_bytes + size > target_bytes:
                # Choose whichever whole-row boundary is nearer to the byte target.
                if abs(selected_bytes + size - target_bytes) >= abs(selected_bytes - target_bytes):
                    done = True
                    break
                done = True
            db.execute('INSERT INTO selected VALUES (?,?,?,?)', (rowid, rng.getrandbits(63), tokens, size))
            selected_bytes += size
            stats['selected'] += 1
            stats['tokens'] += tokens
            stats['reasoning_selected'] += int(bool(has_reasoning(orjson.loads(body))))
            stats['max_tokens'] = max(stats['max_tokens'], tokens)
            seen.add(fp)
            if question is not None:
                groups[(question, language)] += 1
            if done:
                break
        db.commit()
        if time.monotonic() - last_log > 20:
            log(f'{source}: selected {stats["selected"]:,} rows, {selected_bytes / 1e6:.1f}/{target_bytes / 1e6:.1f} MB')
            last_log = time.monotonic()
    stats['bytes'] = selected_bytes
    stats['target_bytes'] = target_bytes
    log(f'{source} complete: {stats["selected"]:,} rows, {selected_bytes:,} bytes')
    return dict(stats)


def classify_replay(db):
    """Split replay candidates by actual nonempty reasoning, preserving priorities."""
    log('Classifying T2T into nonempty-reasoning and ordinary conversations')
    counts = Counter()
    last_rowid = 0
    last_log = time.monotonic()
    while True:
        batch = db.execute("SELECT rowid,body FROM candidates NOT INDEXED WHERE rowid>? AND source='t2t_replay' ORDER BY rowid LIMIT 2048", (last_rowid,)).fetchall()
        if not batch:
            break
        last_rowid = batch[-1][0]
        updates = []
        for rowid, body in batch:
            source = 't2t_cot' if has_reasoning(orjson.loads(body)) else 't2t_final'
            counts[source] += 1
            updates.append((source, rowid))
        db.executemany('UPDATE candidates SET source=? WHERE rowid=?', updates)
        db.commit()
        if time.monotonic() - last_log > 20:
            log(f'T2T classified {sum(counts.values()):,} additional rows')
            last_log = time.monotonic()
    log(f'T2T classification complete: {dict(counts)}')


def select_balanced_replay(db, target_bytes, base_rows, base_reasoning, checker,
                           max_tokens, seen, rng):
    """Build a reasoning surplus, then add equal reasoning/plain replay pairs."""
    stats = {'t2t_cot': Counter(), 't2t_final': Counter()}

    def candidates(source):
        cursor = db.execute('SELECT rowid,fp,body,tokens FROM candidates WHERE source=? ORDER BY priority', (source,))
        while True:
            batch = cursor.fetchmany(128)
            if not batch:
                return
            unknown = [e for e in batch if e[3] is None and e[1] not in seen]
            lengths = checker.count_batch([orjson.loads(e[2]) for e in unknown]) if unknown else []
            measured = {e[0]: n for e, n in zip(unknown, lengths)}
            db.executemany('UPDATE candidates SET tokens=? WHERE rowid=?', [(n, k) for k, n in measured.items()])
            db.commit()
            for rowid, fp, body, tokens in batch:
                stats[source]['considered'] += 1
                if fp in seen:
                    stats[source]['duplicate'] += 1
                    continue
                tokens = measured.get(rowid, tokens)
                if tokens > max_tokens:
                    stats[source]['over_limit'] += 1
                    continue
                yield rowid, fp, body, tokens, len(body.encode('utf-8')) + 1

    streams = {source: candidates(source) for source in stats}
    used_bytes = 0
    def add(source, entry):
        nonlocal used_bytes
        rowid, fp, body, tokens, size = entry
        db.execute('INSERT INTO selected VALUES (?,?,?,?)', (rowid, rng.getrandbits(63), tokens, size))
        seen.add(fp)
        used_bytes += size
        stats[source]['selected'] += 1
        stats[source]['tokens'] += tokens
        stats[source]['bytes'] += size
        stats[source]['max_tokens'] = max(stats[source]['max_tokens'], tokens)
        stats[source]['reasoning_selected'] += int(source == 't2t_cot')

    deficit = max(1, math.ceil((11 * base_rows - 20 * base_reasoning) / 9))
    balancing_source = 't2t_cot'
    log(f'Building reasoning surplus: first select {deficit:,} additional reasoning rows')
    last_log = time.monotonic()
    try:
        for _ in range(abs(deficit)):
            add(balancing_source, next(streams[balancing_source]))
            if time.monotonic() - last_log > 20:
                db.commit()
                log(f'Reasoning balance: selected {stats[balancing_source]["selected"]:,}/{abs(deficit):,}; {used_bytes / 1e6:.1f} MB')
                last_log = time.monotonic()
            if used_bytes > target_bytes:
                raise RuntimeError('Reasoning surplus requires more replay bytes than the budget; adjust competitive reasoning byte share')
        while True:
            cot = next(streams['t2t_cot'])
            plain = next(streams['t2t_final'])
            while plain[1] == cot[1]:
                plain = next(streams['t2t_final'])
            pair_bytes = cot[4] + plain[4]
            if used_bytes + pair_bytes > target_bytes:
                if abs(used_bytes + pair_bytes - target_bytes) < abs(used_bytes - target_bytes):
                    add('t2t_cot', cot)
                    add('t2t_final', plain)
                break
            add('t2t_cot', cot)
            add('t2t_final', plain)
            if time.monotonic() - last_log > 20:
                db.commit()
                log(f'Paired replay: {used_bytes / 1e6:.1f}/{target_bytes / 1e6:.1f} MB')
                last_log = time.monotonic()
    except StopIteration as error:
        raise RuntimeError('Not enough unique replay candidates to exceed 50% reasoning at requested size') from error
    db.commit()
    final_rows = base_rows + sum(s['selected'] for s in stats.values())
    final_reasoning = base_reasoning + stats['t2t_cot']['selected']
    assert 2 * final_reasoning > final_rows
    log(f'Replay complete: {used_bytes:,} bytes; final reasoning {final_reasoning:,}/{final_rows:,} = {final_reasoning/final_rows:.2%}')
    return {k: dict(v) for k, v in stats.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-gib', type=float, default=1.6)
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--per-question-language', type=int, default=3)
    p.add_argument('--cache-only', action='store_true')
    p.add_argument('--in-memory', action='store_true', help='Copy candidate DB into RAM for faster selection; needs about 16 GB free RAM')
    p.add_argument('--cache', type=Path, default=ROOT / '.cache/continued_sft_large/candidates.sqlite')
    p.add_argument('--output', type=Path, default=ROOT / 'dataset/sft_competitive_t2t_ultrachat_1p6g_16k_cot50plus.jsonl')
    args = p.parse_args()
    if args.target_gib <= 0 or args.max_tokens <= 0 or args.per_question_language <= 0:
        p.error('Size, token limit and question cap must be positive')
    output = args.output.resolve()
    report_path, source_path = output.with_suffix('.report.json'), output.with_suffix('.sources.jsonl')
    if any(x.exists() for x in (output, report_path, source_path)):
        raise FileExistsError('Output or sidecar exists; choose a new --output')
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.cache)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-131072')
    manifest = input_manifest()
    exists = db.execute("SELECT count(*) FROM sqlite_master WHERE name='settings'").fetchone()[0]
    if exists:
        settings = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM settings')}
        if not settings.get('complete'):
            raise RuntimeError('Candidate cache is incomplete; use a new --cache path')
        if settings['manifest'] != manifest or settings['seed'] != args.seed or settings['max_tokens'] != args.max_tokens:
            raise RuntimeError('Cache inputs/settings changed; use a new --cache path')
        log('Reusing validated candidate cache')
    else:
        build_cache(db, manifest, args.seed, args.max_tokens)
        settings = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM settings')}
    if args.cache_only:
        db.close()
        return
    if args.in_memory:
        log('Loading candidate cache into RAM for selection')
        memory_db = sqlite3.connect(':memory:')
        last_progress = [time.monotonic()]
        def progress(status, remaining, total):
            if time.monotonic() - last_progress[0] > 20:
                log(f'Candidate cache loading: {(total - remaining) / total:.0%}')
                last_progress[0] = time.monotonic()
        db.backup(memory_db, pages=4096, progress=progress)
        db.close()
        db = memory_db
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA temp_store=MEMORY')
        log('Candidate cache loaded into RAM')
    classify_replay(db)
    db.executescript('DROP TABLE IF EXISTS selected; CREATE TABLE selected (candidate_id INTEGER PRIMARY KEY, shuffle INTEGER, tokens INTEGER, bytes INTEGER);')
    checker = LengthChecker(ROOT / 'model')
    target = round(args.target_gib * 1024**3)
    budgets = {'competitive_cot': round(target * .35), 'competitive_final': round(target * .15),
               'ultrachat_new': round(target * .20)}
    seen, groups, rng, stats = set(), Counter(), random.Random(args.seed + 100), {}
    for source, budget in budgets.items():
        stats[source] = select_source(db, source, budget, checker, args.max_tokens,
                                      seen, groups, args.per_question_language, rng)
        if abs(stats[source]['bytes'] - budget) > max(100000, budget * .01):
            raise RuntimeError(f'Insufficient {source} candidates: {stats[source]}; candidate cache retained for adjustment')
    replay_budget = target - sum(s['bytes'] for s in stats.values())
    stats.update(select_balanced_replay(db, replay_budget,
        sum(s['selected'] for s in stats.values()), sum(s['reasoning_selected'] for s in stats.values()),
        checker, args.max_tokens, seen, rng))
    log('Writing globally shuffled output and aligned provenance')
    db.execute('CREATE INDEX IF NOT EXISTS selected_shuffle ON selected(shuffle)')
    db.commit()
    h = hashlib.sha256()
    counts, lengths, reasoning_counts, language_counts = Counter(), defaultdict(list), Counter(), Counter()
    temporary = output.with_suffix('.jsonl.tmp')
    temporary_sources = source_path.with_suffix('.jsonl.tmp')
    n = 0
    with temporary.open('wb') as f, temporary_sources.open('wb') as meta_file:
        for source, fp, body, meta, tokens, size in db.execute('''
            SELECT c.source,c.fp,c.body,c.meta,s.tokens,s.bytes FROM selected s
            JOIN candidates c ON c.rowid=s.candidate_id ORDER BY s.shuffle'''):
            encoded = (body + '\n').encode('utf-8')
            assert len(encoded) == size
            f.write(encoded)
            h.update(encoded)
            row, provenance = orjson.loads(body), orjson.loads(meta)
            thought = bool(has_reasoning(row))
            provenance.update(source=source, tokens=tokens, bytes=size,
                              fingerprint=fp.hex(), has_reasoning=thought)
            meta_file.write((dumps(provenance) + '\n').encode('utf-8'))
            counts[source] += 1
            lengths[source].append(tokens)
            reasoning_counts[source] += int(thought)
            if source.startswith('competitive'):
                language_counts[provenance['language']] += 1
            n += 1
    token_stats = {}
    total_tokens = sum(map(sum, lengths.values()))
    for source, values in lengths.items():
        values.sort()
        token_stats[source] = {'total': sum(values), 'fraction': sum(values) / total_tokens,
                               'mean': sum(values) / len(values), 'p50': values[len(values)//2],
                               'p95': values[int(len(values)*.95)], 'max': max(values)}
    assert sum(reasoning_counts.values()) * 2 > n
    report = {'output': str(output), 'target_bytes': target, 'bytes': temporary.stat().st_size,
              'gib': temporary.stat().st_size / 1024**3, 'total_rows': n,
              'sha256': h.hexdigest(), 'seed': args.seed, 'max_tokens': args.max_tokens,
              'counts': counts, 'total_tokens': total_tokens, 'token_stats': token_stats,
              'nonempty_reasoning_rows': reasoning_counts, 'competitive_languages': language_counts,
              'reasoning_fraction': sum(reasoning_counts.values()) / n,
              'selection_stats': stats, 'candidate_stats': settings['stats'],
              'max_answers_per_question_language': max(groups.values()),
              'policy': {'ratio_basis': 'UTF-8 JSONL bytes: competitive 50%, T2T 30%, UltraChat 20%',
                         'competitive_reasoning': '70% of competitive bytes retain complete reasoning_content; final answers only in remaining 30%',
                         'reasoning_fraction': 'strictly more than 50% of conversation rows contain nonempty reasoning; empty think tags excluded; build a surplus then sample T2T in balanced pairs',
                         'sampling': 'fixed-seed random priority, without exact duplicate conversations; max 3 answers per question/language',
                         'length': 'complete local chat template, no truncation; CoT candidate character prefilter <= 4 * max_tokens',
                         'previous_data': 'T2T intentional replay; competitive/UltraChat exclude conversation and first-user-prompt overlap with both trained datasets',
                         'splits': 'UltraChat train_sft only; competitive excludes all question IDs with any non-train label across all four local shards',
                         'validation_limits': 'source split labels are not comprehensive benchmark decontamination; generated code has not been executed'},
              'candidate_cache': str(args.cache.resolve()), 'inputs': manifest,
              'tokenizer_sha256': hashlib.sha256((ROOT/'model/tokenizer.json').read_bytes()).hexdigest(),
              'template_sha256': hashlib.sha256((ROOT/'model/tokenizer_config.json').read_bytes()).hexdigest()}
    temporary.replace(output)
    temporary_sources.replace(source_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    db.close()
    log(json.dumps({k: report[k] for k in ['output', 'bytes', 'gib', 'total_rows', 'counts', 'nonempty_reasoning_rows', 'total_tokens']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
