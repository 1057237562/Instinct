"""Build a 1.6-5 GB SFT dataset with bounded buffers and >50% reasoning.

Select about 600 MB competitive reasoning, an 80% hash sample of unique
T2T reasoning, then unseen UltraChat to about 1.8 GB. No truncation.
"""
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

os.environ['RAYON_NUM_THREADS'] = '2'
import orjson
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mix_large_continued_sft import fingerprints, has_reasoning, log, input_manifest
from scripts.mix_continued_sft import LengthChecker, valid
from scripts.mix_sft_datasets import ExternalRandomShuffler


def main():
    seed, limit = 20260905, 16384
    output = ROOT / 'dataset/sft_competitive_t2t_ultrachat_16k_cot50plus.jsonl'
    report_path, sources_path = output.with_suffix('.report.json'), output.with_suffix('.sources.jsonl')
    payload = ROOT / 'dataset/.continued_low_memory_payload.jsonl'
    if any(x.exists() for x in (output, report_path, sources_path, payload)):
        raise FileExistsError('Output or intermediate payload already exists')
    cache = ROOT / '.cache/continued_sft_large/candidates.sqlite'
    db = sqlite3.connect(f'{cache.as_uri()}?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-8192')
    db.execute('PRAGMA mmap_size=0')
    settings = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM settings')}
    assert settings['complete'] and settings['manifest'] == input_manifest()
    checker = LengthChecker(ROOT / 'model')
    shuffler = ExternalRandomShuffler(payload, seed + 200, 128)
    seen, groups, stats = set(), Counter(), defaultdict(Counter)

    def emit(source, row, meta, tokens):
        fp, _ = fingerprints(row)
        assert fp not in seen and 0 < tokens <= limit
        assert valid(row)
        seen.add(fp)
        thought = bool(has_reasoning(row))
        body = json.dumps(row, ensure_ascii=False, separators=(',', ':'))
        size = len(body.encode('utf-8')) + 1
        meta = dict(meta, source=source, tokens=tokens, bytes=size,
                    fingerprint=fp.hex(), has_reasoning=thought)
        assert shuffler.add({'row': row, 'meta': meta})
        stats[source]['selected'] += 1
        stats[source]['bytes'] += size
        stats[source]['tokens'] += tokens
        stats[source]['reasoning_selected'] += int(thought)
        if source == 'competitive_cot':
            groups[(meta['question_id'], meta['language'])] += 1

    log('Selecting complete competitive reasoning from read-only cache')
    cursor = db.execute('''SELECT c.fp,c.body,c.meta,c.question,c.language FROM candidates c
        WHERE source='competitive_cot'
        AND NOT EXISTS (SELECT 1 FROM heldout h WHERE h.question=c.question)
        ORDER BY priority''')
    last_log = time.monotonic()
    finished = False
    while not finished:
        batch = cursor.fetchmany(16)
        if not batch:
            finished = True
            break
        candidates = []
        for fp, body, meta, question, language in batch:
            if fp not in seen and groups[(question, language)] < 3:
                candidates.append((fp, orjson.loads(body), orjson.loads(meta), question, language))
        lengths = checker.count_batch([e[1] for e in candidates])
        for (fp, row, meta, question, language), tokens in zip(candidates, lengths):
            if fp in seen or groups[(question, language)] >= 3:
                continue
            if tokens > limit:
                stats['competitive_cot']['over_limit'] += 1
                continue
            assert has_reasoning(row)
            emit('competitive_cot', row, meta, tokens)
            if stats['competitive_cot']['bytes'] >= 600_000_000:
                finished = True
                break
        if time.monotonic() - last_log > 20:
            log(f'Competitive CoT: {stats["competitive_cot"]["selected"]:,} rows, {stats["competitive_cot"]["bytes"]/1e9:.3f} GB; RAM {psutil.Process().memory_info().rss/1024**2:.0f} MiB')
            last_log = time.monotonic()
    code_count = stats['competitive_cot']['selected']
    log(f'Competitive ready: {code_count:,}; selecting reasoning replay and filling with UltraChat')

    log('Randomly sampling T2T conversations with nonempty reasoning')
    path = ROOT / 'dataset/sft_t2t_mini.jsonl'
    batch = []
    sample_salt = str(seed + 201).encode()
    def emit_replay_batch():
        for (row, line_number), tokens in zip(batch, checker.count_batch([e[0] for e in batch])):
            fp, _ = fingerprints(row)
            if tokens <= limit and fp not in seen:
                emit('t2t_cot', row, {'file': str(path), 'row': line_number}, tokens)
        batch.clear()
    with path.open('rb') as f:
        for i, line in enumerate(f, 1):
            if i % 100000 == 0:
                log(f'T2T scanned {i:,}; selected {stats["t2t_cot"]["selected"]:,}; RAM {psutil.Process().memory_info().rss/1024**2:.0f} MiB')
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not valid(row) or not has_reasoning(row):
                continue
            fp, _ = fingerprints(row)
            if fp in seen:
                continue
            if int.from_bytes(hashlib.sha256(sample_salt + fp).digest()[:8], 'big') >= int(.8 * 2**64):
                continue
            batch.append((row, i))
            if len(batch) == 16:
                emit_replay_batch()
    if batch:
        emit_replay_batch()
    log(f'T2T reasoning replay ready: {stats["t2t_cot"]["selected"]:,}')

    cursor = db.execute("SELECT body,meta FROM candidates WHERE source='ultrachat_new' ORDER BY priority")
    target_bytes = 1_800_000_000
    while sum(s['bytes'] for s in stats.values()) < target_bytes:
        batch = cursor.fetchmany(16)
        if not batch:
            log('Unseen UltraChat pool exhausted; checking actual final size')
            break
        rows = [orjson.loads(e[0]) for e in batch]
        for row, (_, meta), tokens in zip(rows, batch, checker.count_batch(rows)):
            fp, _ = fingerprints(row)
            if fp not in seen and tokens <= limit:
                emit('ultrachat_new', row, orjson.loads(meta), tokens)
                if sum(s['bytes'] for s in stats.values()) >= target_bytes:
                    break
        if time.monotonic() - last_log > 20:
            log(f'UltraChat selected {stats["ultrachat_new"]["selected"]:,}; total {sum(s["bytes"] for s in stats.values())/1e9:.3f} GB; RAM {psutil.Process().memory_info().rss/1024**2:.0f} MiB')
            last_log = time.monotonic()
    db.close()
    log('Globally shuffling selected conversations')
    shuffler.finish()
    counts, thoughts, lengths, languages = Counter(), Counter(), defaultdict(list), Counter()
    h = hashlib.sha256()
    temporary, temporary_sources = output.with_suffix('.jsonl.tmp'), sources_path.with_suffix('.jsonl.tmp')
    log('Writing final training JSONL and aligned provenance')
    with payload.open('rb') as f, temporary.open('wb') as out, temporary_sources.open('wb') as provenance:
        for line in f:
            item = orjson.loads(line)
            row, meta = item['row'], item['meta']
            body = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
            assert len(body) == meta['bytes']
            out.write(body)
            h.update(body)
            provenance.write((json.dumps(meta, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))
            source = meta['source']
            counts[source] += 1
            thoughts[source] += int(meta['has_reasoning'])
            lengths[source].append(meta['tokens'])
            if source == 'competitive_cot':
                languages[meta['language']] += 1
    total_bytes, total_rows = temporary.stat().st_size, sum(counts.values())
    total_tokens = sum(sum(v) for v in lengths.values())
    assert 1_600_000_000 <= total_bytes <= 5_000_000_000
    assert sum(thoughts.values()) / total_rows > .5
    token_stats = {}
    for source, values in lengths.items():
        values.sort()
        token_stats[source] = {'total': sum(values), 'fraction': sum(values)/total_tokens,
                              'mean': sum(values)/len(values), 'p50': values[len(values)//2],
                              'p95': values[int(len(values)*.95)], 'max': max(values)}
    report = {'output': str(output), 'bytes': total_bytes, 'gb': total_bytes/1e9, 'gib': total_bytes/1024**3,
              'min_bytes': 1_600_000_000, 'max_bytes': 5_000_000_000,
              'total_rows': total_rows, 'sha256': h.hexdigest(), 'seed': seed, 'max_tokens': limit,
              'counts': counts, 'nonempty_reasoning_rows': thoughts,
              'reasoning_fraction': sum(thoughts.values())/total_rows,
              'total_tokens': total_tokens, 'token_stats': token_stats,
              'selection_stats': stats, 'competitive_languages': languages,
              'max_answers_per_question_language': max(groups.values()),
              'peak_process_memory_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
              'policy': {'ratio_basis': 'adaptive byte quotas: about 600 MB competitive reasoning, 80% hash sample of unique T2T reasoning, then unseen UltraChat to about 1.8 GB',
                         'reasoning': 'complete nonempty reasoning in all competitive and T2T samples; empty think tags do not count; UltraChat kept as originally provided',
                         'sampling': 'fixed-seed random priority without exact duplicates; at most 3 answers per competitive question/language',
                         'length': 'full local chat template <= 16384 tokens; no truncation; original competitive candidate character prefilter <= 65536',
                         'previous_data': 'T2T is intentional replay; competitive and UltraChat exclude conversation and first-user-prompt overlap with both previously trained files',
                         'splits': 'UltraChat train_sft only; competitive question IDs with any non-train label in all four local shards excluded',
                         'validation_limits': 'code not executed; source split filtering is not comprehensive benchmark decontamination'},
              'inputs': settings['manifest'], 'candidate_cache': str(cache),
              'tokenizer_sha256': hashlib.sha256((ROOT/'model/tokenizer.json').read_bytes()).hexdigest(),
              'template_sha256': hashlib.sha256((ROOT/'model/tokenizer_config.json').read_bytes()).hexdigest()}
    temporary.replace(output)
    temporary_sources.replace(sources_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    payload.unlink()
    log(json.dumps({k: report[k] for k in ['output','gb','gib','total_rows','counts','reasoning_fraction','total_tokens']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
