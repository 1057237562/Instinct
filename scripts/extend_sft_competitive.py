"""Add complete competitive reasoning to an existing mix using bounded buffers."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

os.environ['RAYON_NUM_THREADS'] = '2'
import orjson
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mix_continued_sft import LengthChecker, valid
from scripts.mix_large_continued_sft import fingerprints, has_reasoning, log
from scripts.mix_sft_datasets import ExternalRandomShuffler


def main():
    base = ROOT / 'dataset/sft_competitive_t2t_ultrachat_16k_cot50plus.jsonl'
    output = ROOT / 'dataset/sft_competitive_t2t_ultrachat_16k_coding_heavy.jsonl'
    report_path, sources_path = output.with_suffix('.report.json'), output.with_suffix('.sources.jsonl')
    payload = ROOT / 'dataset/.extended_competitive_payload.jsonl'
    if any(p.exists() for p in (output, report_path, sources_path, payload)):
        raise FileExistsError('Output or intermediate already exists')
    parent = orjson.loads(base.with_suffix('.report.json').read_bytes())
    limit, seed = 16384, 20260906
    shuffler = ExternalRandomShuffler(payload, seed, 128)
    seen, groups, stats = set(), Counter(), defaultdict(Counter)
    checker = LengthChecker(ROOT / 'model')

    def emit(row, meta):
        fp, _ = fingerprints(row)
        assert valid(row) and fp not in seen and fp.hex() == meta['fingerprint']
        assert 0 < meta['tokens'] <= limit
        seen.add(fp)
        assert shuffler.add({'row': row, 'meta': meta})
        source = meta['source']
        stats[source]['selected'] += 1
        stats[source]['bytes'] += meta['bytes']
        stats[source]['tokens'] += meta['tokens']
        stats[source]['reasoning_selected'] += int(meta['has_reasoning'])
        if source == 'competitive_cot':
            key = (meta['question_id'], meta['language'])
            groups[key] += 1
            assert groups[key] <= 3

    log('Preserving the completed base mixture')
    h = hashlib.sha256()
    with base.open('rb') as rows, base.with_suffix('.sources.jsonl').open('rb') as provenance:
        for index, line in enumerate(rows, 1):
            h.update(line)
            meta_line = provenance.readline()
            assert meta_line
            emit(orjson.loads(line), orjson.loads(meta_line))
            if index % 100000 == 0:
                log(f'Base rows preserved: {index:,}; RAM {psutil.Process().memory_info().rss/1024**2:.0f} MiB')
        assert not provenance.readline()
    assert h.hexdigest() == parent['sha256']
    base_code_bytes = stats['competitive_cot']['bytes']
    db = sqlite3.connect(f"{Path(parent['candidate_cache']).as_uri()}?mode=ro", uri=True)
    db.execute('PRAGMA cache_size=-8192')
    db.execute('PRAGMA mmap_size=0')
    cursor = db.execute('''SELECT c.fp,c.body,c.meta,c.question,c.language
        FROM candidates c WHERE source='competitive_cot'
        AND NOT EXISTS (SELECT 1 FROM heldout h WHERE h.question=c.question)
        ORDER BY priority''')
    log('Adding further unique competitive reasoning up to about 1.3 GB total competitive data')
    last_log = time.monotonic()
    while stats['competitive_cot']['bytes'] < 1_300_000_000:
        batch = cursor.fetchmany(16)
        if not batch:
            log('Complete competitive pool exhausted; using all available additional samples')
            break
        candidates = []
        for fp, body, meta, question, language in batch:
            if fp not in seen and groups[(question, language)] < 3:
                candidates.append((fp, orjson.loads(body), orjson.loads(meta), question, language))
        lengths = checker.count_batch([e[1] for e in candidates])
        for (fp, row, meta, question, language), tokens in zip(candidates, lengths):
            if fp in seen or groups[(question, language)] >= 3 or tokens > limit:
                continue
            assert has_reasoning(row)
            size = len(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) + 1
            meta.update(source='competitive_cot', tokens=tokens, bytes=size,
                        fingerprint=fp.hex(), has_reasoning=True)
            emit(row, meta)
            if stats['competitive_cot']['bytes'] >= 1_300_000_000:
                break
        if time.monotonic() - last_log > 20:
            log(f'Competitive total {stats["competitive_cot"]["bytes"]/1e9:.3f} GB, {stats["competitive_cot"]["selected"]:,} rows; RAM {psutil.Process().memory_info().rss/1024**2:.0f} MiB')
            last_log = time.monotonic()
    db.close()
    assert stats['competitive_cot']['bytes'] > base_code_bytes
    log('Globally shuffling the expanded mixture')
    shuffler.finish()
    counts, thoughts, languages, lengths = Counter(), Counter(), Counter(), defaultdict(list)
    temporary, temporary_sources = output.with_suffix('.jsonl.tmp'), sources_path.with_suffix('.jsonl.tmp')
    h = hashlib.sha256()
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
    assert 1_600_000_000 <= total_bytes <= 5_000_000_000
    assert sum(thoughts.values()) / total_rows > .5
    total_tokens = sum(sum(v) for v in lengths.values())
    token_stats = {}
    for source, values in lengths.items():
        values.sort()
        token_stats[source] = {'total': sum(values), 'fraction': sum(values)/total_tokens,
                              'mean': sum(values)/len(values), 'p50': values[len(values)//2],
                              'p95': values[int(len(values)*.95)], 'max': max(values)}
    report = dict(parent)
    report.pop('verification', None)
    report.update(output=str(output), bytes=total_bytes, gb=total_bytes/1e9, gib=total_bytes/1024**3,
                  total_rows=total_rows, sha256=h.hexdigest(), seed=seed, counts=dict(counts),
                  nonempty_reasoning_rows=dict(thoughts), reasoning_fraction=sum(thoughts.values())/total_rows,
                  total_tokens=total_tokens, token_stats=token_stats, selection_stats=dict(stats),
                  competitive_languages=dict(languages), max_answers_per_question_language=max(groups.values()),
                  peak_process_memory_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
                  parent_dataset={'path': str(base), 'sha256': parent['sha256']},
                  additional_competitive_bytes=stats['competitive_cot']['bytes'] - base_code_bytes)
    report['policy'] = dict(parent['policy'], ratio_basis='preserve base replay/UltraChat; expand unique competitive reasoning from about 600 MB to about 1.3 GB')
    temporary.replace(output)
    temporary_sources.replace(sources_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    payload.unlink()
    log(json.dumps({k: report[k] for k in ['output','gb','gib','total_rows','counts','reasoning_fraction','peak_process_memory_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
