"""Mix verified MiniCPM5 Code samples into InstinctV2 at a 16K context."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import os
from pathlib import Path
import random
import shutil
import time

import orjson

import build_instinctv2_sft as base
from augment_instinctv2_sft import source_pairs, take
from expand_instinctv2_sft import benchmark_keys, keys, pick, write_json

BASELINE = base.DATA / 'instinctv2_sft_v3'
REPO = 'openbmb/UltraData-SFT-2605'
REVISION = 'affda6aca75e7cff78e73f93ad08d4c3b01f097c'
CONTEXT = 16384
RENDER_CAP = CONTEXT - 64
OFFICIAL_TOKENS = 15_000_000
THINK_QUOTA = 5_000_000


def init_worker():
    base.CAP = RENDER_CAP
    base.init_worker()


def screen_batch(items):
    start = time.process_time()
    prepared, stats = [], Counter()
    for split, row, meta in items:
        stats[split, 'source_rows'] += 1
        turns, uid, family = row.get('messages'), row.get('uid'), row.get('source')
        if (row.get('domain') != 'Code' or row.get('think_type') != split or not uid or not family
                or not isinstance(turns, list) or any(not isinstance(t, dict) for t in turns)):
            stats[split, 'source_schema_filtered'] += 1
            continue
        assistant = [t for t in turns if t.get('role') == 'assistant']
        reasoning = any(isinstance(trace, str) and trace.strip()
                        for t in assistant if (trace := t.get('reasoning_content')))
        if not assistant or reasoning != (split == 'think'):
            stats[split, 'think_mode_filtered'] += 1
            continue
        category = f'minicpm5_code_{split}'
        prepared.append(('train', category, {'conversations': turns},
                         {**meta, 'category': category, 'group': uid, 'family': family}))
    accepted, filters, terms, pid, _ = base.screen_batch(prepared)
    for (_, category), counts in filters.items():
        split = category.removeprefix('minicpm5_code_')
        for name, count in counts.items():
            stats[split, name] += count

    screened = []
    for _, category, row, meta in accepted:
        code = '\n'.join(base.CODE_BLOCKS(row['conversations'][-1]['content']))
        if code:
            try:
                tree = ast.parse(code)
                key = base.CODE_KEY(code)
                names = {node.name for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
                if key in base.CODES or names & base.FUNCTIONS:
                    split = category.removeprefix('minicpm5_code_')
                    stats[split, 'benchmark_python_filtered'] += 1
                    continue
                meta['python_ast_sha256'] = key
            except (SyntaxError, ValueError, RecursionError):
                # A fence can contain pseudocode, a partial snippet, or a non-Python language.
                pass
        screened.append((row, meta))
    return screened, stats, terms, pid, time.process_time() - start


def source_rows(source, manifest):
    def verify(item):
        path = source / Path(item['path']).name
        if path.stat().st_size != item['size'] or base.digest(path) != item['sha256']:
            raise ValueError(f'Unverified source shard: {path.name}')
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(verify, manifest['files']))
    for item, path in zip(manifest['files'], paths, strict=True):
        split = item['path'].split('/')[1]
        if split not in {'think', 'no_think'}:
            raise ValueError(f'Unexpected source partition: {split}')
        for number, row in enumerate(base.rows(path), 1):
            yield split, row, {'path': item['path'], 'source_row': number,
                               'repository': REPO, 'revision': REVISION}


def screen(source, work, workers):
    manifest = orjson.loads((source / 'source.manifest.json').read_bytes())
    if (manifest['repository'], manifest['revision']) != (REPO, REVISION):
        raise ValueError('Incorrect pinned dataset')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    counts, terms, cpu = Counter(), Counter(), Counter()
    work.mkdir(parents=True, exist_ok=False)
    spool = work / 'official.candidates.jsonl'
    start = time.perf_counter()
    with spool.open('xb') as output, ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        for index, window in enumerate(base.batches(base.batches(source_rows(source, manifest), 64), 2 * workers), 1):
            for rows, stats, hits, pid, elapsed in pool.map(screen_batch, window):
                counts.update(stats)
                terms.update(hits)
                cpu[str(pid)] += elapsed
                for row, meta in rows:
                    base.emit(output, {'row': row, 'meta': meta})
            if index % 5 == 0:
                print('screened', {split: counts[split, 'source_rows'] for split in ('think', 'no_think')}, flush=True)
    write_json(work / 'screen.report.json', {
        'source': manifest, 'render_cap': RENDER_CAP,
        'counts': {split: {key: val for (mode, key), val in counts.items() if mode == split}
                   for split in ('think', 'no_think')},
        'candidate_sha256': base.digest(spool), 'identity_keyword_hits': dict(terms),
        'worker_cpu_seconds': dict(cpu), 'wall_seconds': time.perf_counter() - start,
    })


def length_band(tokens):
    for bound in (2048, 4096, 8192, CONTEXT):
        if tokens <= bound:
            return str(bound)
    raise ValueError('input exceeds the training window')


def choose(candidates, seen, code, answers, benchmarks, rng):
    exact, grams, _ = benchmarks
    groups, unique = set(), {'think': [], 'no_think': []}
    stats, selected = Counter(), []
    for item in base.rows(candidates):
        row, meta = item['row'], item['meta']
        if meta['group'] in groups:
            stats['duplicate_uid'] += 1
            continue
        groups.add(meta['group'])
        unique[meta['category'].removeprefix('minicpm5_code_')].append((row, meta))

    # Budget by length so both shorter explanations and 8K–16K traces appear.
    used = 0
    think = unique['think']
    for band, quota in (('16384', 2_000_000), ('8192', 1_500_000), ('4096', 1_000_000), ('2048', 500_000)):
        cohort = [pair for pair in think if length_band(pair[1]['tokens']) == band]
        rows, tokens = pick(cohort, seen, code, answers, exact, grams, quota, stats, rng)
        selected.extend(rows)
        used += tokens
    if used < THINK_QUOTA:
        rows, tokens = pick(think, seen, code, answers, exact, grams, THINK_QUOTA - used, stats, rng)
        selected.extend(rows)
        used += tokens
    rows, tokens = pick(unique['no_think'], seen, code, answers, exact, grams,
                        OFFICIAL_TOKENS - used, stats, rng)
    selected.extend(rows)
    return selected, used + tokens, stats


def assemble(work, out):
    audit = orjson.loads((work / 'screen.report.json').read_bytes())
    candidates = work / 'official.candidates.jsonl'
    if base.digest(candidates) != audit['candidate_sha256']:
        raise ValueError('Candidate hash mismatch')
    prior_path = BASELINE / 'build.report.json'
    prior = orjson.loads(prior_path.read_bytes())
    checked = orjson.loads((BASELINE / 'verification.report.json').read_bytes())
    if checked['status'] != 'verified' or checked['build_report_sha256'] != base.digest(prior_path):
        raise ValueError('Baseline report is unverified')
    for split, meta in prior['splits'].items():
        for file, key in ((f'{split}_identity_clean.jsonl', 'sha256'),
                          (f'{split}.provenance.jsonl.gz', 'provenance_sha256')):
            if base.digest(BASELINE / file) != meta[key]:
                raise ValueError(f'Baseline hash mismatch: {file}')

    train, validation = list(source_pairs(BASELINE, 'train')), list(source_pairs(BASELINE, 'validation'))
    prompts, _, code, answers = keys(train + validation)
    benchmarks = benchmark_keys(base.ROOT / 'tmp')
    rng = random.Random(f'{base.SEED}/instinctv2-official-code')
    official, official_tokens, exclusions = choose(candidates, prompts, code, answers, benchmarks, rng)
    if not official or not any(meta['thinking'] and meta['tokens'] > 8192 for _, meta in official):
        raise ValueError('Official Code mix lacks complete 8K–16K reasoning examples')
    fallback = [(row, meta) for row, meta in train if meta['category'] == 'magicoder_fresh']
    rng.shuffle(fallback)
    fallback_budget = sum(meta['tokens'] for _, meta in fallback) - official_tokens
    kept, kept_tokens = take(fallback, set(), fallback_budget)
    train = [(row, meta) for row, meta in train if meta['category'] != 'magicoder_fresh'] + kept + official
    rng.shuffle(train)

    out.mkdir(parents=True, exist_ok=False)
    train_path = out / 'train_identity_clean.jsonl'
    provenance = out / 'train.provenance.jsonl.gz'
    by_rows, by_tokens, by_band, official_by_band = (Counter() for _ in range(4))
    with train_path.open('xb') as data, provenance.open('xb') as raw, gzip.GzipFile(
            fileobj=raw, mode='wb', mtime=0, compresslevel=4) as meta_out:
        for index, (row, meta) in enumerate(train, 1):
            category, length = meta['category'], meta['tokens']
            by_rows[category] += 1
            by_tokens[category] += length
            by_band[length_band(length)] += 1
            if category.startswith('minicpm5_code_'):
                official_by_band[f'{category}/{length_band(length)}'] += 1
            base.emit(data, row)
            base.emit(meta_out, {**meta, 'row': index})
    for name in ('validation_identity_clean.jsonl', 'validation.provenance.jsonl.gz'):
        shutil.copyfile(BASELINE / name, out / name)
    config_path = base.ROOT / 'trainer/config_instinct_v2.json'
    config = orjson.loads(config_path.read_bytes())
    config['max_position_embeddings'] = CONTEXT
    config_out = out / 'model_16k.config.json'
    write_json(config_out, config)
    total = sum(by_tokens.values())
    write_json(out / 'build.report.json', {
        'identity': 'InstinctV2', 'status': 'jsonl_built',
        'training_max_seq_len': CONTEXT, 'rendered_token_cap': RENDER_CAP,
        'context_model_config_sha256': base.digest(config_out),
        'context_model_config_base_sha256': base.digest(config_path),
        'tokenizer_sha256': base.digest(base.ROOT / 'model/tokenizer.json'),
        'baseline_build_sha256': base.digest(prior_path),
        'baseline_input_tokens': prior['splits']['train']['tokens'],
        'magicoder_replaced_input_tokens': sum(meta['tokens'] for _, meta in fallback) - kept_tokens,
        'official_input_tokens': official_tokens, 'official_screen': audit,
        'official_selection_exclusions': dict(exclusions), 'extra_benchmarks': benchmarks[2],
        'official_length_bands_upper_inclusive': dict(official_by_band),
        'length_bands_upper_inclusive': dict(by_band),
        'limitations': ['The official answers and traces are upstream SFT targets; they were not locally executed.',
                        'The earlier 2048-token throughput does not establish 16K GPU throughput or peak memory.'],
        'splits': {'train': {'rows': len(train), 'tokens': total, 'source_rows': dict(by_rows),
                             'source_tokens': dict(by_tokens), 'sha256': base.digest(train_path),
                             'provenance_sha256': base.digest(provenance),
                             'thinking_rows': sum(meta['thinking'] for _, meta in train),
                             'supervised_tokens': sum(meta['supervised_tokens'] for _, meta in train),
                             'max_tokens': max(meta['tokens'] for _, meta in train)},
                   'validation': prior['splits']['validation']},
    })
    print(f'rows={len(train):,} input_tokens={total:,} official={official_tokens:,} '
          f'longest={max(meta["tokens"] for _, meta in train):,}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('screen')
    prepare.add_argument('source', type=Path)
    prepare.add_argument('work', type=Path)
    prepare.add_argument('--workers', type=int, default=12)
    release = sub.add_parser('assemble')
    release.add_argument('work', type=Path)
    release.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'screen':
        screen(args.source, args.work, args.workers)
    else:
        assemble(args.work, args.out)


if __name__ == '__main__':
    main()
