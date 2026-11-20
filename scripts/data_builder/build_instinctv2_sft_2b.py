"""Screen candidate shards and assemble the 2B InstinctV2 SFT release."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import orjson

import build_instinctv2_sft as base
from add_instinctv2_official_code import CONTEXT, RENDER_CAP, REPO, REVISION, length_band
from augment_instinctv2_sft import source_pairs
from expand_instinctv2_sft import answer_key, benchmark_keys, benchmark_overlap, write_json

BASELINE = base.DATA / 'instinctv2_sft_v4'
TARGET = 2_000_000_000
T2T = base.DATA / 'sft_t2t_mini.jsonl'
T2T_SHA256 = '10fae2358aeed89a4828e9270c36a7d764adf20dcfb428870cc3612831f1dd6c'
DOMAINS = {'Code': 'code', 'Math': 'math', 'Chinese-general': 'chinese',
           'Knowledge': 'knowledge', 'IF': 'instruction'}
CHARACTER_CEILING = 131072
QUOTAS = {'t2t': 450_000_000,
          'code_think': 130_000_000, 'code_no_think': 390_000_000,
          'math_think': 180_000_000, 'math_no_think': 270_000_000,
          'chinese_think': 55_000_000, 'chinese_no_think': 150_000_000,
          'knowledge_think': 55_000_000, 'knowledge_no_think': 180_000_000,
          'instruction_think': 45_000_000, 'instruction_no_think': 100_000_000}


def init_worker():
    base.CAP = RENDER_CAP
    base.init_worker()
    global BENCHMARKS
    BENCHMARKS = benchmark_keys(base.ROOT / 'tmp')


def emit_candidate(output, row, meta, baseline=False):
    # Two compact JSON values follow the native merge key columns.
    prompt = meta['first_user_sha256'][:32]
    group = base.fingerprint(f"{meta.get('repository', '')}\n{meta['group']}")[:32]
    answer = '' if baseline else (answer_key(row) or '')[:32]
    flags = int(bool(meta['thinking'])) + 2 * int(bool(meta['tool_calls']))
    prefix = '\t'.join((prompt, group, answer, str(meta['tokens']), str(meta['supervised_tokens']),
                        meta['category'], str(flags))).encode('ascii')
    output.write(prefix + b'\t' + orjson.dumps(row) + b'\t' + orjson.dumps(meta) + b'\n')


def raw_rows(path, first, last, t2t):
    with path.open('rb') as stream:
        if first:
            stream.seek(first - 1)
            if stream.read(1) != b'\n':
                stream.readline()
        row_number = 0
        while stream.tell() < last:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row_number += 1
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f'Malformed JSON at {path.name}:{offset}') from error
            yield row, ({'source_byte_offset': offset} if t2t else {'source_row': row_number})


def prepare(row, origin, source, counts):
    counts['input_rows'] += 1
    t2t = source is None
    turns = row.get('conversations' if t2t else 'messages')
    if not isinstance(turns, list) or not turns or any(not isinstance(turn, dict) for turn in turns):
        counts['source_schema_filtered'] += 1
        return None
    fields = [turn.get(key) for turn in turns for key in ('content', 'reasoning_content')]
    if any(value is not None and not isinstance(value, str) for value in fields):
        counts['source_schema_filtered'] += 1
        return None
    if sum(len(value) for value in fields if value) > CHARACTER_CEILING:
        counts['oversize_character_prefilter'] += 1
        return None
    assistant = [turn for turn in turns if turn.get('role') == 'assistant']
    has_thinking = any((turn.get('reasoning_content') or '').strip() for turn in assistant)
    if t2t:
        replay = any(turn.get('tools') or turn.get('tool_calls') or turn.get('role') == 'tool' for turn in turns)
        category = 't2t_mini_full_tools' if replay else 't2t_mini_full'
        gate = 'tool_replay' if replay else category
        meta = {'repository': 'local:sft_t2t_mini', 'path': 'dataset/sft_t2t_mini.jsonl',
                'group': f"byte:{origin['source_byte_offset']}", 'family': 'tool_replay' if replay else 't2t'}
    else:
        mode, domain = source['path'].split('/')[1:3]
        uid = row.get('uid')
        if (row.get('domain') != domain or row.get('think_type') != mode
                or not isinstance(uid, str) or not uid or has_thinking != (mode == 'think')):
            counts['source_schema_filtered'] += 1
            return None
        category = f'ultradata_{DOMAINS[domain]}_{mode}'
        gate = category
        meta = {'repository': REPO, 'revision': REVISION, 'path': source['path'],
                'group': uid, 'family': row.get('source')}
    return 'train', gate, {'conversations': turns}, {**origin, **meta, 'category': category}


def screen_batch(inputs, source, counts, terms, output):
    work = [item for row, origin in inputs if (item := prepare(row, origin, source, counts)) is not None]
    accepted, filters, hits, _, _ = base.screen_batch(work)
    terms.update(hits)
    for stats in filters.values():
        counts.update({f'screen_{key}': count for key, count in stats.items()})
    exact, grams, _ = BENCHMARKS
    for _, _, row, meta in accepted:
        if name := benchmark_overlap(row, exact, grams):
            counts[f'benchmark_{name}'] += 1
            continue
        emit_candidate(output, row, meta)
        counts['candidate_rows'] += 1
        counts['candidate_tokens'] += meta['tokens']
        counts['candidate_supervised_tokens'] += meta['supervised_tokens']


def screen_job(path, first, last, source, work):
    start = time.process_time()
    counts, terms = Counter(), Counter()
    partial = work.with_suffix('.tsv.partial')
    with partial.open('wb', buffering=1 << 20) as output:
        for batch in base.batches(raw_rows(path, first, last, source is None), 96):
            screen_batch(batch, source, counts, terms, output)
    partial.replace(work)
    result = {'source': source or {'repository': 'local:sft_t2t_mini', 'path': 'dataset/sft_t2t_mini.jsonl',
                                   'sha256': T2T_SHA256, 'first_byte': first, 'end_byte': last},
              'spool': work.name, 'counts': dict(counts), 'identity_keyword_hits': dict(terms),
              'sha256': base.digest(work), 'worker_cpu_seconds': time.process_time() - start}
    write_json(work.with_suffix('.report.json'), result)
    return work.name, counts['candidate_rows'], counts['candidate_tokens'], result['worker_cpu_seconds']


def source_jobs(source_root, work):
    for marker in sorted(set(source_root.rglob('*.ready.json'))):
        source = orjson.loads(marker.read_bytes())
        if (source['repository'], source['revision']) != (REPO, REVISION):
            raise ValueError(f'Unexpected provenance at {marker}')
        path = source_root / source['path']
        mode, domain = source['path'].split('/')[1:3]
        if mode not in {'think', 'no_think'} or domain not in DOMAINS or path.stat().st_size != source['bytes']:
            raise ValueError(f'Invalid source marker: {marker}')
        destination = work / f'{DOMAINS[domain]}_{mode}_{path.stem}.tsv'
        yield path, 0, path.stat().st_size, source, destination


def screen(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    args.work.mkdir(parents=True, exist_ok=True)
    if base.digest(T2T) != T2T_SHA256:
        raise ValueError('The local T2T source differs from the pinned token-count report')
    size = T2T.stat().st_size
    t2t_jobs = [(T2T, size * i // 24, size * (i + 1) // 24, None,
                 args.work / f't2t_{i:02}.tsv') for i in range(24)]
    seen, running = set(), {}
    deadline = time.monotonic() + args.watch_seconds
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        while True:
            for job in (*t2t_jobs, *source_jobs(args.sources, args.work)):
                output = job[-1]
                if len(running) >= 2 * args.workers:
                    break
                if output in seen or output.with_suffix('.report.json').exists():
                    continue
                if output.exists():
                    raise ValueError(f'Incomplete screen result without report: {output}')
                seen.add(output)
                running[pool.submit(screen_job, *job)] = output
            if not running:
                if time.monotonic() >= deadline:
                    break
                time.sleep(5)
                continue
            done, _ = wait(running, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                name, rows, tokens, cpu = future.result()
                print(f'[screen] {name}: {rows:,} rows, {tokens:,} tokens, {cpu:.1f} CPU s', flush=True)
                deadline = time.monotonic() + args.watch_seconds


def baseline(work):
    report_path = BASELINE / 'build.report.json'
    build = orjson.loads(report_path.read_bytes())
    verified = orjson.loads((BASELINE / 'verification.report.json').read_bytes())
    if verified['status'] != 'verified' or verified['build_report_sha256'] != base.digest(report_path):
        raise ValueError('Baseline is not verified')
    for split, info in build['splits'].items():
        pairs = ((f'{split}_identity_clean.jsonl', 'sha256'),
                 (f'{split}.provenance.jsonl.gz', 'provenance_sha256'))
        if any(base.digest(BASELINE / name) != info[key] for name, key in pairs):
            raise ValueError(f'Baseline hash mismatch: {split}')
        out = work / f'baseline_{split}.tsv'
        partial = out.with_suffix('.tsv.partial')
        with partial.open('wb', buffering=1 << 20) as stream:
            for row, meta in source_pairs(BASELINE, split):
                emit_candidate(stream, row, meta, baseline=True)
        partial.replace(out)
    return build


def source_group(source):
    if 'first_byte' in source:
        return 't2t'
    mode, domain = source['path'].split('/')[1:3]
    return f'{DOMAINS[domain]}_{mode}'


def candidates(work):
    by_group = {group: [] for group in QUOTAS}
    for path in sorted(set(work.glob('*.report.json'))):
        report = orjson.loads(path.read_bytes())
        by_group[source_group(report['source'])].append(report)
    t2t = by_group['t2t']
    if len(t2t) != 24 or sum(r['counts']['input_rows'] for r in t2t) != 905718:
        raise ValueError('Full T2T source coverage has not been established')
    boundaries = [(r['source']['first_byte'], r['source']['end_byte']) for r in t2t]
    if boundaries[0][0] != 0 or boundaries[-1][1] != T2T.stat().st_size or any(
            end != next_start for (_, end), (next_start, _) in zip(boundaries, boundaries[1:])):
        raise ValueError('T2T byte ranges are not consecutive')
    for group, reports in by_group.items():
        if group != 't2t':
            random.Random(f'{base.SEED}/2b/{group}').shuffle(reports)
    return by_group


def verify_candidate(work, report):
    path = work / report['spool']
    if base.digest(path) != report['sha256']:
        raise ValueError(f'Candidate SHA-256 mismatch: {path.name}')


def assemble(args):
    by_group = candidates(args.work)
    reports = [report for group in by_group.values() for report in group]
    upper_bound = sum(r['counts'].get('candidate_tokens', 0) for r in reports) + 127_995_048
    if upper_bound < TARGET:
        raise ValueError(f'Candidate upper bound is only {upper_bound:,} tokens')
    if args.out.exists():
        raise ValueError(f'Output already exists: {args.out}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(lambda report: verify_candidate(args.work, report), reports):
            pass
    old = baseline(args.work)
    plan = args.work / f'plan-{args.scratch.name}.tsv'
    with plan.open('x', encoding='utf-8', newline='\n') as stream:
        for group, quota in QUOTAS.items():
            for report in by_group[group]:
                path = args.work / report['spool']
                stream.write(f'{group}\t{quota}\t{path.resolve()}\n')
    mixer = args.work / 'mix_sft_2b.exe'
    subprocess.run(['rustc', '--edition=2021', '-O',
                    str(Path(__file__).with_name('mix_instinctv2_sft_2b.rs')), '-o', str(mixer)], check=True)
    result = subprocess.run([str(mixer.resolve()), str(args.work / 'baseline_train.tsv'),
                             str(args.work / 'baseline_validation.tsv'), str(plan), str(TARGET),
                             str(args.scratch), str(args.out)], capture_output=True, text=True)
    print(result.stderr, end='', flush=True)
    print(result.stdout, end='', flush=True)
    log = args.work / f'mix-{args.scratch.name}.log'
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    result.check_returncode()
    per_category = {}
    for line in result.stdout.splitlines():
        kind, *values = line.split('\t')
        if kind == 'summary':
            rows, tokens, supervised, thinking, tools, longest, duplicates, over_budget = map(int, values)
        elif kind == 'category':
            category, *counts = values
            per_category[category] = dict(zip(
                ('rows', 'tokens', 'supervised_tokens', 'thinking_rows', 'tool_rows', 'max_tokens'),
                map(int, counts), strict=True))
    if not 0 <= TARGET - tokens <= 1000:
        raise ValueError('Native assembly reported an invalid final budget')
    for name in ('validation_identity_clean.jsonl', 'validation_identity_clean.parquet',
                 'validation.provenance.jsonl.gz', 'model_16k.config.json'):
        shutil.copyfile(BASELINE / name, args.out / name)
    write_json(args.out / 'build.report.json', {
        'identity': 'InstinctV2', 'status': 'assembled', 'target_input_tokens': TARGET,
        'training_max_seq_len': CONTEXT, 'rendered_token_cap': RENDER_CAP,
        'tokenizer_sha256': base.digest(base.ROOT / 'model/tokenizer.json'),
        'chat_template_config_sha256': base.digest(base.ROOT / 'model/tokenizer_config.json'),
        'context_model_config_sha256': base.digest(args.out / 'model_16k.config.json'),
        'baseline': {'directory': BASELINE.relative_to(base.ROOT).as_posix(),
                     'tokens': old['splits']['train']['tokens'],
                     'verification_sha256': base.digest(BASELINE / 'verification.report.json')},
        'preferred_addition_token_quotas': QUOTAS, 'shortfall': TARGET - tokens,
        'train': {'rows': rows, 'tokens': tokens, 'supervised_tokens': supervised, 'thinking_rows': thinking,
                  'tool_rows': tools, 'max_tokens': longest, 'per_category': per_category},
        'validation': old['splits']['validation'],
        'native_mix': {'duplicates_or_split_overlap': duplicates, 'rows_over_remaining_budget': over_budget,
                       'shuffle_workers': 8, 'shards': 32, 'seed': 20261109},
        'screening_sources': reports, 'extra_benchmarks': benchmark_keys(base.ROOT / 'tmp')[2],
        'limitations': [
            'Full T2T input was screened; its upstream source chain is incomplete.',
            'New official records are upstream SFT targets, without local solution execution.',
            'Benchmark filters are lexical and normalized-prompt heuristics.',
            'The original 2048-token throughput calibration does not measure 16K training.',
        ],
    })


def audit_shard(data, provenance, indexed, observe=None):
    from scripts.data_builder.audit_identity_contamination import classify_row
    from scripts.data_builder.filter_anomaly_candidates import inspect_row

    counts = Counter()
    categories = Counter()
    bands, t2t = Counter(), Counter()
    max_length, longest_index = 0, 0
    for index, (row, meta) in enumerate(zip(base.rows(data), base.rows(provenance), strict=True), 1):
        turns = row['conversations']
        if indexed and meta['release_row'] != index:
            raise ValueError(f'{data.name}:{index}: provenance order changed')
        base.verify_dialogue(turns)
        prompt = next(t['content'] for t in turns if t['role'] == 'user')
        if base.fingerprint(base.normalize_text(prompt)) != meta['first_user_sha256']:
            raise ValueError(f'{data.name}:{index}: provenance prompt mismatch')
        if verdict := classify_row(row):
            if (verdict['hard_claim'] or verdict['assistant_content_identity_review']
                    or verdict['assistant_reasoning_identity_review']
                    or verdict['assistant_agreement_with_user_claim']
                    or verdict['assistant_brand_affiliation_facts']
                    or verdict['assistant_commercial_affiliation_claims']):
                raise ValueError(f'{data.name}:{index}: identity audit requires review')
        if inspect_row(row, 'identity') or any(re.search(r'instinct\s*v\s*1', t['content'], re.I) for t in turns):
            raise ValueError(f'{data.name}:{index}: identity triage requires review')
        if meta.get('path') == 'dataset/sft_t2t_mini.jsonl' and meta['tool_calls']:
            base.verify_replay(turns)
        if meta['category'] in {'identity', 'identity_t2t_mini'} and 'InstinctV2' not in turns[-1]['content']:
            raise ValueError(f'{data.name}:{index}: identity anchor mismatch')
        if not 0 < meta['supervised_tokens'] <= meta['tokens'] <= RENDER_CAP:
            raise ValueError(f'{data.name}:{index}: invalid token counts')
        if observe is not None:
            observe(row, meta)
        counts.update(rows=1, tokens=meta['tokens'], supervised_tokens=meta['supervised_tokens'],
                      thinking_rows=int(bool(meta['thinking'])), tool_rows=int(bool(meta['tool_calls'])))
        categories[meta['category']] += meta['tokens']
        bands[length_band(meta['tokens'])] += 1
        if (meta.get('repository') == 'local:sft_t2t_mini'
                or meta.get('path') == 'dataset/sft_t2t_mini.jsonl'):
            t2t.update(rows=1, tokens=meta['tokens'])
        if meta['tokens'] > max_length:
            max_length, longest_index = meta['tokens'], index - 1
    return {**counts, 'max_tokens': max_length, 'longest_index': longest_index,
            'source_tokens': dict(categories), 'length_bands': dict(bands), 't2t_mini': dict(t2t),
            'jsonl_sha256': base.digest(data),
            'provenance_sha256': base.digest(provenance), 'identity_review_rows': 0}


def compile_and_verify(compiler, source, parquet, rows, threads):
    if not parquet.exists():
        subprocess.run([str(compiler), 'compile', '-i', str(source), '-o', str(parquet),
                        '--format', 'sft', '--threads', str(threads), '--quiet'], check=True)
    result = subprocess.run([str(compiler), 'verify', str(source), str(parquet), '--format', 'sft',
                             '--rows', str(rows)], capture_output=True, text=True, check=True)
    if f'compared={rows} value_mismatches=0 row_count=match' not in result.stdout:
        raise ValueError(f'Full Parquet comparison was not confirmed: {parquet.name}')
    return {'parquet_sha256': base.digest(parquet), 'parquet_bytes': parquet.stat().st_size,
            'compared_rows': rows, 'value_mismatches': 0}


def smoke(out, reports):
    import datasets  # noqa: F401, before torch on Windows
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from scripts.data_loader.lm_dataset import SFTDataset
    from scripts.data_loader.source_format import resolve_source

    source = resolve_source(out / 'train_parquet')
    if not source.is_parquet or len(source.files) != len(reports):
        raise ValueError('Training directory does not resolve to all Parquet shards')
    schema = pq.read_schema(source.files[0]).remove_metadata()
    if any(not pq.read_schema(path).remove_metadata().equals(schema) for path in source.files[1:]):
        raise ValueError('Training Parquet shards disagree on their Arrow schema')
    name, info = max(reports.items(), key=lambda item: item[1]['max_tokens'])
    tokenizer = AutoTokenizer.from_pretrained(base.ROOT / 'model', local_files_only=True)
    sample = SFTDataset(str(out / 'train_parquet' / name), tokenizer, max_length=CONTEXT,
                        packing=True, packing_mode='bucket', seq_bucket=1, packing_num_proc=1,
                        bucket_token_budget_override=CONTEXT, sample_indices=[info['longest_index']])
    tokens, labels, sequence_ids = sample[0]
    active = labels.ne(-100)
    target = tokenizer.decode(labels[active].tolist(), skip_special_tokens=False)
    if not (len(sample) == 1 and 8192 < info['max_tokens'] <= len(tokens) == len(sequence_ids) <= CONTEXT
            and active.any() and target.rstrip().endswith(tokenizer.eos_token)):
        raise ValueError('The longest sample failed the 16K trainer packing check')
    return {'parquet_files': len(source.files), 'longest_shard': name,
            'rendered_tokens': info['max_tokens'], 'packed_block_tokens': len(tokens),
            'supervised_tokens': int(active.sum()), 'final_assistant_marker_present': True}


def verify(args):
    out = args.out
    build = orjson.loads((out / 'build.report.json').read_bytes())
    compiler = base.ROOT / 'dataset_compiler/target/release/dataset_compiler.exe'
    train = sorted((out / 'train_jsonl').glob('*.jsonl'))
    if len(train) != build['native_mix']['shards']:
        raise ValueError('The native mix is missing one or more source shards')
    (out / 'train_parquet').mkdir(exist_ok=True)
    totals, categories, bands, t2t, reports = Counter(), Counter(), Counter(), Counter(), {}
    with ProcessPoolExecutor(max_workers=args.workers) as audit, ThreadPoolExecutor(max_workers=4) as compile_pool:
        pending = {audit.submit(audit_shard, source, out / 'provenance' / source.name, True): source
                   for source in train}
        validation = audit.submit(audit_shard, out / 'validation_identity_clean.jsonl',
                                  out / 'validation.provenance.jsonl.gz', False)
        jobs = {}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                source = pending.pop(future)
                audited = future.result()
                path = out / 'train_parquet' / source.with_suffix('.parquet').name
                jobs[compile_pool.submit(compile_and_verify, compiler, source, path, audited['rows'], 4)] = path.name
                reports[path.name] = audited
                totals.update({key: audited[key] for key in ('rows', 'tokens', 'supervised_tokens', 'thinking_rows', 'tool_rows')})
                categories.update(audited['source_tokens'])
                bands.update(audited['length_bands'])
                t2t.update(audited['t2t_mini'])
                print(f"[audit] {source.name}: {audited['rows']:,} rows, {audited['tokens']:,} tokens", flush=True)
        for job, name in jobs.items():
            reports[name].update(job.result())
            print(f'[verified] {name}: full value comparison passed', flush=True)
        validation_report = validation.result()
    validation_report.update(compile_and_verify(compiler, out / 'validation_identity_clean.jsonl',
                                                out / 'validation_identity_clean.parquet',
                                                validation_report['rows'], 4))
    for key, value in totals.items():
        if build['train'][key] != value:
            raise ValueError(f'Train {key} do not conserve across native mix and verification')
    if (dict(categories) != {category: counts['tokens'] for category, counts in build['train']['per_category'].items()}
            or max(report['max_tokens'] for report in reports.values()) != build['train']['max_tokens']):
        raise ValueError('Source or context statistics do not conserve')
    for key, audit_key in (('rows', 'rows'), ('tokens', 'tokens'), ('sha256', 'jsonl_sha256'),
                           ('provenance_sha256', 'provenance_sha256')):
        if validation_report[audit_key] != build['validation'][key]:
            raise ValueError(f'Validation {key} mismatch')
    loader = smoke(out, reports)
    write_json(out / 'verification.report.json', {
        'status': 'verified', 'identity': 'InstinctV2', 'build_report_sha256': base.digest(out / 'build.report.json'),
        'train': dict(totals), 'train_shards': dict(sorted(reports.items())), 'validation': validation_report,
        'length_bands_upper_inclusive': dict(sorted(bands.items(), key=lambda item: int(item[0]))),
        't2t_mini': dict(t2t),
        'identity_method': 'Full source audit and every-field equality on every corresponding Parquet row.',
        'prompt_overlap': 0, 'trainer_16k_packing': loader,
    })
    print(f"[verified] {totals['rows']:,} rows and {totals['tokens']:,} input tokens", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('screen')
    prepare.add_argument('sources', type=Path)
    prepare.add_argument('work', type=Path)
    prepare.add_argument('--workers', type=int, default=12)
    prepare.add_argument('--watch-seconds', type=int, default=180)
    mix = sub.add_parser('assemble')
    mix.add_argument('work', type=Path)
    mix.add_argument('scratch', type=Path)
    mix.add_argument('out', type=Path)
    check = sub.add_parser('verify')
    check.add_argument('out', type=Path)
    check.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    {'screen': screen, 'assemble': assemble, 'verify': verify}[args.command](args)


if __name__ == '__main__':
    main()
