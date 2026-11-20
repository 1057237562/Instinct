"""Expand the audited InstinctV2 mix with fresh, token-budgeted examples."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
import warnings

import orjson
import pyarrow.parquet as pq

import build_instinctv2_sft as base
from augment_instinctv2_sft import source_pairs

BASELINE = base.DATA / 'instinctv2_sft_v2'
QUOTAS = {'magicoder_fresh': 32_000_000, 'numina_math': 18_000_000,
          't2t_fresh': 23_000_000, 'ultrachat_fresh': 19_000_000}
NUMINA_FAMILIES = {'cn_k12', 'cn_contest', 'orca_math'}
NUMBER = re.compile(r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?(?:\\%)?')
FENCE = re.compile(r'(?m)^' + chr(96) * 3)
EXTRA_BENCHMARKS = {
    'MBPP+': ('MbppPlus-v0.2.0.jsonl.gz', 'prompt', 378,
              'https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0'),
    'MATH-500': ('MATH-500-6e4ed1a.jsonl', 'problem', 500,
                 'https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'),
    'IFEval': ('IFEval-966cd89.jsonl', 'prompt', 541,
               'https://huggingface.co/datasets/google/IFEval/tree/966cd89545d6b6acfd7638bc708b98261ca58e84'),
}


def pair(question, answer, reasoning=''):
    target = {'role': 'assistant', 'content': answer}
    if reasoning:
        target['reasoning_content'] = reasoning
    return {'conversations': [{'role': 'user', 'content': question}, target]}


def source_files():
    return {
        'magicoder_fresh': [
            ('ise-uiuc/Magicoder-OSS-Instruct-75K', base.DATA / 'magicoder-oss-75k/data-oss_instruct-decontaminated.jsonl'),
            ('ise-uiuc/Magicoder-Evol-Instruct-110K', base.DATA / 'magicoder-110k/data-evol_instruct-decontaminated.jsonl'),
        ],
        'numina_math': [('project-numina/NuminaMath-1.5', p)
                        for p in sorted((base.DATA / 'numinamath-1.5/data').glob('train-*.parquet'))],
        't2t_fresh': [('local:sft_t2t_mini', base.DATA / 'sft_t2t_mini.jsonl')],
        'ultrachat_fresh': [('HuggingFaceH4/ultrachat_200k', p)
                            for p in sorted((base.DATA / 'ultrachat-200k/data').glob('train_sft-*.parquet'))],
    }


def math_candidate(raw, mode):
    if (raw['source'] not in NUMINA_FAMILIES or raw['question_type'] != 'math-word-problem'
            or raw['problem_is_valid'] != 'Yes' or raw['solution_is_valid'] != 'Yes'):
        return None
    question, solution, answer = (str(raw[key] or '').strip() for key in ('problem', 'solution', 'answer'))
    if not question or not NUMBER.fullmatch(answer) or len(solution) > 8500:
        return None
    boxes = re.findall(r'\\boxed\{([^{}]+)\}', solution)
    if not boxes or boxes[-1].strip() != answer or len(solution) - solution.rfind(r'\boxed{') > 200:
        return None
    if re.search(r'\\(?:includegraphics|begin\{tikzpicture\})|\[(?:asy|diagram)\]', question, re.I):
        return None
    if mode < 7:
        # Keep the original conclusion with its units and qualifiers in the visible answer.
        paragraphs = re.split(r'\n\s*\n', solution)
        if len(paragraphs) >= 2 and r'\boxed{' in paragraphs[-1] and len(paragraphs[-1]) <= 450:
            return pair(question, paragraphs[-1], '\n\n'.join(paragraphs[:-1]))
    return pair(question, solution)


def convert(category, raw, number):
    if category == 'numina_math':
        return math_candidate(raw, number % 10)
    if category == 'ultrachat_fresh':
        return {'conversations': raw['messages']}
    if category == 't2t_fresh':
        turns = raw.get('conversations') or []
        if not turns or not 12 <= len(turns[0].get('content', '').strip()) <= 6000:
            return None
        if any(t.get('tools') or t.get('tool_calls') or t['role'] == 'tool' for t in turns):
            return None
        return raw
    question = raw.get('instruction', raw.get('problem'))
    solution = raw.get('response', raw.get('solution'))
    if raw.get('lang', 'python') != 'python' or not question or not solution or not FENCE.search(solution):
        return None
    return pair(question, solution)


def screened(items):
    start = time.process_time()
    ready, counts = [], Counter()
    for category, raw, meta in items:
        counts[category, 'sampled'] += 1
        sample = convert(category, raw, meta['source_row'])
        if sample is None:
            counts[category, 'source_filtered'] += 1
            continue
        # The shared code gate checks Python syntax and benchmark AST/function signatures.
        gate = 'magicoder' if category == 'magicoder_fresh' else category
        ready.append(('train', gate, sample, {'category': category, **meta}))
    accepted, filters, hits, pid, _ = base.screen_batch(ready)
    for (_, gate), result in filters.items():
        category = 'magicoder_fresh' if gate == 'magicoder' else gate
        for name, count in result.items():
            counts[category, name] += count
    return accepted, counts, hits, pid, time.process_time() - start


def parquet_rows(path):
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for number, row in enumerate(batch.to_pylist(), offset + 1):
            yield number, row
        offset += batch.num_rows


def sample_source(category, files):
    for repo, path in files:
        rng = random.Random(f'{base.SEED}/expansion/{path.name}')
        candidates = parquet_rows(path) if path.suffix == '.parquet' else enumerate(base.rows(path), 1)
        for number, row in candidates:
            if category == 'numina_math' and (row['source'] not in NUMINA_FAMILIES or rng.random() >= .48):
                continue
            if category == 'ultrachat_fresh' and rng.random() >= .35:
                continue
            source_id = row.get('prompt_id') or f'{path.name}:{number}'
            yield category, row, {'path': path.relative_to(base.ROOT).as_posix(), 'source_row': number,
                                  'repository': repo, 'group': str(source_id),
                                  'family': row.get('source', repo)}


def sample_t2t(work, path, limit):
    rng = random.Random(f'{base.SEED}/expansion/t2t')
    # The tail of the source is tool replay and is held to the baseline's stricter policy.
    indices = rng.sample(range(2, 771075), limit)
    metadata = {number: ('train', 't2t_fresh', {'group': f't2t:{number}',
                 'repository': 'local:sft_t2t_mini', 'family': 't2t'}) for number in indices}
    for _, category, row, meta in base.selected_rows({path: metadata}, work):
        yield category, row, meta


def screen(work, workers, t2t_samples):
    work.mkdir(parents=True, exist_ok=False)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    sources = source_files()
    counts, terms, cpu = Counter(), Counter(), Counter()
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=base.init_worker) as pool:
        for category, files in sources.items():
            before = time.perf_counter()
            items = (sample_t2t(work, files[0][1], t2t_samples) if category == 't2t_fresh'
                     else sample_source(category, files))
            with (work / f'{category}.jsonl').open('xb') as spool:
                for index, window in enumerate(base.batches(base.batches(items, 256), 2 * workers), 1):
                    for found, filtered, hits, pid, elapsed in pool.map(screened, window):
                        counts.update(filtered)
                        terms.update(hits)
                        cpu[str(pid)] += elapsed
                        for _, _, row, meta in found:
                            base.emit(spool, {'row': row, 'meta': meta})
                    if index % 5 == 0:
                        print(f'[{category}] sampled={counts[category, "sampled"]:,} '
                              f'eligible={counts[category, "eligible"]:,}', flush=True)
            print(f'[{category}] eligible={counts[category, "eligible"]:,} '
                  f'elapsed={time.perf_counter() - before:.1f}s', flush=True)
    files = [(category, repo, path) for category, items in sources.items() for repo, path in items]
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(base.digest, (path for _, _, path in files)))
    report = {
        'source_files': [{'category': category, 'repository': repo, 'revision': None,
                          'path': path.relative_to(base.ROOT).as_posix(), 'sha256': digest,
                          'bytes': path.stat().st_size}
                         for (category, repo, path), digest in zip(files, hashes, strict=True)],
        'revision_note': 'Where the original download commit is unknown, the SHA-256 pins the exact local file.',
        'counts': {category: {key: value for (source, key), value in counts.items() if source == category}
                   for category in sources},
        'candidate_sha256': {category: base.digest(work / f'{category}.jsonl') for category in sources},
        'identity_keyword_hits': dict(terms), 'worker_cpu_seconds': dict(cpu),
        'screen_wall_seconds': time.perf_counter() - start, 't2t_sample_size': t2t_samples,
    }
    write_json(work / 'screen.report.json', report)


def write_json(path, value):
    with path.open('xb') as stream:
        stream.write(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b'\n')


def benchmark_keys(directory):
    exact, grams, sources = {}, {}, {}
    for name, (file, key, expected, url) in EXTRA_BENCHMARKS.items():
        path = directory / file
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as stream:
            prompts = [base.normalize_text(json.loads(line)[key]) for line in stream]
        if len(prompts) != expected or any(not prompt for prompt in prompts):
            raise ValueError(f'{name}: unexpected benchmark contents')
        for prompt in prompts:
            exact[prompt] = name
            words = prompt.split()
            for i in range(len(words) - 12):
                grams[tuple(words[i:i + 13])] = name
        sources[name] = {'url': url, 'rows': expected, 'sha256': base.digest(path)}
    return exact, grams, sources


def benchmark_overlap(row, exact, grams):
    for turn in row['conversations']:
        if turn['role'] != 'user':
            continue
        prompt = base.normalize_text(turn['content'])
        if prompt in exact:
            return exact[prompt]
        words = prompt.split()
        for i in range(len(words) - 12):
            if name := grams.get(tuple(words[i:i + 13])):
                return name
    return None


def answer_key(row):
    answer = '\n'.join(turn['content'] for turn in row['conversations'] if turn['role'] == 'assistant')
    return base.fingerprint(base.normalize_text(answer)) if len(answer) >= 100 else None


def keys(pairs):
    from scripts.data_builder.build_quality_python_sft import blocks, code_key

    prompts, groups, code, answers = set(), set(), set(), set()
    for row, meta in pairs:
        prompts.add(meta['first_user_sha256'])
        groups.add((meta['category'], meta['group']))
        if key := answer_key(row):
            answers.add(key)
        for turn in row['conversations']:
            if turn['role'] != 'assistant':
                continue
            for block in blocks(turn['content']):
                try:
                    code.add(code_key(block))
                except (SyntaxError, ValueError, RecursionError):
                    continue
    return prompts, groups, code, answers


def candidate_pairs(work, category, blocked_groups, blocked_code, counts):
    from scripts.data_builder.build_quality_python_sft import blocks, code_key

    for entry in base.rows(work / f'{category}.jsonl'):
        row, meta = entry['row'], entry['meta']
        if (category, meta['group']) in blocked_groups:
            counts['duplicate_group'] += 1
            continue
        if category == 'magicoder_fresh':
            code = '\n'.join(blocks(row['conversations'][-1]['content']))
            try:
                compile(code, '<sft candidate>', 'exec')
            except (SyntaxError, ValueError, RecursionError):
                counts['python_compile_filtered'] += 1
                continue
            key = code_key(code)
            if key in blocked_code:
                counts['duplicate_python_ast'] += 1
                continue
            meta['python_ast_sha256'] = key
        yield row, meta


def pick(pool, seen, blocked_code, answers, exact, grams, target, counts, rng):
    chosen, tokens = [], 0
    for row, meta in base.ordered_by_family(pool, rng):
        key, code = meta['first_user_sha256'], meta.get('python_ast_sha256')
        if key in seen or code and code in blocked_code:
            counts['duplicate_selected_prompt_or_ast'] += 1
            continue
        if tokens + meta['tokens'] > target:
            continue
        if benchmark := benchmark_overlap(row, exact, grams):
            counts[f'benchmark_{benchmark}'] += 1
            continue
        answer = answer_key(row)
        if answer and answer in answers:
            counts['duplicate_long_answer'] += 1
            continue
        chosen.append((row, meta))
        seen.add(key)
        if code:
            blocked_code.add(code)
        if answer:
            answers.add(answer)
        tokens += meta['tokens']
        if target - tokens < 12:
            break
    return chosen, tokens


def assemble(work, out, benchmarks):
    warnings.filterwarnings('ignore', category=SyntaxWarning, module=__name__)
    audit = orjson.loads((work / 'screen.report.json').read_bytes())
    baseline = orjson.loads((BASELINE / 'build.report.json').read_bytes())
    verified = orjson.loads((BASELINE / 'verification.report.json').read_bytes())
    if verified['status'] != 'verified' or verified['build_report_sha256'] != base.digest(BASELINE / 'build.report.json'):
        raise ValueError('Baseline build report differs from the verified release')
    for split, info in baseline['splits'].items():
        for name, hash_key in ((f'{split}_identity_clean.jsonl', 'sha256'),
                               (f'{split}.provenance.jsonl.gz', 'provenance_sha256')):
            if base.digest(BASELINE / name) != info[hash_key]:
                raise ValueError(f'Baseline hash mismatch for {name}')
    validation = list(source_pairs(BASELINE, 'validation'))
    exact, grams, benchmark_sources = benchmark_keys(benchmarks)
    train, baseline_exclusions = [], Counter()
    for row, meta in source_pairs(BASELINE, 'train'):
        if benchmark := benchmark_overlap(row, exact, grams):
            baseline_exclusions[benchmark] += 1
        else:
            train.append((row, meta))
    seen, groups, code, answers = keys(train + validation)
    counts, selected = {}, {}
    rng = random.Random(f'{base.SEED}/instinctv2-sft-v3')
    for category, quota in QUOTAS.items():
        if base.digest(work / f'{category}.jsonl') != audit['candidate_sha256'][category]:
            raise ValueError(f'Candidate hash mismatch for {category}')
        stats = Counter()
        candidates = list(candidate_pairs(work, category, groups, code, stats))
        additions, tokens = pick(candidates, seen, code, answers, exact, grams, quota, stats, rng)
        selected[category] = {'rows': len(additions), 'input_tokens': tokens, 'target': quota}
        counts[category] = dict(stats)
        train.extend(additions)
        print(f'[{category}] selected {len(additions):,} rows and {tokens:,}/{quota:,} tokens', flush=True)
    rng.shuffle(train)
    out.mkdir(parents=True, exist_ok=False)
    data_path = out / 'train_identity_clean.jsonl'
    origin_path = out / 'train.provenance.jsonl.gz'
    by_category, tokens_by_category, supervised_by_category, trace_tokens = (Counter() for _ in range(4))
    with data_path.open('xb') as data, origin_path.open('xb') as raw, gzip.GzipFile(
            fileobj=raw, mode='wb', mtime=0, compresslevel=4) as origin:
        for index, (row, meta) in enumerate(train, 1):
            category = meta['category']
            by_category[category] += 1
            tokens_by_category[category] += meta['tokens']
            supervised_by_category[category] += meta['supervised_tokens']
            if meta['thinking']:
                trace_tokens[category] += meta['tokens']
            base.emit(data, row)
            base.emit(origin, {**meta, 'row': index})
    for name in ('validation_identity_clean.jsonl', 'validation.provenance.jsonl.gz'):
        shutil.copyfile(BASELINE / name, out / name)
    total = sum(tokens_by_category.values())
    target = baseline['splits']['train']['tokens'] + sum(QUOTAS.values())
    calibration = baseline['budget']
    estimated_tps = calibration['estimated_nonpadding_input_tps'] * (1 - calibration['wall_time_reserve_fraction'])
    write_json(out / 'build.report.json', {
        'identity': 'InstinctV2', 'status': 'jsonl_built', 'training_max_seq_len': 2048,
        'tokenizer_sha256': base.digest(base.ROOT / 'model/tokenizer.json'),
        'chat_template_config_sha256': base.digest(base.ROOT / 'model/tokenizer_config.json'),
        'baseline': {'directory': 'dataset/instinctv2_sft_v2', 'train_rows': baseline['splits']['train']['rows'],
                     'train_tokens': baseline['splits']['train']['tokens'],
                     'verification_sha256': base.digest(BASELINE / 'verification.report.json')},
        'budget': {'target_input_tokens': target, 'realized_input_tokens': total, 'unfilled_tokens': target - total,
                   'one_pass_hours_with_reserve': total / (estimated_tps * 3600),
                   'hours_assumed': 8, 'input_token_capacity': calibration['nonpadding_token_capacity']['8'],
                   'calibration': calibration['calibration'],
                   'calibration_log_sha256': calibration['calibration_log_sha256'],
                   'wall_time_reserve_fraction': calibration['wall_time_reserve_fraction'],
                   'estimated_optimizer_updates': math.ceil(total / (2048 * .9916 * 8)),
                   'comparison_input_token_checkpoints': [36_000_000, 72_000_000, total]},
        'official_sft': baseline['official_sft'],
        'selection': selected, 'selection_filters': counts, 'screening': audit,
        'extra_benchmark_screen': {'method': 'exact normalized user prompt or a shared 13-word span',
                                   'sources': benchmark_sources, 'baseline_exclusions': dict(baseline_exclusions)},
        'limitations': [
            'Magicoder additions passed static Python checks and structural deduplication; they were not executed.',
            'NuminaMath additions require upstream validity flags and boxed-answer agreement; derivations were not independently solved.',
            'T2T has incomplete upstream provenance; replay is sampled and the original source remains unchanged.',
            'Baseline licensing limitations still apply, including KodCode CC BY-NC 4.0.',
            'Deduplication and benchmark overlap filters are lexical and structural heuristics.',
        ],
        'splits': {'train': {'rows': len(train), 'tokens': total,
                             'sha256': base.digest(data_path), 'provenance_sha256': base.digest(origin_path),
                             'source_rows': dict(by_category), 'source_tokens': dict(tokens_by_category),
                             'source_supervised_tokens': dict(supervised_by_category),
                             'thinking_rows': sum(meta['thinking'] for _, meta in train),
                             'thinking_row_input_tokens': dict(trace_tokens),
                             'supervised_tokens': sum(supervised_by_category.values()),
                             'max_tokens': max(meta['tokens'] for _, meta in train)},
                   'validation': baseline['splits']['validation']},
    })
    print(f'[train] rows={len(train):,} input={total:,} supervised={sum(supervised_by_category.values()):,} '
          f'estimated_hours={total / (estimated_tps * 3600):.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('screen')
    prepare.add_argument('work', type=Path)
    prepare.add_argument('--workers', type=int, default=12)
    prepare.add_argument('--t2t-samples', type=int, default=150000)
    release = sub.add_parser('assemble')
    release.add_argument('work', type=Path)
    release.add_argument('output', type=Path)
    release.add_argument('--benchmarks', type=Path, default=base.ROOT / 'tmp')
    args = parser.parse_args()
    if args.command == 'screen':
        screen(args.work, args.workers, args.t2t_samples)
    else:
        assemble(args.work, args.output, args.benchmarks)


if __name__ == '__main__':
    main()
