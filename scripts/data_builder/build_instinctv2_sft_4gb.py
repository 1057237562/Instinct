"""Build the byte-budgeted InstinctV2 coding and tool-use SFT mix."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

import orjson
import pyarrow.parquet as pq

import build_instinctv2_sft as base
import build_instinctv2_sft_2b as previous
from build_clean_easy_competitive_sft import PROMPT_PREFIX, clean_reasoning, easy_difficulty, normalized_splits
from expand_instinctv2_sft import benchmark_overlap, pair, write_json


ROOT = base.ROOT
WORK = ROOT / 'tmp/instinctv2_sft_4gb_work'
SOURCES = ROOT / 'tmp/instinctv2_sft_4gb_sources'
HOLDOUTS = ROOT / 'tmp/instinctv2_sft_4gb_holdouts'
CACHED = ROOT / 'tmp/instinctv2_sft_2b_screen'
BASELINE = ROOT / 'dataset/instinctv2_sft_2b'
CAP = 16320
BUDGET = 4_000_000_000
QUOTAS = {'contest': 1_400_000_000, 'code': 700_000_000, 'tools': 600_000_000,
          'general': 900_000_000, 'math': 400_000_000}
FENCE = re.compile(r'```(?:python3?|py|c\+\+|cpp|cc)?[^\S\n]*\n(.*?)\n[ \t]*```', re.I | re.S)
FILE_IO = re.compile(r'\bfreopen\s*\(|\bopen\s*\([\'\"][^\'\"]+\.(?:in|out)[\'\"]')


def holdout_prompts():
    for name in ('bfcl', 'lcb'):
        path = HOLDOUTS / f'{name}.json'
        info = orjson.loads(path.read_bytes())
        yield info['benchmark'], info['prompts']
    for file in sorted((ROOT / 'dataset/code_contests/_hf_parquet/data').glob('*.parquet')):
        if file.name.startswith(('valid-', 'test-')):
            values = pq.read_table(file, columns=['description'])['description'].to_pylist()
            yield f'CodeContests {file.name}', values


def init_worker():
    previous.init_worker()
    global EXACT, GRAMS
    EXACT, GRAMS, _ = previous.BENCHMARKS
    for name, prompts in holdout_prompts():
        for prompt in prompts:
            normalized = base.normalize_text(prompt)
            EXACT[normalized] = name
            if not name.startswith('BFCL'):
                words = normalized.split()
                GRAMS.update((tuple(words[i:i + 13]), name) for i in range(len(words) - 12))


def arguments(value):
    value = orjson.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError('arguments are not an object')
    return value


def align_returns(calls, responses):
    call_ids = [call.get('id') for call in calls]
    response_ids = [response.get('tool_call_id') for response in responses]
    if any(call_ids) or any(response_ids):
        if not all(call_ids) or not all(response_ids) or len(set(call_ids)) != len(calls):
            raise ValueError('incomplete tool call identifiers')
        indexed = dict(zip(response_ids, responses, strict=True))
        if set(call_ids) != set(indexed):
            raise ValueError('call and return identifiers differ')
        responses = [indexed[call_id] for call_id in call_ids]
    return responses


@lru_cache(maxsize=512)
def validated_schema(encoded):
    from jsonschema import Draft7Validator
    from jsonschema.exceptions import SchemaError

    aliases = {'dict': 'object', 'list': 'array', 'str': 'string', 'int': 'integer',
               'float': 'number', 'bool': 'boolean'}

    def normalize(value):
        if isinstance(value, dict):
            result = {key: normalize(part) for key, part in value.items()}
            if '$ref' in value:
                raise ValueError('referenced tool schemas require an explicit resolver')
            kind = value.get('type')
            if isinstance(kind, str):
                result['type'] = aliases.get(kind, kind)
            elif isinstance(kind, list) and all(isinstance(part, str) for part in kind):
                result['type'] = [aliases.get(part, part) for part in kind]
            return result
        if isinstance(value, list):
            return [normalize(part) for part in value]
        return value

    result = normalize(orjson.loads(encoded))
    try:
        Draft7Validator.check_schema(result)
    except SchemaError as error:
        raise ValueError('invalid tool parameter schema') from error
    return result


def tool_episode(raw):
    tools = raw['tools']
    if not isinstance(tools, list) or not tools or len(tools) > 40:
        raise ValueError('invalid tool list')
    tools = [{'type': 'function', 'function': dict(t['function'])} for t in tools if t.get('type') == 'function']
    if not tools or any(not isinstance(t['function'].get('parameters'), dict) for t in tools):
        raise ValueError('missing function schemas')
    for tool in tools:
        fn = tool['function']
        fn['parameters'] = validated_schema(orjson.dumps(fn['parameters']))
    turns = list(raw['messages'])
    system = turns.pop(0).get('content', '') if turns and turns[0]['role'] == 'system' else ''
    if turns and turns[0]['role'] == 'assistant' and not turns[0].get('tool_calls'):
        turns.pop(0)  # Initial agent greeting has no user request to supervise.
    if turns and turns[-1]['role'] == 'user':
        turns.pop()  # A trailing unanswered user turn adds no target.
    if not turns or turns[-1]['role'] != 'assistant' or turns[-1].get('tool_calls'):
        raise ValueError('incomplete trajectory')
    if any(m.get('loss') in (0, False, '0', 'false') for m in turns if m['role'] == 'assistant'):
        raise ValueError('upstream masked assistant turn')
    converted = [{'role': 'system', 'content': system or '', 'tools': tools}]
    index, calls_total, parallel = 0, 0, 0
    while index < len(turns):
        turn = turns[index]
        index += 1
        role = turn['role']
        if role not in ('user', 'assistant'):
            raise ValueError('unexpected message outside call block')
        target = {'role': role, 'content': turn.get('content') or ''}
        if role == 'assistant' and turn.get('reasoning_content'):
            target['reasoning_content'] = turn['reasoning_content']
        converted.append(target)
        calls = turn.get('tool_calls') or []
        if not calls:
            continue
        if role != 'assistant' or len(calls) > 8:
            raise ValueError('invalid call block')
        responses = turns[index:index + len(calls)]
        if len(responses) != len(calls) or any(t['role'] != 'tool' for t in responses):
            raise ValueError('incomplete tool returns')
        responses = align_returns(calls, responses)
        normalized = [call.get('function', call) for call in calls]
        target['tool_calls'] = [{'function': {'name': f['name'], 'arguments': arguments(f['arguments'])}}
                                for f in normalized]
        for response in responses:
            content = response.get('content')
            if isinstance(content, str):
                try:
                    orjson.loads(content)
                except orjson.JSONDecodeError:
                    content = base.compact(content)
            else:
                content = base.compact(content)
            converted.append({'role': 'tool', 'content': content})
        index += len(calls)
        calls_total += len(calls)
        parallel += int(len(calls) > 1)
    if calls_total > 20:
        raise ValueError('too many calls')
    sample = {'conversations': converted}
    if len(orjson.dumps(sample)) > 140_000:
        raise ValueError('oversize tool trajectory')
    return sample, {'parallel_blocks': parallel, 'source_tool_calls': calls_total}


def difficulty_band(value):
    text = str(value or '').strip().lower()
    if easy_difficulty(text):
        return 'easy'
    if text in {'intermediate', 'interview', 'medium', 'medium_easy'} or (text.isdecimal() and int(text) <= 1700):
        return 'medium'
    if text in {'hard', 'medium_hard', 'very_hard', 'competition', 'advanced'} or text.isdecimal():
        return 'hard'
    return 'unknown'


def canonical_problem(text):
    text = PROMPT_PREFIX.sub('', text, count=1)
    text = re.sub(r'^\s*(?:problem description\s*[.:]\s*|(?:the )?problem\s*:\s*)', '', text, flags=re.I)
    return base.normalize_text(text)


def raw_problem(raw):
    first = next(t['content'] for t in raw['messages'] if t['role'] == 'user')
    return PROMPT_PREFIX.sub('', first, count=1).strip()


def contest_row(problem, code, reasoning, lang, preserve_reasoning=False):
    if not isinstance(problem, str) or not isinstance(code, str) or not isinstance(reasoning, str):
        return None
    problem = problem.strip()
    if not 70 <= len(problem) <= 22000 or not 25 <= len(code) <= 18000 or FILE_IO.search(code):
        return None
    if lang == 'python':
        try:
            if not ast.parse(code).body:
                return None
        except (SyntaxError, ValueError, RecursionError):
            return None
    elif code.count('{') != code.count('}') or not re.search(r'\bmain\s*\(|\bclass\s+Solution\b', code):
        return None
    if reasoning and not preserve_reasoning:
        reasoning, _ = clean_reasoning(reasoning, code)
    prompt = f'Solve the following programming problem in {"Python" if lang == "python" else "C++"}. '
    prompt += f'Return the final solution as one `{lang}` code block.\n\n{problem}'
    return pair(prompt, f'```{lang}\n{code.strip()}\n```', reasoning or '')


def local_contest(row, meta):
    lang = meta['language']
    user, assistant = row['conversations']
    problem = user['content'].split('\n\n', 1)[-1]
    code = FENCE.findall(assistant.get('content', ''))
    if len(code) != 1:
        return None
    return contest_row(problem, code[0], assistant.get('reasoning_content') or '', lang, preserve_reasoning=True), problem


def ocr_contest(row, lang, problem):
    if not normalized_splits(row['split']) or set(normalized_splits(row['split'])) != {'train'}:
        return None
    judgement = str(row['judgement']).strip().casefold()
    try:
        passed = float(row['pass_rate']) == 1.0
    except (TypeError, ValueError):
        return None
    if judgement != 'right' or not passed:
        return None
    generation = row['r1_generation'] or ''
    reasoning, delimiter, _ = generation.partition('</think>')
    if not delimiter:
        reasoning = ''
    code = row.get('solution') or ''
    if fenced := FENCE.findall(code):
        code = fenced[-1]
    if not code:
        return None
    reasoning = reasoning.removeprefix('<think>').strip()
    reasoning = clean_long_reasoning(reasoning)
    return contest_row(problem, code, reasoning, lang, preserve_reasoning=True), problem


def clean_long_reasoning(value):
    if not isinstance(value, str) or not value:
        return ''
    value = FENCE.sub('\n', value).replace('<think>', '').replace('</think>', '').strip()
    if len(value) > 49000:
        return ''
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', value) if p.strip()]
    if len(paragraphs) >= 4 and len(set(paragraphs)) / len(paragraphs) < .85:
        return ''
    return value


def make_prompt_index(paths):
    target = WORK / 'competitive_prompts.sqlite'
    if target.exists():
        return target
    temporary = target.with_suffix('.partial')
    with sqlite3.connect(temporary) as db:
        db.execute('CREATE TABLE prompts (id TEXT PRIMARY KEY, problem TEXT NOT NULL) WITHOUT ROWID')
        seen = set()
        for path in paths:
            with path.open('rb') as stream:
                for line in stream:
                    match = re.search(rb'"question_id"\s*:\s*"([0-9a-f]{32})"', line[-640:])
                    if not match or match[1] in seen:
                        continue
                    raw = orjson.loads(line.split(b'\t', 1)[1])
                    problem = raw_problem(raw)
                    db.execute('INSERT INTO prompts VALUES (?, ?)', (raw['question_id'], problem))
                    seen.add(match[1])
        print(f'[index] {len(seen):,} train-only competitive problem statements', flush=True)
    db.close()
    temporary.replace(target)
    return target


def raw_sample_jobs(paths):
    for path in paths:
        size = path.stat().st_size
        n = max(1, (size + 83_999_999) // 84_000_000)
        language = 'python' if 'python' in path.name else 'cpp'
        for i in range(n):
            yield 'raw', path, f'raw_{path.stem}_{i:02}', {
                'repository': 'nvidia/Nemotron-SFT-Competitive-Programming-v2',
                'path': f'data/{path.stem}.jsonl', 'language': language,
                'first': size * i // n, 'last': size * (i + 1) // n}


def local_contest_jobs():
    names = ('sft_easy_medium_competitive_clean_4k', 'sft_competitive_t2t_ultrachat_16k_coding_heavy')
    for source in names:
        data = ROOT / 'dataset' / f'{source}.jsonl'
        meta = data.with_suffix('.sources.jsonl')
        if not data.exists() or not meta.exists():
            print(f'[source] unavailable: {data.name}', flush=True)
            continue
        out = WORK / 'local_pairs' / source
        marker = out / 'complete.json'
        if not marker.exists():
            out.mkdir(parents=True, exist_ok=True)
            count = 0
            with data.open('rb') as rows, meta.open('rb') as provenance:
                for lines in base.batches(zip(rows, provenance, strict=True), 20000):
                    pairs = [(row, info) for row, info in lines if b'"question_id":' in info]
                    if not pairs:
                        continue
                    with (out / f'part-{count:03}.tsv').open('wb', buffering=1 << 20) as stream:
                        for row, info in pairs:
                            stream.write(info.rstrip(b'\r\n') + b'\t' + row)
                    count += 1
            write_json(marker, {'shards': count, 'report_sha256': base.digest(data.with_suffix('.report.json'))})
        for part in sorted(out.glob('*.tsv')):
            yield 'local', part, f'local_{names.index(source)}_{part.stem}', {'path': str(data.relative_to(ROOT))}


def code_contests_jobs():
    source = ROOT / 'dataset/code_contests/code_contests.clean.jsonl'
    size = source.stat().st_size
    for part in range(16):
        yield 'accepted', source, f'accepted_{part:02}', {'first': size * part // 16, 'last': size * (part + 1) // 16,
                'repository': 'deepmind/code_contests', 'path': str(source.relative_to(ROOT))}


def downloaded_jobs():
    for marker in sorted(SOURCES.glob('*/*.ready.json')):
        info = orjson.loads(marker.read_bytes())
        path = marker.parent / Path(info['path']).name
        if path.stat().st_size != info['bytes']:
            raise ValueError(f'Download size changed: {path}')
        category = info['category']
        if category == 'tools':
            # Split by byte ranges so large tool shards can be processed concurrently.
            size = info['bytes']
            n = max(1, (size + 95_999_999) // 96_000_000)
            for i in range(n):
                yield category, path, f'{category}_{path.stem}_{i:02}', {**info, 'first': size * i // n,
                                                                        'last': size * (i + 1) // n}
        else:
            yield category, path, f'{category}_{path.stem}', info


def candidate_rows(kind, path, source, stats):
    origin = {key: source[key] for key in ('repository', 'revision', 'path') if key in source}
    if kind == 'local':
        with path.open('rb') as stream:
            for line in stream:
                meta, row = map(orjson.loads, line.split(b'\t', 1))
                stats['input'] += 1
                converted = local_contest(row, meta)
                if converted and converted[0]:
                    yield converted, {**meta, **origin, 'original_path': meta['file'], 'tier': 'static'}
        return
    if kind == 'accepted':
        for raw, where in previous.raw_rows(path, source['first'], source['last'], True):
            stats['input'] += 1
            if raw['file_io'] or not raw['description']:
                continue
            for lang, key in [('python', 'solutions_python3'), ('cpp', 'solutions_cpp')]:
                # Choose a short representative among accepted submissions for the problem.
                for code in sorted(raw[key], key=len):
                    if row := contest_row(raw['description'], code, '', lang):
                        yield (row, raw['description']), {**origin, **where, 'question_id': raw['problem_id'],
                            'language': lang, 'difficulty': raw['cf_rating'] or raw['difficulty'],
                            'tier': 'accepted_submission', 'family': raw['source']}
                        break
        return
    if kind == 'tools':
        for raw, where in previous.raw_rows(path, source['first'], source['last'], True):
            stats['input'] += 1
            try:
                row, tool_meta = tool_episode(raw)
                yield (row, None), {**origin, **where, **tool_meta, 'group': raw['uuid'], 'family': raw['source']}
            except (KeyError, ValueError, TypeError, AttributeError) as error:
                stats['tool_' + str(error).split(':')[0][:100]] += 1
        return
    if kind == 'raw':
        lang = source['language']
        with path.open('rb') as stream:
            first, last = source['first'], source['last']
            if first:
                stream.seek(first - 1)
                if stream.read(1) != b'\n':
                    stream.readline()
            while stream.tell() < last:
                stats['input'] += 1
                offset, data = stream.readline().split(b'\t', 1)
                raw = orjson.loads(data)
                if set(normalized_splits(raw['split'])) != {'train'}:
                    raise ValueError('The train-only index selected a held-out candidate')
                problem = raw_problem(raw)
                assistant = raw['messages'][-1]
                content = assistant.get('content') or ''
                reasoning = assistant.get('reasoning_content') or ''
                if '</think>' in content and not reasoning:
                    reasoning, content = content.split('</think>', 1)
                code = FENCE.findall(content)
                if len(code) != 1:
                    continue
                reasoning = clean_long_reasoning(reasoning)
                if not reasoning:
                    continue
                if row := contest_row(problem, code[0], reasoning, lang, preserve_reasoning=True):
                    yield (row, problem), {**origin, 'source_byte_offset': int(offset), 'question_id': raw['question_id'],
                                           'language': lang, 'difficulty': raw['difficulty'], 'tier': 'static',
                                           'source_id': raw['uuid'], 'family': raw['source']}
        return
    if kind.startswith('rstar_'):
        verified = kind == 'rstar_seed'
        selected = ['question', 'response', 'code']
        if verified:
            selected += ['question_id', 'verified', 'is_passed']
        else:
            selected += ['seed_question']
        number = 0
        kept = Counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=selected):
            for raw in batch.to_pylist():
                number += 1
                stats['input'] += 1
                if verified and not (raw['verified'] is True and raw['is_passed'] is True):
                    continue
                if not verified and isinstance(raw['seed_question'], str):
                    seed = {'conversations': [{'role': 'user', 'content': raw['seed_question']}]}
                    if benchmark := benchmark_overlap(seed, EXACT, GRAMS):
                        stats[f'benchmark_seed_{benchmark}'] += 1
                        continue
                problem = raw['question']
                if not isinstance(problem, str):
                    continue
                key = raw.get('question_id') or base.fingerprint(canonical_problem(problem))
                if kept[key] >= 2:
                    continue
                trace = raw.get('response') or ''
                if not isinstance(trace, str):
                    continue
                reasoning, delimiter, _ = trace.partition('</think>')
                if not delimiter:
                    reasoning = ''
                code = raw.get('code') or ''
                if isinstance(code, str) and (fences := FENCE.findall(code)):
                    code = fences[-1]
                row = contest_row(problem, code, clean_long_reasoning(reasoning), 'python', preserve_reasoning=True)
                if row:
                    kept[key] += 1
                    tier = 'rstar_verified' if verified else 'rstar_synthetic'
                    yield (row, problem), {**origin, 'source_row': number, 'question_id': str(key), 'language': 'python',
                                           'difficulty': 'unknown', 'tier': tier, 'family': kind}
        return
    lang = 'python' if kind == 'ocr2_py' else 'cpp'
    counts = Counter()
    selected = ('question', 'r1_generation', 'solution', 'judgement', 'pass_rate', 'split', 'difficulty', 'question_id', 'id', 'source')
    number = 0
    with sqlite3.connect(WORK / 'competitive_prompts.sqlite') as db:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=list(selected)):
            for raw in batch.to_pylist():
                number += 1
                stats['input'] += 1
                key = raw['question_id']
                if counts[key] >= 3:
                    continue
                problem = db.execute('SELECT problem FROM prompts WHERE id = ?', (key,)).fetchone()
                if not problem:
                    stats['missing_train_problem'] += 1
                    continue
                converted = ocr_contest(raw, lang, problem[0])
                if converted and converted[0]:
                    counts[key] += 1
                    yield converted, {**origin, 'source_row': number, 'question_id': key, 'language': lang,
                                      'difficulty': raw['difficulty'], 'source_id': raw['id'],
                                      'family': raw['source'], 'tier': 'upstream_pass_100pct'}
    db.close()


def screen_job(kind, path, name, source):
    start = time.process_time()
    counters, terms, written, sizes = Counter(), Counter(), Counter(), Counter()
    outputs = {}
    try:
        for batch in base.batches(candidate_rows(kind, path, source, counters), 96):
            items = []
            for (row, problem), meta in batch:
                if kind == 'tools':
                    label = 'tools_call' if meta['source_tool_calls'] else 'tools_no_call'
                    gate = 'tools'
                else:
                    canonical = canonical_problem(problem)
                    if canonical in EXACT:
                        counters['benchmark_filtered'] += 1
                        continue
                    meta['canonical_problem_sha256'] = base.fingerprint(canonical)
                    meta['group'] = base.fingerprint(f"{meta['language']}\n{canonical}")
                    label = f"contest_{meta['tier']}_{difficulty_band(meta['difficulty'])}"
                    gate = label
                meta['category'] = label
                items.append(('train', gate, row, meta))
            accepted, filters, hits, _, _ = base.screen_batch(items)
            terms.update(hits)
            for counts in filters.values():
                counters.update({f'screen_{key}': value for key, value in counts.items()})
            for _, _, row, meta in accepted:
                if benchmark := benchmark_overlap(row, EXACT, GRAMS):
                    counters[f'benchmark_{benchmark}'] += 1
                    continue
                label = meta['category']
                if label not in outputs:
                    outputs[label] = (WORK / f'{name}--{label}.partial').open('wb', buffering=1 << 20)
                previous.emit_candidate(outputs[label], row, meta)
                written[label] += 1
                sizes[label] += len(orjson.dumps(row)) + 1
    finally:
        for output in outputs.values():
            output.close()
    files = {}
    for label, output in outputs.items():
        partial = Path(output.name)
        final = partial.with_suffix('.tsv')
        partial.replace(final)
        files[final.name] = {'rows': written[label], 'bytes': sizes[label], 'sha256': base.digest(final)}
    result = {'source': source, 'counts': dict(counters), 'identity_keyword_hits': dict(terms), 'files': files,
              'worker_cpu_seconds': round(time.process_time() - start, 2)}
    write_json(WORK / f'{name}.report.json', result)
    return name, counters['input'], sum(written.values()), sum(sizes.values()), result['worker_cpu_seconds']


def screen(workers):
    from concurrent.futures import FIRST_COMPLETED, wait

    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        pending, submitted = {}, set()
        local = list(local_contest_jobs()) + list(code_contests_jobs())
        while True:
            new = [*(job for job in downloaded_jobs() if job[0] == 'tools'), *local]
            for kind, path, name, origin in new:
                if name in submitted or (WORK / f'{name}.report.json').exists():
                    continue
                if len(pending) >= 2 * workers:
                    break
                pending[pool.submit(screen_job, kind, path, name, origin)] = name
                submitted.add(name)
            if not pending:
                manifest = orjson.loads((SOURCES / 'manifest.json').read_bytes())
                if all((SOURCES / entry['category'] / Path(entry['path']).name).exists()
                       for entry in manifest if entry['category'] == 'tools'):
                    break
                if time.monotonic() - start > 1800:
                    raise TimeoutError('Download and screening exceeded 30 minutes')
                time.sleep(4)
                continue
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for job in done:
                pending.pop(job)
                name, scanned, eligible, size, cpu = job.result()
                print(f'[screen] {name}: {eligible:,}/{scanned:,} rows, {size / 1e6:.1f} MB, {cpu:.1f} CPU s', flush=True)


def screen_competitive(workers):
    from concurrent.futures import FIRST_COMPLETED, wait

    paths = sorted((SOURCES / 'local_raw').glob('*.tsv'))
    if len(paths) != 4:
        raise ValueError('Four raw competitive sample indices are required')
    make_prompt_index(paths)
    jobs = list(raw_sample_jobs(paths))
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        submitted, pending = set(), {}
        while True:
            ocr = [(kind, path, 'verified_' + name, info) for kind, path, name, info in downloaded_jobs()
                   if kind.startswith('ocr2_')]
            for kind, path, name, origin in [*ocr, *jobs]:
                if name in submitted or (WORK / f'{name}.report.json').exists():
                    continue
                if len(pending) >= 2 * workers:
                    break
                pending[pool.submit(screen_job, kind, path, name, origin)] = name
                submitted.add(name)
            if not pending:
                manifest = orjson.loads((SOURCES / 'manifest.json').read_bytes())
                if len(ocr) == sum(item['category'].startswith('ocr2_') for item in manifest):
                    break
                if time.monotonic() - start > 1800:
                    raise TimeoutError('Competitive source screening exceeded 30 minutes')
                time.sleep(4)
                continue
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                name, scanned, eligible, size, cpu = future.result()
                print(f'[screen] {name}: {eligible:,}/{scanned:,} rows, {size / 1e6:.1f} MB, {cpu:.1f} CPU s', flush=True)


def screen_additional(workers):
    from concurrent.futures import FIRST_COMPLETED, wait

    manifest = orjson.loads((SOURCES / 'rstar.manifest.json').read_bytes())
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        pending, submitted = set(), set()
        while not (WORK / 'selection_closed').exists():
            jobs = [(kind, path, name, info) for kind, path, name, info in downloaded_jobs() if kind.startswith('rstar_')]
            for job in jobs:
                name = job[2]
                if name in submitted or (WORK / f'{name}.report.json').exists():
                    continue
                pending.add(pool.submit(screen_job, *job))
                submitted.add(name)
            if not pending:
                if len(jobs) == len(manifest) or time.monotonic() - start > 1200:
                    break
                time.sleep(4)
                continue
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                name, scanned, eligible, size, cpu = future.result()
                print(f'[screen] {name}: {eligible:,}/{scanned:,} rows, {size / 1e6:.1f} MB, {cpu:.1f} CPU s', flush=True)


def official_contest_file(path, supplemental=False):
    name = 'supplement' if supplemental else 'official'
    out = WORK / f'{name}_{path.stem}--contest_ultradata_unknown.tsv'
    marker = WORK / f'{name}_{path.stem}.report.json'
    if marker.exists():
        return out
    count, size = 0, 0
    families = ('rStar-Coder', 'AM-Thinking-v1') if supplemental else ('OpenCodeReasoning', 'OpenCodeReasoning-2')
    needles = tuple(f'"family":"{family}"'.encode() for family in families)
    with path.open('rb') as source, out.with_suffix('.partial').open('wb', buffering=1 << 20) as stream:
        for line in source:
            if not any(needle in line for needle in needles):
                continue
            _, data, info = line.rsplit(b'\t', 2)
            row, meta = orjson.loads(data), orjson.loads(info)
            turns = row['conversations']
            prompt = next(turn['content'] for turn in turns if turn['role'] == 'user')
            if meta['family'] == 'AM-Thinking-v1' and not all(word in prompt.casefold() for word in ('input', 'output', 'constraint')):
                continue
            answer = turns[-1]['content']
            codes = FENCE.findall(answer)
            if len(codes) != 1:
                continue
            lang = 'cpp' if re.search(r'```(?:cpp|c\+\+|cc)\b', answer, re.I) else 'python'
            if not contest_row(prompt, codes[0], '', lang):
                continue
            canonical = canonical_problem(prompt)
            meta.update(category='contest_ultradata_unknown', tier='ultradata', language=lang,
                        canonical_problem_sha256=base.fingerprint(canonical),
                        group=base.fingerprint(f'{lang}\n{canonical}'))
            previous.emit_candidate(stream, row, meta)
            count += 1
            size += len(data) + 1
    out.with_suffix('.partial').replace(out)
    write_json(marker, {'source': path.name, 'files': {out.name: {'rows': count, 'bytes': size, 'sha256': base.digest(out)}}})
    return out


def official_contest(workers, supplemental=False):
    from itertools import repeat

    files = [*CACHED.glob('code_think_*.tsv'), *CACHED.glob('code_no_think_*.tsv')]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(official_contest_file, files, repeat(supplemental)):
            pass
    print(f'[source] indexed OpenCodeReasoning families from {len(files)} screened UltraData files', flush=True)


def write_pinned(target):
    counts = Counter()
    identity_from_t2t = WORK / 't2t_identity.tsv'
    excluded = set()
    with (CACHED / 'baseline_validation.tsv').open('rb') as validation:
        excluded.update(line.split(b'\t', 1)[0] for line in validation)
    with (CACHED / 'baseline_train.tsv').open('rb') as source, target.open('xb') as anchors, \
            target.with_name('local_tools.tsv').open('xb') as tools:
        with identity_from_t2t.open('rb') as authored:
            for line in authored:
                fields = line.split(b'\t', 8)
                if fields[0] not in excluded:
                    anchors.write(line)
                    excluded.add(fields[0])
                    counts['identity_bytes'] += len(fields[7]) + 1
        for line in source:
            fields = line.split(b'\t', 8)
            if fields[5] == b'identity' and fields[0] not in excluded:
                anchors.write(line)
                excluded.add(fields[0])
                counts['identity_bytes'] += len(fields[7]) + 1
            elif fields[5] in (b'tools', b't2t_mini_full_tools'):
                tools.write(line)
                counts['local_tools_bytes'] += len(fields[7]) + 1
    return counts


IDENTITY_QUESTIONS = {
    'developer': re.compile(r'谁.{0,8}(?:开发|训练|设计|创造)了?你|你.{0,9}(?:由谁|谁开发|谁训练|开发者|制造者)|'
                            r'你的.{0,6}(?:开发团队|开发背景|开发来源|创造者)|你.{0,12}(?:被创造|开发的)|'
                            r'who (?:made|created|trained|developed|built) you|who is your developer', re.I),
    'experience': re.compile(r'你.{0,8}(?:有意识|有感情|有情绪|主观意识|自我意识|自己的经历|个人经历|亲身体验)|'
                             r'你.{0,16}(?:自身|自我)的?(?:意识|认知|存在的边界)|你的.{0,5}(?:意识|个人经历)|'
                             r'do you have (?:feelings|emotions|consciousness|personal experiences)', re.I),
    'capability': re.compile(r'你能帮忙做什么|你能帮我做什么|你(?:能|可以|擅长)做(?:些)?什么|你有(?:哪些|什么)功能|'
                             r'你的(?:主要|基本)?功能是什么|what can you (?:do|help with)|what are your (?:main )?capabilities[?？]?$', re.I),
    'identity': re.compile(r'你是谁|你叫(?:什么|啥)|你的(?:模型)?(?:名字|名称)是|你(?:是|属于)(?:哪种|哪款|什么|哪个).{0,8}(?:模型|助手|人工智能)|'
                           r'你.{0,8}(?:真实身份|真实来源|哪个模型|哪款模型)|你是一个语言模型吗|你是否基于.{0,14}模型|'
                           r'(?:介绍一下你自己|请.{0,5}自我介绍)|\bwho are you\b|what(?: is|\x27s) your name|'
                           r'what model are you|introduce yourself|tell me (?:a little bit )?about yourself', re.I),
}


def t2t_identity():
    source = ROOT / 'dataset/sft_t2t_mini.jsonl'
    digest = base.digest(source)
    if digest != previous.T2T_SHA256:
        raise ValueError('Original T2T has changed since the pinned input audit')
    report = WORK / 't2t_identity.report.json'
    if report.exists() and orjson.loads(report.read_bytes()).get('policy_revision') == 3:
        return
    from scripts.data_builder.audit_identity_contamination import classify_row
    from scripts.data_builder.filter_anomaly_candidates import inspect_row
    from scripts.data_builder.instinctv2_sft_scenarios import IDENTITY

    scan_expression = r'(?i)instinct|\bl1bra\b'
    authored = re.compile(r'\bInstinct(?:V1Moe|V1|V2)?\b|\bL1bra\b', re.I)
    counters, hits = Counter(), Counter()
    approved = {}
    command = ['rg', '--color', 'never', '-n', scan_expression, str(source)]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as scanner:
        for line in scanner.stdout:
            number, content = line.split(b':', 1)
            turns = orjson.loads(content)['conversations']
            for index, turn in enumerate(turns):
                answer = turn.get('content') or ''
                if index == 0 or turn['role'] != 'assistant' or not authored.search(answer):
                    continue
                counters['assistant_identity_mentions'] += 1
                base.COUNT_TERMS(answer, hits)
                user = turns[index - 1]
                prompt = user.get('content') or ''
                if (user['role'] != 'user' or not 2 <= len(prompt) <= 160 or '\n' in prompt
                        or base.CONTROLS.search(prompt) or base.FOREIGN.search(prompt)
                        or prompt.endswith(('，', ',', ' as a digital'))):
                    continue
                topic = next((topic for topic, pattern in IDENTITY_QUESTIONS.items() if pattern.search(prompt)), None)
                if topic is None:
                    continue
                lang = 'zh' if re.search(r'[\u4e00-\u9fff]', prompt) else 'en'
                variant = 4 if lang == 'zh' else 5
                if topic == 'capability':
                    target = ('我是 InstinctV2，可以根据你提供的上下文回答问题、分析文本和辅助编程。' if lang == 'zh' else
                              'I am InstinctV2. I can answer questions, analyze text, and help with programming using the context you provide.')
                else:
                    target = IDENTITY[{'identity': 0, 'developer': 2, 'experience': 3}[topic]][variant]
                    if topic == 'developer' and IDENTITY_QUESTIONS['identity'].search(prompt):
                        target = ('我是 InstinctV2，由 L1bra 开发，是一个从头训练的语言模型。' if lang == 'zh' else
                                  'I am InstinctV2, a language model developed by L1bra and trained from scratch.')
                    if re.search(r'capabilit|functions|what (?:can|do) you do|what you can do', prompt, re.I):
                        target += ' I can help answer questions, analyze text, and write code.'
                    if re.search(r'education|personal experience', prompt, re.I):
                        target += ' I do not have personal education or life experiences.'
                    if re.search(r'why should I trust you', prompt, re.I):
                        target += ' I can make mistakes, so you should verify important claims.'
                row = pair(prompt, target)
                verdict = classify_row(row)
                if inspect_row(row, 'identity') or (verdict and any(verdict[key] for key in (
                        'hard_claim', 'assistant_content_identity_review', 'assistant_reasoning_identity_review',
                        'assistant_agreement_with_user_claim', 'assistant_brand_affiliation_facts',
                        'assistant_commercial_affiliation_claims'))):
                    counters['identity_review_filtered'] += 1
                    continue
                key = base.fingerprint(base.normalize_text(prompt))
                if key not in approved:
                    approved[key] = (row, {'path': source.relative_to(ROOT).as_posix(), 'source_row': int(number),
                                           'source_assistant_turn': index, 'repository': 'local:sft_t2t_mini',
                                           'group': f't2t_identity:{key}', 'category': 'identity_t2t_mini',
                                           'family': f'identity/{topic}', 'language': lang, 'target_normalized': True})
                else:
                    counters['duplicate_prompt'] += 1
        if scanner.wait() not in (0, 1):
            raise RuntimeError('The original T2T identity scan failed')
    selected, family_counts = [], Counter()
    for key in sorted(approved):
        row, meta = approved[key]
        family = meta['family'], meta['language']
        if family_counts[family] < 80:
            selected.append(('train', 'identity', row, meta))
            family_counts[family] += 1
    rows, filters, _, _, cpu = base.screen_batch(selected)
    for values in filters.values():
        counters.update({f'screen_{k}': v for k, v in values.items()})
    target = WORK / 't2t_identity.tsv'
    with target.with_suffix('.partial').open('wb') as output:
        for _, _, row, meta in rows:
            previous.emit_candidate(output, row, meta, baseline=True)
            counters[f'selected_{meta["family"]}_{meta["language"]}'] += 1
    target.with_suffix('.partial').replace(target)
    if not rows:
        raise ValueError('No original T2T self-identity rows passed the InstinctV2 audit')
    pending_report = report.with_suffix('.partial')
    write_json(pending_report, {'source': {'path': source.relative_to(ROOT).as_posix(), 'sha256': digest},
                               'counts': dict(counters), 'reviewed_unique_prompts': len(approved),
                               'output_rows': len(rows), 'output_sha256': base.digest(target), 'policy_revision': 3,
                               'identity_keyword_hits': dict(hits), 'worker_cpu_seconds': cpu})
    pending_report.replace(report)
    print(f'[identity] retained {len(rows):,} original T2T self-identity questions with InstinctV2 targets', flush=True)


def write_holdout_index(target):
    with (CACHED / 'baseline_validation.tsv').open('rb') as source, target.open('xb') as output:
        shutil.copyfileobj(source, output)
        prompts = {base.fingerprint(base.normalize_text(prompt)) for _, rows in holdout_prompts() for prompt in rows}
        quarantine = WORK / 'benchmark_quarantine.txt'
        if quarantine.exists():
            prompts.update(quarantine.read_text(encoding='utf-8').splitlines())
        for key in sorted(prompts):
            output.write(f'{key[:32]}\t{key[:32]}\t\t12\t1\tholdout\t0\t{{}}\t{{}}\n'.encode('ascii'))


def write_mix_plan(path, anchors, local_tools):
    import random

    def ordered(paths, key):
        files = sorted(paths)
        random.Random(f'20261113/{key}').shuffle(files)
        return files

    plans = []

    def add(group, quota, paths, low=12, high=CAP, cap=1):
        plans.extend((group, min(quota, QUOTAS[group]), f, low, high, cap) for f in paths)

    def cached(prefix):
        root = WORK / 'holdout_screened' if prefix.startswith('code_') else CACHED
        return ordered(root.glob(f'{prefix}_*.tsv'), prefix)

    tool_calls = ordered(WORK.glob('tools_*--tools_call.tsv'), 'tool_calls')
    tool_no_calls = ordered(WORK.glob('tools_*--tools_no_call.tsv'), 'tool_no_calls')
    add('tools', 60_000_000, tool_no_calls)
    add('tools', 65_000_000, [local_tools])
    add('tools', 300_000_000, tool_calls, high=4096)
    add('tools', 500_000_000, tool_calls, 4097, 8192)
    add('tools', QUOTAS['tools'], [*tool_calls, *tool_no_calls])

    contest = list(WORK.glob('*--contest_*.tsv'))
    static = [f for f in contest if '--contest_static_' in f.name]
    upstream = [f for f in contest if '--contest_upstream_pass_100pct_' in f.name]
    accepted = [f for f in contest if '--contest_accepted_submission_' in f.name]
    official = sorted((WORK / 'holdout_screened').glob('*--contest_ultradata_*.tsv'))
    rstar_verified = [f for f in contest if '--contest_rstar_verified_' in f.name]
    rstar_synthetic = [f for f in contest if '--contest_rstar_synthetic_' in f.name]
    known, total = [], QUOTAS['contest']
    add('contest', 275_000_000, [*ordered(upstream, 'upstream_first'), *ordered(rstar_verified, 'rstar_first')])
    add('contest', 390_000_000, ordered(static, 'long_first'), 8193, CAP)
    add('contest', 610_000_000, ordered(static, 'medium_first'), 4097, 8192)
    for band, quota in [('easy', 700_000_000), ('medium', 1_260_000_000), ('hard', total)]:
        verified = [f for f in [*upstream, *accepted] if f.stem.endswith('_' + band)]
        unverified = [f for f in static if f.stem.endswith('_' + band)]
        paths = [*ordered(verified, band + '/verified'), *ordered(unverified, band + '/static')]
        known.extend(paths)
        add('contest', quota, paths, high=8192)
    add('contest', total, known)
    add('contest', total, ordered(rstar_verified, 'rstar_verified'))
    add('contest', total, [*ordered(upstream, 'upstream_second'), *ordered(accepted, 'accepted_second'),
                           *ordered(rstar_verified, 'rstar_verified_second')], cap=2)
    add('contest', total, ordered(official, 'official'), high=8192)
    add('contest', total, ordered(rstar_synthetic, 'rstar_synthetic'), high=8192)
    add('contest', total, ordered(static, 'static_medium_second'), 4097, 8192, cap=2)
    add('contest', total, ordered(static, 'static_second'), high=4096, cap=2)
    add('contest', total, [*ordered(official, 'official_long'), *ordered(static, 'static_long'),
                           *ordered(rstar_synthetic, 'rstar_synthetic_long')], cap=2)

    code_think = cached('code_think')
    code_plain = cached('code_no_think')
    add('code', 170_000_000, code_think)
    add('code', QUOTAS['code'], code_plain, high=4096)
    add('code', QUOTAS['code'], [*code_plain, *code_think])

    general_quota = QUOTAS['general'] - anchors
    t2t = cached('t2t')
    instruction = [*cached('instruction_no_think'), *cached('instruction_think')]
    chinese = [*cached('chinese_no_think'), *cached('chinese_think')]
    knowledge = [*cached('knowledge_no_think'), *cached('knowledge_think')]
    add('general', 550_000_000, t2t, high=4096)
    add('general', 730_000_000, instruction, high=4096)
    add('general', general_quota, [*chinese, *knowledge, *t2t, *instruction], high=4096)
    add('general', general_quota, [*chinese, *knowledge, *t2t, *instruction])

    math_think, math_plain = cached('math_think'), cached('math_no_think')
    add('math', 240_000_000, math_think, high=4096)
    add('math', QUOTAS['math'], math_plain, high=4096)
    add('math', QUOTAS['math'], [*math_think, *math_plain])

    with path.open('x', encoding='utf-8', newline='\n') as output:
        for row in plans:
            print(*row, sep='\t', file=output)


def parse_mix(path):
    summary, by_category = {}, {}
    fields = ('rows', 'tokens', 'supervised_tokens', 'thinking_rows', 'tool_rows', 'max_tokens')
    for line in path.read_text(encoding='utf-8').splitlines():
        key, *values = line.split('\t')
        if key == 'summary':
            summary.update(zip((*fields, 'dedup_rejections', 'budget_rejections'), map(int, values), strict=True))
        elif key == 'category':
            name, *counts = values
            by_category[name] = dict(zip(fields, map(int, counts), strict=True))
        elif key == 'bytes':
            name, size = values
            if name == 'total':
                summary['bytes'] = int(size)
            else:
                by_category[name]['bytes'] = int(size)
    summary['per_category'] = by_category
    return summary


def main_group(name):
    if name.startswith('contest_'):
        return 'contest'
    if name.startswith(('tools', 't2t_mini_full_tools')):
        return 'tools'
    if name.startswith('ultradata_code_'):
        return 'code'
    if name.startswith('ultradata_math_'):
        return 'math'
    return 'general'


def assemble():
    run = WORK / f'assembly_{time.time_ns()}'
    run.mkdir()
    output = run / 'release'
    anchors = run / 'identity.tsv'
    kept = write_pinned(anchors)
    validation = run / 'holdout.tsv'
    write_holdout_index(validation)
    plan = run / 'plan.tsv'
    write_mix_plan(plan, kept['identity_bytes'], run / 'local_tools.tsv')
    mixer = run / 'mixer.exe'
    subprocess.run(['rustc', '--edition=2021', '-O', str(Path(__file__).with_name('mix_instinctv2_sft_2b.rs')),
                    '-o', str(mixer)], check=True)
    summary = run / 'mix.tsv'
    with summary.open('wb') as stdout:
        subprocess.run([str(mixer), '--bytes', str(anchors), str(validation), str(plan),
                        str(BUDGET), str(run / 'lanes'), str(output)], stdout=stdout, check=True)
    train = parse_mix(summary)
    budgets = Counter()
    for name, counts in train['per_category'].items():
        budgets[main_group(name)] += counts['bytes']
    sources = {}
    for path in sorted(WORK.glob('*.report.json')):
        item = orjson.loads(path.read_bytes())
        sources[path.name] = item
    cached_holdouts = Counter()
    for path in (WORK / 'holdout_screened').glob('*.report.json'):
        cached_holdouts.update(orjson.loads(path.read_bytes())['counts'])
    report = {'status': 'assembled', 'target_bytes': BUDGET, 'target_group_bytes': QUOTAS,
              'group_bytes': dict(budgets), 'train': train, 'identity': 'InstinctV2',
              'max_training_context': 16384, 'rendered_token_cap': CAP,
              'cached_screen_report_sha256': base.digest(BASELINE / 'build.report.json'),
              'cached_code_holdout_screen': dict(cached_holdouts),
              'holdouts': {path.stem: {k: v for k, v in orjson.loads(path.read_bytes()).items() if k != 'prompts'}
                           for path in HOLDOUTS.glob('*.json')},
              'sources': sources, 'plan_sha256': base.digest(plan),
              'known_limits': ['CodeContests accepted solutions, OCR2 pass rates and rStar seed pass flags are upstream checks. '
                               'Static, UltraData and rStar synthetic targets were not re-judged locally.',
                               'LiveCodeBench filter covers test6.jsonl. BFCL screening uses exact normalized user prompts.']}
    write_json(output / 'build.report.json', report)
    print(f'[assembled] {train["bytes"]:,} bytes, {train["rows"]:,} rows, {train["tokens"]:,} tokens: {dict(budgets)}', flush=True)
    print(output, flush=True)
    return output


def audit_worker():
    global AUDIT_EXACT, AUDIT_GRAMS
    AUDIT_EXACT, AUDIT_GRAMS = {}, {}
    for name, prompts in holdout_prompts():
        for prompt in prompts:
            text = base.normalize_text(prompt)
            AUDIT_EXACT[text] = name
            if not name.startswith('BFCL'):
                words = text.split()
                AUDIT_GRAMS.update((tuple(words[i:i + 13]), name) for i in range(len(words) - 12))


def filter_cached_code(path):
    out = WORK / 'holdout_screened' / path.name
    report = out.with_suffix('.report.json')
    if out.exists() and report.exists():
        return out.name, 0
    counts = Counter()
    with path.open('rb') as input_file, out.with_suffix('.partial').open('wb', buffering=1 << 20) as output:
        for line in input_file:
            data = line.split(b'\t', 8)[7]
            row = orjson.loads(data)
            counts['input'] += 1
            if name := benchmark_overlap(row, AUDIT_EXACT, AUDIT_GRAMS):
                counts[name] += 1
                continue
            counts['retained'] += 1
            output.write(line)
    out.with_suffix('.partial').replace(out)
    write_json(report, {'source': str(path.relative_to(ROOT)), 'counts': dict(counts), 'sha256': base.digest(out)})
    return out.name, counts['input'] - counts['retained']


def screen_cached_code(workers):
    output = WORK / 'holdout_screened'
    output.mkdir(exist_ok=True)
    jobs = [*CACHED.glob('code_think_*.tsv'), *CACHED.glob('code_no_think_*.tsv'),
            *WORK.glob('*--contest_ultradata_*.tsv')]
    with ProcessPoolExecutor(max_workers=workers, initializer=audit_worker) as pool:
        removed = 0
        for name, count in pool.map(filter_cached_code, jobs):
            removed += count
        print(f'[holdouts] {len(jobs)} cached code files, {removed:,} removed matches', flush=True)


def audit_mix_shard(source, provenance):
    band_rows, band_tokens = Counter(), Counter()
    tools, overlaps = Counter(), Counter()
    quarantine = set()
    contest = Counter()
    start = time.process_time()

    def observe(row, meta):
        length = meta['tokens']
        band = '0-4096' if length <= 4096 else '4097-8192' if length <= 8192 else '8193-16384'
        band_rows[band] += 1
        band_tokens[band] += length
        group = main_group(meta['category'])
        if group == 'contest':
            contest[f'{meta["canonical_problem_sha256"]}/{meta["language"]}'] += 1
        if group == 'tools':
            tools['with_calls' if meta['tool_calls'] else 'without_calls'] += 1
            tools['calls'] += meta['tool_calls']
            tools['multiple_calls'] += int(meta['tool_calls'] > 1)
            tools['parallel_trajectories'] += int(any(len(orjson.loads(turn['tool_calls'])) > 1
                                                      for turn in row['conversations'] if turn.get('tool_calls')))
        grams = AUDIT_GRAMS if group in {'contest', 'code'} else {}
        if benchmark := benchmark_overlap(row, AUDIT_EXACT, grams):
            overlaps[benchmark] += 1
            quarantine.add(meta['first_user_sha256'])

    result = previous.audit_shard(source, provenance, True, observe)
    result.update(jsonl_bytes=source.stat().st_size, context_rows=dict(band_rows), context_tokens=dict(band_tokens),
                  tools=dict(tools), benchmark_overlaps=dict(overlaps), benchmark_quarantine=sorted(quarantine),
                  worker_cpu_seconds=round(time.process_time() - start, 2))
    return result, contest


def verify(output, workers):
    report = orjson.loads((output / 'build.report.json').read_bytes())
    paths = sorted((output / 'train_jsonl').glob('*.jsonl'))
    totals, context_rows, context_tokens, tools, overlaps, contest = (Counter() for _ in range(6))
    records, quarantine = {}, set()
    with ProcessPoolExecutor(max_workers=workers, initializer=audit_worker) as pool:
        jobs = {pool.submit(audit_mix_shard, file, output / 'provenance' / file.name): file for file in paths}
        for future in as_completed(jobs):
            name = jobs[future].with_suffix('.parquet').name
            audited, keys = future.result()
            records[name] = audited
            totals.update({key: audited[key] for key in ('rows', 'tokens', 'supervised_tokens', 'thinking_rows', 'tool_rows')})
            totals['bytes'] += audited['jsonl_bytes']
            context_rows.update(audited['context_rows'])
            context_tokens.update(audited['context_tokens'])
            tools.update(audited['tools'])
            overlaps.update(audited['benchmark_overlaps'])
            quarantine.update(audited.pop('benchmark_quarantine'))
            contest.update(keys)
            print(f'[audit] {name}: {audited["rows"]:,} rows, {audited["tokens"]:,} tokens', flush=True)
    if quarantine:
        path = WORK / 'benchmark_quarantine.txt'
        if path.exists():
            quarantine.update(path.read_text(encoding='utf-8').splitlines())
        path.write_text('\n'.join(sorted(quarantine)) + '\n', encoding='utf-8')
        raise ValueError(f'Holdout matches require rebuilding with quarantined prompts: {dict(overlaps)}')
    if any(report['train'][key] != value for key, value in totals.items()):
        raise ValueError('Final audit totals differ from the mixer totals')
    if max(contest.values(), default=0) > 2:
        raise ValueError('Question-language cap violated')
    if any(abs(report['group_bytes'].get(group, 0) - quota) > 10_000_000
           for group, quota in report['target_group_bytes'].items()):
        raise ValueError(f'One or more groups missed the byte target: {report["group_bytes"]}')
    compiler = ROOT / 'dataset_compiler/target/release/dataset_compiler.exe'
    (output / 'train_parquet').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(previous.compile_and_verify, compiler, file,
                            output / 'train_parquet' / file.with_suffix('.parquet').name,
                            records[file.with_suffix('.parquet').name]['rows'], 4): file for file in paths}
        for future in as_completed(jobs):
            name = jobs[future].with_suffix('.parquet').name
            records[name].update(future.result())
            print(f'[verified] {name}: full JSONL to Parquet comparison passed', flush=True)
    context = previous.smoke(output, records)
    for name in ('model_16k.config.json', 'validation_identity_clean.parquet', 'validation_identity_clean.jsonl',
                 'validation.provenance.jsonl.gz'):
        shutil.copy2(BASELINE / name, output / name)
    write_json(output / 'verification.report.json', {
        'status': 'verified', 'build_report_sha256': base.digest(output / 'build.report.json'),
        'train': dict(totals), 'context_rows': dict(context_rows), 'context_tokens': dict(context_tokens),
        'tools': dict(tools), 'contest': {'questions': len({key.split('/')[0] for key in contest}),
                                         'question_language_pairs': len(contest), 'max_answers_per_pair': max(contest.values())},
        'benchmark_overlaps': dict(overlaps), 'trainer_16k_packing': context,
        'train_shards': dict(sorted(records.items())),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('screen', 'competitive', 'additional', 'official', 'supplement', 'holdouts', 'identity', 'assemble', 'verify'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    if args.action == 'screen':
        screen(args.workers)
    elif args.action == 'competitive':
        screen_competitive(args.workers)
    elif args.action == 'additional':
        screen_additional(args.workers)
    elif args.action in {'official', 'supplement'}:
        official_contest(args.workers, args.action == 'supplement')
    elif args.action == 'holdouts':
        screen_cached_code(args.workers)
    elif args.action == 'identity':
        previous.init_worker()
        t2t_identity()
    elif args.action == 'assemble':
        assemble()
    elif args.release:
        verify(args.release, args.workers)
    else:
        parser.error('--release is required for verification')


if __name__ == '__main__':
    main()
