"""Build the InstinctV2 bilingual SFT release from pinned local sources."""
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import ast
import gzip
import hashlib
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import orjson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_builder.instinctv2_sft_scenarios import compact, identity_rows, tool_episode

DATA = ROOT / 'dataset'
SEED = 20261108
CAP = 2000  # 2048-token training budget, with room for loader augmentation.
OLD_SYSTEM = ('你是 InstinctV1Moe，一个从头训练的语言模型。你没有主观意识或个人经历。'
              '根据对话上下文遵循用户要求；不确定时说明不确定。')
IDENTITY = re.compile(r'instinct(?:\s*v\d+\s*(?:moe)?)?|\bl1bra\b|你是谁|你有(?:主观|自我)意识|你(?:是什么|是哪款|是哪种)模型|你背后的模型|你(?:的开发者|由谁(?:开发|训练))|\bwho are you\b|\bwhat model are you\b|\bwho (?:made|trained|developed) you\b', re.I)
FOREIGN = re.compile(r'\b(?:qwen\w*|chatgpt|openai|deepseek|claude|anthropic|gemini|llama|mistral|alibaba|gpt[- ]?[2345](?:[.\w-]+)?)\b|通义|千问|阿里云|阿里巴巴|文心一言|豆包|智谱', re.I)
MODEL_BIO = re.compile(r'我是(?:一个|一款|由.{0,20}训练的)?(?:大语言模型|人工智能助手)|I am (?:an? |a large )(?:AI assistant|language model)|(?:我的知识|知识更新).{0,8}(?:截止|截至)|my knowledge cutoff', re.I)
BENCH_NAME = re.compile(r'\b(?:human[ _-]?eval(?:plus)?|mbpp|livecodebench|bigcodebench|gsm8k test)\b', re.I)
CONTROLS = re.compile(r'<\|[^|\n]{1,64}\|>|</?(?:think|tools|tool_call|tool_response)>|\ufffd', re.I)
WORDS = re.compile(r'\w+')
MATH_SUFFIX = {
    '请逐步说明，并在最后一行写“#### 数字”。': '只输出最终数字。',
    " Show the steps and end with '#### <number>'.": ' Output only the final number.',
    '给出简短算式，最后一行用 #### 数字。': '只输出最终数字。',
    ' Give a short calculation and finish with #### <number>.': ' Output only the final number.',
}
MATH_SUFFIX.update({'请用总路程除以总时间，逐步说明，并在最后一行写“#### 数字”。': '只输出最终数字。',
                    " Use total distance divided by total time, show the steps, and end with '#### <number>'.": ' Output only the final number.'})
UNITS = {'kg': ('mass', 1), 'g': ('mass', .001), 'mg': ('mass', .000001), 'lb': ('mass', .45359237),
         'oz': ('mass', .028349523125), 'm': ('distance', 1), 'cm': ('distance', .01),
         'mm': ('distance', .001), 'km': ('distance', 1000), 'inch': ('distance', .0254),
         'ft': ('distance', .3048), 'mile': ('distance', 1609.344), 'l': ('volume', 1), 'ml': ('volume', .001)}
ALIASES = {'pounds': 'lb', 'pound': 'lb', 'lbs': 'lb', 'in': 'inch', 'inches': 'inch', 'feet': 'ft', 'miles': 'mile'}
QUOTAS = {
    'train': {'t2t': 8384, 'ultrachat': 8000, 'magicoder': 3008, 'executed_code': 1600,
              'verified_math': 3072, 'verified_logic': 3072, 'instruction': 2048, 'tools': 3328, 'identity': 256},
    'validation': {'t2t': 272, 'ultrachat': 112, 'magicoder': 128, 'verified_math': 112,
                   'verified_logic': 112, 'instruction': 128, 'tools': 128, 'identity': 32},
}


def rows(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as stream:
        yield from map(orjson.loads, stream)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_text(text):
    return ' '.join(WORDS.findall(text.casefold()))


def emit(stream, value):
    stream.write(orjson.dumps(value) + b'\n')


def ordered_by_family(items, rng):
    rng.shuffle(items)
    families = defaultdict(deque)
    for item in items:
        families[item[1].get('family', '')].append(item)
    while families:
        for family, bucket in list(families.items()):
            yield bucket.popleft()
            if not bucket:
                del families[family]


def build_plans():
    base = DATA / 'v1moe_sft64k'
    pilot = ROOT / 'tmp/minicpm5_sft'
    plans, metadata = {}, {}

    def add(path, split, category, record, row):
        plans.setdefault(path, {})[row] = (split, category, record)

    for split in QUOTAS:
        rng = random.Random(f'{SEED}/{split}')
        source = (DATA / 'sft_v1moe_balanced_64k_identity_clean.jsonl' if split == 'train' else
                  base / 'validation_identity_clean.jsonl')
        meta_path = base / f'{split}.provenance.jsonl.gz'
        metadata[source] = meta_path
        groups = defaultdict(list)
        for line, item in enumerate(rows(meta_path), 1):
            key = item['_source']
            if key in {'t2t', 'ultrachat', 'magicoder'} and item['token_count'] <= CAP + 45:
                groups[key].append((line, item))
        for category, candidates in groups.items():
            rng.shuffle(candidates)
            for line, item in candidates[:math.ceil(QUOTAS[split][category] * 1.28)]:
                add(source, split, category, {
                    'group': item['group_sha256'], 'original_path': item['_path'], 'original_row': item['_row'],
                    'repository': item['_repository'], 'revision': item['_revision'],
                }, line)

        for category, folder, sidecar in (
            ('verified_math', 'verified_cot_sft', 'provenance.jsonl.gz'),
            ('verified_logic', 'reasoning_sft_strict', 'verification_records.jsonl.gz'),
            ('instruction', 'instruction_understanding_sft', 'provenance.jsonl.gz'),
        ):
            source = DATA / folder / f'{split}.jsonl'
            meta_path = DATA / folder / sidecar
            metadata[source] = meta_path
            candidates = []
            for item in rows(meta_path):
                if item['file_split'] != split:
                    continue
                family = item.get('family', item.get('kind'))
                group = item.get('group_id', item.get('semantic_hash'))
                candidates.append((item['row'] + 1, {'group': group, 'family': family,
                                                    'language': item['language']}))
            seen_groups = set()
            for line, item in ordered_by_family(candidates, rng):
                if item['group'] in seen_groups:
                    continue
                add(source, split, category, item, line)
                seen_groups.add(item['group'])
                if len(seen_groups) >= math.ceil(QUOTAS[split][category] * 1.17):
                    break

    for phase in ('deep', 'hybrid'):
        path = pilot / f'{phase}_identity_clean.jsonl'
        meta_path = pilot / f'{phase}_provenance.jsonl'
        metadata[path] = meta_path
        for line, item in enumerate(rows(meta_path), 1):
            if item['kind'] in {'kodcode', 'l3'}:
                add(path, 'train', 'executed_code', {
                    'group': item['id'], 'family': item['kind'], 'original_path': item['path'],
                    'original_id': item['id'], 'verification': 'previous WASI execution audit',
                }, line)

    raw = DATA / 'sft_t2t_mini.jsonl'
    rng = random.Random(f'{SEED}/tool_replay')
    for line in rng.sample(range(771075, 905719), 18000):
        add(raw, 'train', 'tool_replay', {'repository': 'local:sft_t2t_mini', 'group': str(line)}, line)
    return plans, metadata


def selected_rows(plans, work):
    binary = work / ('selector.exe' if os.name == 'nt' else 'selector')
    subprocess.run(['rustc', '--edition=2021', '-O', str(Path(__file__).with_name('select_sft_rows.rs')),
                    '-o', str(binary)], check=True)
    command, paths = [str(binary)], []
    for index, (path, selections) in enumerate(plans.items()):
        ids, output = work / f'{index}.ids', work / f'{index}.jsonl'
        ids.write_text(''.join(f'{line}\n' for line in sorted(selections)), encoding='ascii')
        command.extend((str(path), str(ids), str(output)))
        paths.append((path, output, selections))
    subprocess.run(command, check=True)
    for path, output, selections in paths:
        for line, row in zip(sorted(selections), rows(output), strict=True):
            split, category, meta = selections[line]
            yield split, category, row, {**meta, 'path': path.relative_to(ROOT).as_posix(), 'source_row': line}


def normalize_dialogue(row, category):
    turns = []
    for item in row['conversations']:
        turn = {key: item[key] for key in ('role', 'content', 'reasoning_content', 'tools', 'tool_calls')
                if item.get(key) is not None}
        if turn['role'] == 'system' and turn.get('content', '').startswith(OLD_SYSTEM):
            turn['content'] = turn['content'][len(OLD_SYSTEM):].strip()
            if not turn['content'] and 'tools' not in turn:
                continue
        turn.setdefault('content', '')
        if not isinstance(turn['content'], str) or not isinstance(turn.get('reasoning_content', ''), str):
            raise ValueError('non-text message')
        for key in ('tools', 'tool_calls'):
            if key in turn:
                value = orjson.loads(turn[key]) if isinstance(turn[key], str) else turn[key]
                if not value:
                    turn.pop(key)
                    continue
                if key == 'tool_calls':
                    calls = []
                    for entry in value:
                        function = entry.get('function', entry)
                        args = function['arguments']
                        if isinstance(args, str):
                            args = orjson.loads(args)
                        calls.append({'type': 'function', 'function': {'name': function['name'], 'arguments': args}})
                    value = calls
                turn[key] = compact(value)
        turns.append(turn)
    if category not in {'tools', 'tool_replay'} and any(t['role'] == 'tool' or t.get('tools') or t.get('tool_calls') for t in turns):
        raise ValueError('tool dialogue outside tool mix')
    return turns


def verify_dialogue(turns):
    from jsonschema import Draft7Validator

    expected, pending = 'user', 0
    declared = {}
    for index, turn in enumerate(turns):
        role = turn['role']
        if role == 'system' and index == 0:
            for item in orjson.loads(turn.get('tools', '[]')):
                fn = item['function']
                name = fn['name']
                if name in declared or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
                    raise ValueError('invalid tool name')
                declared[name] = fn['parameters']
            continue
        if role != expected or (not turn['content'].strip() and not turn.get('tool_calls')):
            raise ValueError('incomplete message sequence')
        if role == 'user':
            expected = 'assistant'
        elif role == 'tool':
            orjson.loads(turn['content'])
            pending -= 1
            expected = 'tool' if pending else 'assistant'
        elif turn.get('tool_calls'):
            calls = orjson.loads(turn['tool_calls'])
            for call in calls:
                fn = call['function']
                if fn['name'] not in declared or not isinstance(fn['arguments'], dict):
                    raise ValueError('undeclared tool or invalid arguments')
                if not Draft7Validator(declared[fn['name']]).is_valid(fn['arguments']):
                    raise ValueError('arguments fail tool schema')
            pending = len(calls)
            expected = 'tool'
        else:
            expected = 'user'
    if expected != 'user' or not turns or turns[-1]['role'] != 'assistant':
        raise ValueError('unfinished conversation')


def verify_replay(turns):
    """Select only locally checkable T2T single-call outcomes."""
    if [t['role'] for t in turns] != ['system', 'user', 'assistant', 'tool', 'assistant']:
        raise ValueError('not a single-call trace')
    calls = orjson.loads(turns[2].get('tool_calls', '[]'))
    if len(calls) != 1:
        raise ValueError('not a single-call trace')
    fn = calls[0]['function']
    name, args = fn['name'], fn['arguments']
    result = orjson.loads(turns[3]['content'])['result']
    if type(result) not in (int, float) or not math.isfinite(result):
        raise ValueError('no scalar result')
    if name == 'random_number':
        prompt = turns[1]['content']
        if (not re.search(r'random|随机', prompt, re.I) or re.search(r'guess|猜|don.t|不要|不要告诉|别告诉', prompt, re.I)
                or type(result) is not int or not args['min'] <= result <= args['max']):
            raise ValueError('random result inconsistent')
        inputs = [int(n) for n in re.findall(r'(?<!\d)-?\d+(?!\d)', prompt)]
        if len(inputs) != 2 or set(inputs) != {args['min'], args['max']}:
            raise ValueError('random bounds not explicit')
    elif name == 'unit_converter':
        def unit(name):
            name = name.casefold()
            return UNITS[ALIASES.get(name, name)]
        source, factor = unit(args['from_unit'])
        destination, divisor = unit(args['to_unit'])
        if source != destination or not math.isclose(result, args['value'] * factor / divisor, rel_tol=.001, abs_tol=.001):
            raise ValueError('unit conversion inconsistent')
    else:
        raise ValueError('tool result not locally verifiable')
    numbers = [float(number.replace(',', '')) for number in
               re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', turns[-1]['content'])]
    if not any(math.isclose(result, n, rel_tol=.001, abs_tol=.001) for n in numbers):
        raise ValueError('result absent in final answer')
    for turn in turns:
        turn.pop('reasoning_content', None)
    return name


def separate_math_trace(turns, meta, category):
    if category not in {'verified_math', 'verified_logic'} or int(meta['group'][-1], 36) % 3 == 0:
        return False
    question, answer = turns
    derivation, delimiter, final = answer['content'].rpartition('\n#### ')
    if not delimiter or not re.fullmatch(r'-?\d+(?:[./]\d+)?', final.strip()):
        return False
    for old in sorted(MATH_SUFFIX, key=len, reverse=True):
        if question['content'].endswith(old):
            question['content'] = question['content'][:-len(old)] + MATH_SUFFIX[old]
            answer['reasoning_content'] = derivation.strip()
            answer['content'] = final.strip()
            return True
    return False


def init_worker():
    import datasets  # noqa: F401, before torch on Windows
    from transformers import AutoTokenizer
    from scripts.data_loader.lm_dataset import _create_chat_prompt, _generate_sft_labels
    from scripts.data_builder.audit_identity_contamination import classify_row, count_terms
    from scripts.data_builder.build_quality_python_sft import blocks, code_key
    from scripts.data_builder.filter_anomaly_candidates import inspect_row
    global TOKENIZER, RENDER, LABELS, BOS, EOS, CLASSIFY, COUNT_TERMS, INSPECT, GRAMS, CODES, FUNCTIONS, CODE_BLOCKS, CODE_KEY
    TOKENIZER = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    RENDER, LABELS = _create_chat_prompt, _generate_sft_labels
    BOS = TOKENIZER(f'{TOKENIZER.bos_token}assistant\n', add_special_tokens=False).input_ids
    EOS = TOKENIZER(f'{TOKENIZER.eos_token}\n', add_special_tokens=False).input_ids
    CLASSIFY, COUNT_TERMS, INSPECT = classify_row, count_terms, inspect_row
    CODE_BLOCKS, CODE_KEY = blocks, code_key
    GRAMS, CODES, FUNCTIONS = set(), set(), set()
    for path in (DATA / 'humaneval/HumanEval.jsonl.gz', ROOT / 'tmp/HumanEvalPlus-v0.1.10.jsonl.gz',
                 DATA / 'gsm8k/test.jsonl'):
        for item in rows(path):
            text = normalize_text(item.get('prompt', item.get('question', ''))).split()
            GRAMS.update(tuple(text[i:i + 13]) for i in range(len(text) - 12))
            if 'entry_point' in item:
                FUNCTIONS.add(item['entry_point'])
                CODES.add(CODE_KEY(item['prompt'] + item['canonical_solution']))


def screen_batch(items):
    start = time.process_time()
    counters = defaultdict(Counter)
    terms, ready, prompts = Counter(), [], []
    for split, category, row, meta in items:
        stats = counters[split, category]
        stats['input'] += 1
        try:
            turns = normalize_dialogue(row, category)
            verify_dialogue(turns)
        except (ValueError, TypeError, KeyError, AttributeError):
            stats['schema_filtered'] += 1
            continue
        sample = {'conversations': turns}
        text = '\n'.join(t.get(field, '') for t in turns for field in ('content', 'reasoning_content', 'tools', 'tool_calls'))
        COUNT_TERMS(text, terms)
        if CONTROLS.search(text):
            stats['control_token_filtered'] += 1
            continue
        if category != 'identity' and (IDENTITY.search(text) or FOREIGN.search(text) or MODEL_BIO.search(text)
                                       or CLASSIFY(sample) or INSPECT(sample, 'identity')):
            stats['identity_quarantined'] += 1
            continue
        if category == 'tool_replay':
            try:
                meta['family'] = verify_replay(turns)
            except (ValueError, TypeError, KeyError, OverflowError):
                stats['unverified_tool_outcome'] += 1
                continue
        words = normalize_text(text).split()
        if BENCH_NAME.search(text) or any(tuple(words[i:i + 13]) in GRAMS for i in range(len(words) - 12)):
            stats['benchmark_filtered'] += 1
            continue
        if category in {'magicoder', 'executed_code'}:
            try:
                code = '\n'.join(CODE_BLOCKS(turns[-1]['content']))
                tree = ast.parse(code)
                if not code or not tree.body:
                    raise ValueError('missing Python code')
                if CODE_KEY(code) in CODES or any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in FUNCTIONS
                                                for n in ast.walk(tree)):
                    stats['benchmark_filtered'] += 1
                    continue
            except (SyntaxError, ValueError, RecursionError):
                stats['code_filtered'] += 1
                continue
        if separate_math_trace(turns, meta, category):
            stats['converted_to_think'] += 1
        first = next(t['content'] for t in turns if t['role'] == 'user')
        meta['first_user_sha256'] = fingerprint(normalize_text(first))
        meta['user_turns'] = sum(t['role'] == 'user' for t in turns)
        meta['thinking'] = any(t.get('reasoning_content') for t in turns)
        meta['tool_calls'] = sum(len(orjson.loads(t['tool_calls'])) for t in turns if t.get('tool_calls'))
        try:
            prompt = RENDER(TOKENIZER, turns)
        except (TypeError, ValueError):
            stats['template_filtered'] += 1
            continue
        prompts.append(prompt)
        ready.append((split, category, sample, meta))
    accepted = []
    for item, ids in zip(ready, TOKENIZER(prompts, add_special_tokens=False).input_ids if prompts else [], strict=True):
        split, category, _, meta = item
        stats = counters[split, category]
        if not 12 <= len(ids) <= CAP:
            stats['length_filtered'] += 1
            continue
        active = sum(label != -100 for label in LABELS(ids, BOS, EOS, CAP))
        if not active:
            stats['no_supervision'] += 1
            continue
        meta.update(tokens=len(ids), supervised_tokens=active)
        stats['eligible'] += 1
        accepted.append(item)
    return accepted, dict(counters), terms, os.getpid(), time.process_time() - start


def batches(items, size):
    pending = []
    for item in items:
        pending.append(item)
        if len(pending) == size:
            yield pending
            pending = []
    if pending:
        yield pending


def select_release(pools):
    selected = {}
    prompts, all_groups = set(), set()
    for split in ('validation', 'train'):
        group_in_split = set()
        output = []
        rng = random.Random(f'{SEED}/{split}/output')
        for category, limit in QUOTAS[split].items():
            candidates = pools[split, category]
            if category == 'tools' and split == 'train':
                replay = list(ordered_by_family(pools['train', 'tool_replay'], rng))[:512]
                candidates = replay + list(ordered_by_family(candidates, rng))
            else:
                candidates = list(ordered_by_family(candidates, rng))
            count = 0
            for sample, meta in candidates:
                prompt = meta['first_user_sha256']
                group = (category, meta['group'])
                if prompt in prompts or group in all_groups:
                    continue
                prompts.add(prompt)
                group_in_split.add(group)
                output.append((category, sample, meta))
                count += 1
                if count == limit:
                    break
            if count < limit:
                raise RuntimeError(f'{split}/{category}: selected {count}/{limit} from {len(candidates)} eligible')
        all_groups.update(group_in_split)
        rng.shuffle(output)
        selected[split] = output
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA / 'instinctv2_sft_v1')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # outer worker pool provides parallelism
    plans, metadata = build_plans()
    pools = defaultdict(list)
    counts = defaultdict(Counter)
    terms, cpu_seconds = Counter(), Counter()

    def inputs(work):
        yield from selected_rows(plans, work)
        for split in QUOTAS:
            for row, meta in identity_rows(split):
                yield split, 'identity', row, meta
            for index in range(5200 if split == 'train' else 320):
                row, meta = tool_episode(index, split)
                yield split, 'tools', row, meta

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='instinctv2-sft-', dir=ROOT / 'tmp') as temporary:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
            # Bound the number of materialized chunks passed to Windows spawned workers.
            for window in batches(batches(inputs(Path(temporary)), 192), 2 * args.workers):
                for found, screened, hits, pid, elapsed in pool.map(screen_batch, window):
                    for split, category, row, meta in found:
                        pools[split, category].append((row, meta))
                    for key, value in screened.items():
                        counts[key].update(value)
                    terms.update(hits)
                    cpu_seconds[pid] += elapsed
                total = sum(counter['input'] for counter in counts.values())
                print(f'[screen] {total} candidates processed', flush=True)

    selected = select_release(pools)
    upstream = {item['source']: item for item in orjson.loads((DATA / 'v1moe_sft64k/sources.json').read_bytes())}
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(plans) + list(dict.fromkeys(metadata.values()))
        hashes = dict(zip(paths, pool.map(digest, paths), strict=True))
    report = {
        'status': 'jsonl_built', 'identity': 'InstinctV2', 'seed': SEED, 'training_max_seq_len': 2048,
        'rendered_token_cap': CAP, 'identity_keyword_hits': dict(terms),
        'screen_counts': {f'{split}/{category}': dict(value) for (split, category), value in counts.items()},
        'worker_cpu_seconds': {str(pid): elapsed for pid, elapsed in cpu_seconds.items()},
        'build_wall_seconds_before_publication': time.perf_counter() - start,
        'repo_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'builder_sha256': {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name('instinctv2_sft_scenarios.py'),
                                                    Path(__file__).with_name('select_sft_rows.rs'))},
        'tokenizer_sha256': {p.name: digest(p) for p in (ROOT / 'model/tokenizer.json', ROOT / 'model/tokenizer_config.json')},
        'source_files': {p.relative_to(ROOT).as_posix(): {'sha256': hashes[p], 'planned_rows': len(plans[p]),
                         'provenance_sha256': hashes.get(metadata.get(p))} for p in plans},
        'upstream': {k: upstream[k] for k in ('t2t', 'ultrachat', 'magicoder')},
        'pilot_upstream': orjson.loads((ROOT / 'tmp/minicpm5_sft/corpus_report.json').read_bytes())['sources'],
        'audits': {'identity': 'legacy system identity removed at its exact known boundary; ambiguous source rows quarantined',
                   'benchmarks': 'HumanEval, HumanEval+ and GSM8K test: named benchmark, shared 13-word spans, Python code AST and entry points',
                   'split': 'upstream validation splits; first user text deduplicated globally; authored tools hold out services',
                   'tools': 'mock outcomes are deterministic; T2T replay restricted to checkable random bounds and unit conversions',
                   'licenses': 'Mixed terms. KodCode subset is CC BY-NC 4.0; T2T upstream license chain is incomplete.'},
        'splits': {},
    }
    for split, values in selected.items():
        path = args.output / f'{split}_identity_clean.jsonl'
        provenance = args.output / f'{split}.provenance.jsonl.gz'
        by_category, token_category, by_family = Counter(), Counter(), Counter()
        with path.open('xb') as corpus, provenance.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as origin:
            for index, (category, row, meta) in enumerate(values, 1):
                emit(corpus, row)
                emit(origin, {'row': index, 'category': category, **meta})
                by_category[category] += 1
                token_category[category] += meta['tokens']
                if 'family' in meta:
                    by_family[f'{category}/{meta["family"]}'] += 1
        report['splits'][split] = {
            'rows': len(values), 'tokens': sum(token_category.values()), 'source_rows': dict(by_category),
            'source_tokens': dict(token_category), 'families': dict(by_family),
            'thinking_rows': sum(meta['thinking'] for _, _, meta in values),
            'multi_turn_rows': sum(meta['user_turns'] > 1 for _, _, meta in values),
            'tool_calls': sum(meta['tool_calls'] for _, _, meta in values),
            't2t_tool_rows': sum(meta.get('path') == 'dataset/sft_t2t_mini.jsonl' for _, _, meta in values),
            'max_tokens': max(meta['tokens'] for _, _, meta in values),
            'supervised_tokens': sum(meta['supervised_tokens'] for _, _, meta in values),
            'sha256': digest(path), 'provenance_sha256': digest(provenance),
        }
    (args.output / 'build.report.json').write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(report['splits'], option=orjson.OPT_INDENT_2).decode(), flush=True)


if __name__ == '__main__':
    main()
