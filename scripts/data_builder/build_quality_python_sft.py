"""Build a continuation-SFT mix from pinned sources and the actual prior corpus.

No downloaded code/tests are executed. Static validation is distinct from the
upstream execution-verification claim. Original datasets are never modified.
"""
import ast
import builtins
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import symtable
import sys
import warnings

import orjson
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dataset' / 'quality_python_sft'
MAX_TOKENS = 3968  # reserve room for the trainer's optional system message
SEED = 20260915
warnings.simplefilter('ignore', SyntaxWarning)


def norm(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def fingerprint(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()


def code_key(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = [v for v in node.body if not (isinstance(v, ast.Expr) and isinstance(v.value, ast.Constant) and isinstance(v.value.value, str))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def blocks(text):
    return re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', text, re.S)


def static_check(code):
    try:
        tree = ast.parse(code)
        compile(tree, '<candidate>', 'exec')
    except (SyntaxError, ValueError, TypeError):
        return 'syntax', None
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not 1 <= len(functions) <= 4 or any(isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)):
        return 'not_short_functions', None
    if len({n.name for n in functions}) != len(functions):
        return 'redefined_function', None
    if len(code.splitlines()) > 100 or len(code) > 6000:
        return 'long_code', None
    if any(isinstance(n, (ast.Pass, ast.Global, ast.Nonlocal)) or isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):
        return 'placeholder_or_global_state', None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
                continue
            except (ValueError, TypeError):
                pass
        return 'top_level_example_or_execution', None
    for node in ast.walk(tree):
        modules = [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else ([node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
        if any(module not in sys.stdlib_module_names or module in {'os', 'sys', 'subprocess', 'socket', 'http', 'urllib', 'shutil', 'pathlib'} for module in modules):
            return 'non_foundational_dependency', None
        if isinstance(node, ast.ImportFrom) and (node.level or any(alias.name == '*' for alias in node.names)):
            return 'relative_or_star_import', None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'eval', 'exec', 'open', 'input', '__import__'}:
            return 'dynamic_or_io_code', None
    table = symtable.symtable(code, '<candidate>', 'exec')
    available = set(dir(builtins)) | {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
    def unresolved(scope):
        return any(s.is_referenced() and s.is_global() and s.get_name() not in available for s in scope.get_symbols()) or any(unresolved(child) for child in scope.get_children())
    if unresolved(table):
        return 'unresolved_global', None
    return None, tree


def tags_for(prompt, code):
    text = (prompt + '\n' + code).lower()
    rules = {
        'state_and_loops': r'prefix sum|cumulative|running|rolling|balance|parenthes|stack|while |for ',
        'types_and_containers': r'dictionary|\bdict\b|\blist\b|tuple|isinstance|counter|frequency',
        'numeric_basics': r'median|mean|average|derivative|polynomial|modulo|digit|prime|divisor|factorial|round|base |integer|sum',
        'strings_and_sequences': r'string|substring|prefix|suffix|palindrome|sort|split|slice|character',
        'contracts_and_boundaries': r'empty|negative|duplicate|unique|valid|invalid|at least|at most|otherwise|raise |if ',
    }
    return [name for name, pattern in rules.items() if re.search(pattern, text)] or ['other_short_function']


def category(messages):
    text = '\n'.join(str(m.get('content', '')) for m in messages)
    if any(m.get('tool_calls') or m.get('tools') or m.get('role') == 'tool' for m in messages):
        return 'tools'
    if re.search(r'```|\bdef \w+\(|\bclass \w+|#include|\bSELECT\b', text):
        return 'code_heuristic'
    if re.search(r'[一-鿿]', text):
        return 'chinese_general'
    return 'other_general'


def prior_profile():
    rng = random.Random(SEED)
    profiles = {}
    old_prompts = set()
    old_answers = set()
    replay = []
    for filename, field, epochs in [('pretrain_codespecialist.jsonl', 'text', 2), ('sft_t2t_mini.jsonl', 'conversations', 1)]:
        path = ROOT / 'dataset' / filename
        digest = hashlib.sha256(); counts = Counter(); total = 0
        for line in path.open('rb'):
            digest.update(line)
            record = orjson.loads(line); total += 1
            if field == 'text':
                text = record.get('text', '')
                if re.search(r'\bdef \w+\(', text): counts['contains_python_def'] += 1
                if re.search(r'[一-鿿]', text): counts['contains_chinese'] += 1
                if '```' in text: counts['contains_fenced_code'] += 1
            else:
                messages = record[field]
                kind = category(messages); counts[kind] += 1
                counts['with_reasoning_content'] += any(m.get('reasoning_content') for m in messages)
                for m in messages:
                    if m.get('role') == 'user': old_prompts.add(fingerprint(str(m.get('content', ''))))
                    if m.get('role') == 'assistant': old_answers.add(fingerprint(str(m.get('content', ''))))
                # Reservoir of general/dialogue replay, not the old weak code mixture.
                if kind != 'code_heuristic' and len(line) < 45000:
                    counts['replay_eligible'] += 1
                    item = {'conversations': messages, 'old_row': total - 1, 'category': kind}
                    if len(replay) < 40000: replay.append(item)
                    else:
                        index = rng.randrange(counts['replay_eligible'])
                        if index < len(replay): replay[index] = item
            if total % 500000 == 0: print(f'Profile {filename}: {total:,} rows', flush=True)
        profiles[filename] = {'rows': total, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest(),
                              'epochs_user_confirmed': epochs, 'heuristic_counts': dict(counts)}
    (OUT / 'prior_training_profile.json').write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Prior profile complete', profiles, flush=True)
    return profiles, old_prompts, old_answers, replay


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    profiles, old_prompts, old_answers, replay = prior_profile()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    benchmark_grams = set(); benchmark_code = set()
    with gzip.open(ROOT / 'dataset/humaneval/HumanEval.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); words = norm(r['prompt']).split()
            benchmark_grams.update(tuple(words[i:i+13]) for i in range(len(words)-12))
            benchmark_code.add(code_key(r['prompt'] + r['canonical_solution']))
    def contaminated(prompt, code=None):
        if re.search(r'human.?eval|mbpp|livecodebench|bigcodebench', prompt, re.I): return True
        words = norm(prompt).split()
        if any(tuple(words[i:i+13]) in benchmark_grams for i in range(len(words)-12)): return True
        return code is not None and code_key(code) in benchmark_code
    seen_prompt = set(); seen_code = set(); candidates = []; rejects = defaultdict(Counter)
    sources = [
        ('kodcode', sorted((OUT / 'raw/kodcode_sft_4o/data').glob('train-*.parquet'))),
        ('bigcode', [ROOT / 'dataset/bigcode-self-oss-instruct-50k/data/train-00000-of-00001.parquet']),
    ]
    for source, paths in sources:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for r in batch.to_pylist():
                    rejects[source]['scanned'] += 1
                    if source == 'kodcode':
                        if r['subset'] not in {'Prefill', 'Algorithm', 'Data_Structure', 'Docs'}:
                            rejects[source]['other_subset'] += 1; continue
                        if r['gpt_difficulty'] != 'easy' or str(r['4o_correctness']).lower() != 'true' or r['4o_pass_trial_num'] < 2:
                            rejects[source]['insufficient_upstream_verification_or_difficulty'] += 1; continue
                        prompt, code = r['question'].strip(), r['4o_solution'].strip()
                        source_id, subset = r['question_id'], r['subset']
                        test = r['test_code']
                        try:
                            test_tree = ast.parse(test)
                            required = {a.name for n in ast.walk(test_tree) if isinstance(n, ast.ImportFrom) and n.module == 'solution' for a in n.names}
                            if len([n for n in ast.walk(test_tree) if isinstance(n, ast.Assert)]) < 3:
                                rejects[source]['few_test_assertions'] += 1; continue
                        except (SyntaxError, TypeError):
                            rejects[source]['invalid_test_syntax'] += 1; continue
                    else:
                        prompt = r['instruction'].strip()
                        fenced = blocks(r['response'])
                        if len(fenced) != 1:
                            rejects[source]['ambiguous_code_blocks'] += 1; continue
                        code = fenced[0].strip(); source_id = str(r['id']); subset = 'exec_filter'; test = None; required = set()
                    if not 30 <= len(prompt) <= 5500:
                        rejects[source]['prompt_length'] += 1; continue
                    reason, tree = static_check(code)
                    if reason:
                        rejects[source][reason] += 1; continue
                    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
                    if not required <= defined:
                        rejects[source]['test_function_name_mismatch'] += 1; continue
                    if contaminated(prompt, code):
                        rejects[source]['benchmark_overlap_heuristic'] += 1; continue
                    ph, ch = fingerprint(prompt), code_key(code)
                    if ph in old_prompts or fingerprint(code) in old_answers:
                        rejects[source]['prior_sft_exact_overlap'] += 1; continue
                    if ph in seen_prompt or ch in seen_code:
                        rejects[source]['cross_source_duplicate'] += 1; continue
                    messages = [{'role': 'user', 'content': prompt + '\n\nReturn a complete Python solution in one code block.'},
                                {'role': 'assistant', 'content': '```python\n' + code + '\n```'}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    count = len(tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids)
                    if count > MAX_TOKENS:
                        rejects[source]['overlength'] += 1; continue
                    seen_prompt.add(ph); seen_code.add(ch)
                    candidates.append({'conversations': messages, 'source': source, 'source_id': source_id,
                        'subset': subset, 'prompt_hash': ph, 'code_hash': ch, 'tokens': count,
                        'tags': tags_for(prompt, code), 'test_code': test,
                        'verification': 'upstream execution-verified; local static checks only',
                        'license': 'CC-BY-NC-4.0' if source == 'kodcode' else 'ODC-BY'})
            print(f'Filtered {path.name}: candidates={len(candidates):,}', flush=True)
    # Check exact normalized pretrain records/fenced implementations against candidates.
    lookup = defaultdict(set)
    for i, r in enumerate(candidates):
        lookup[fingerprint(r['conversations'][0]['content'].rsplit('\n\nReturn a complete', 1)[0])].add(i)
        lookup[fingerprint(r['conversations'][1]['content'][10:-4])].add(i)
    prior_hits = set()
    for line in (ROOT / 'dataset/pretrain_codespecialist.jsonl').open('rb'):
        text = orjson.loads(line).get('text', '')
        for value in [text, *blocks(text)]: prior_hits.update(lookup.get(fingerprint(value), ()))
    candidates = [r for i, r in enumerate(candidates) if i not in prior_hits]
    print(f'After exact pretrain overlap removal: {len(candidates):,}; removed={len(prior_hits)}', flush=True)
    # Validation consists only of new problems. Split by normalized prompt hash.
    train, validation = [], []
    for row in candidates:
        (validation if int(row['prompt_hash'][:8], 16) % 20 == 0 else train).append(row)
    val_prompts = {r['prompt_hash'] for r in validation}
    rng = random.Random(SEED); rng.shuffle(train); rng.shuffle(validation); rng.shuffle(replay)
    def write_records(path, records):
        with path.open('wb') as stream:
            for row in records:
                stream.write(orjson.dumps({'conversations': row['conversations']}) + b'\n')
    for source in ('kodcode', 'bigcode'):
        write_records(OUT / f'{source}_train.jsonl', [r for r in train if r['source'] == source])
        write_records(OUT / f'{source}_validation.jsonl', [r for r in validation if r['source'] == source])
    # Experimental continuation mix: 75% newly selected code / 25% prior SFT replay by rendered tokens.
    new_tokens = sum(r['tokens'] for r in train); replay_target = new_tokens / 3
    selected_replay = []; replay_tokens = 0; replay_rejects = Counter(); replay_seen = set()
    for r in replay:
        messages = r['conversations']
        text = '\n'.join(str(m.get('content', '')) for m in messages)
        prompt_keys = {fingerprint(str(m.get('content', ''))) for m in messages if m.get('role') == 'user'}
        h = hashlib.sha256(orjson.dumps(messages)).hexdigest()
        if prompt_keys & (val_prompts | seen_prompt) or contaminated(text) or h in replay_seen:
            replay_rejects['duplicate_or_benchmark'] += 1; continue
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            n = len(tokenizer.backend_tokenizer.encode(rendered, add_special_tokens=False).ids)
        except Exception:
            replay_rejects['template_error'] += 1; continue
        if n > MAX_TOKENS:
            replay_rejects['overlength'] += 1; continue
        r.update(source='prior_sft_replay', tokens=n)
        selected_replay.append(r); replay_tokens += n; replay_seen.add(h)
        if replay_tokens >= replay_target: break
    mixed = [*train, *selected_replay]; rng.shuffle(mixed)
    write_records(OUT / 'continue_sft_train.jsonl', mixed)
    write_records(OUT / 'continue_sft_validation.jsonl', validation)
    with gzip.open(OUT / 'provenance_and_tests.jsonl.gz', 'wt', encoding='utf-8') as f:
        for split, records in [('train', mixed), ('validation', validation)]:
            for index, row in enumerate(records):
                f.write(json.dumps({'split': split, 'row': index, **{k:v for k,v in row.items() if k != 'conversations'}}, ensure_ascii=False) + '\n')
    files = {}
    for path in OUT.glob('*.jsonl'):
        digest = hashlib.sha256(); count = 0
        with path.open('rb') as f:
            for line in f:
                digest.update(line); record = orjson.loads(line); assert isinstance(record['conversations'], list); count += 1
        files[path.name] = {'rows': count, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    report = {'prior_training': profiles, 'filter_counts': {k:dict(v) for k,v in rejects.items()},
        'prior_pretrain_exact_hits_removed': len(prior_hits), 'selected_new_train': len(train),
        'new_train_tokens': new_tokens, 'replay_rows': len(selected_replay), 'replay_tokens': replay_tokens,
        'replay_token_fraction': replay_tokens / (new_tokens + replay_tokens),
        'replay_rejects': dict(replay_rejects), 'train_source_rows': dict(Counter(r['source'] for r in mixed)),
        'train_source_tokens': dict(Counter({s:sum(r['tokens'] for r in mixed if r['source']==s) for s in {r['source'] for r in mixed}})),
        'selected_subset_rows': dict(Counter(r['subset'] for r in candidates)),
        'selected_tags_nonexclusive': dict(Counter(tag for r in candidates for tag in r['tags'])),
        'token_limit': MAX_TOKENS, 'split_seed': SEED, 'files': files,
        'limitations': ['Upstream tests are not a correctness proof; no downloaded code executed locally.',
            'Prior overlap detection: normalized exact user prompts/assistant text and pretrain records/fenced code, not semantic overlap.',
            'HumanEval screening: explicit benchmark names, shared 13-word spans, exact AST reference implementations; not proof of no contamination.',
            'Mix and KodCode outputs contain CC-BY-NC-4.0 data: non-commercial; BigCode has separate ODC-BY outputs.',
            '75/25 rendered-token mix is an experimental starting point, not an empirically optimized ratio.',
            'Validation is from newly sourced prompts and is separate from the current model evaluation benchmarks.']}
    (OUT / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('selected_new_train','new_train_tokens','replay_rows','replay_tokens','replay_token_fraction','train_source_rows','files')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
