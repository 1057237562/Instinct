"""Convert sampled OpenBMB L3 Python tasks into execution-checked SFT candidates."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
import sys
import time

import orjson

import build_instinctv2_sft as base
from scripts.data_builder.build_quality_python_sft import blocks, code_key, static_check

sys.path.insert(0, str(base.ROOT / 'tmp'))
from minicpm5_sft_build import CODE_PROMPT, messages, prepare_test, sandbox

REPO = 'openbmb/UltraData-Code'
REVISION = '525c9ac8639c5032633392d096e54db8c84a005e'
BASE_DIR = base.DATA / 'instinctv2_sft_v1'


def exclude_keys():
    prompts, codes, uuids = set(), set(), set()
    for split in ('train', 'validation'):
        for row in base.rows(BASE_DIR / f'{split}_identity_clean.jsonl'):
            turns = row['conversations']
            question = next(turn['content'] for turn in turns if turn['role'] == 'user')
            prompts.add(base.normalize_text(question.removeprefix(CODE_PROMPT).lstrip()))
            for code in blocks(turns[-1]['content']):
                try:
                    codes.add(code_key(code))
                except (ValueError, SyntaxError, RecursionError):
                    continue
        for meta in base.rows(BASE_DIR / f'{split}.provenance.jsonl.gz'):
            if meta.get('family') == 'l3':
                uuids.add(meta['original_id'])
    return prompts, codes, uuids


def prepare(record, counts, exclusions):
    counts['input'] += 1
    fields = ('uuid', 'task', 'analysis', 'solution', 'test')
    if not all(isinstance(record[key], str) and record[key].strip() for key in fields):
        counts['schema_filtered'] += 1
        return None
    uid, question, reasoning, code, test = (record[key].strip() for key in fields)
    key = base.normalize_text(question)
    prompts, codes, uuids = exclusions
    if key in prompts or uid in uuids:
        counts['duplicate_prompt_or_id'] += 1
        return None
    if len(question) + len(code) > 8200 or len(test) > 11000:
        counts['character_filtered'] += 1
        return None
    if base.CONTROLS.search(test) or base.FOREIGN.search(test) or base.BENCH_NAME.search(test):
        counts['test_metadata_filtered'] += 1
        return None
    reason, _ = static_check(code)
    if reason:
        counts[f'code_{reason}'] += 1
        return None
    try:
        fingerprint = code_key(code)
        if fingerprint in codes:
            counts['duplicate_code'] += 1
            return None
        solution_names = {node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
        tree = ast.parse(test)
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in solution_names
               for node in ast.walk(tree)):
            raise ValueError('test overrides solution')
        program = prepare_test(code, test)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        counts['test_schema_filtered'] += 1
        return None
    # Stable mix of solution-only and separate, reviewed-format analysis targets.
    use_reasoning = int(base.fingerprint(uid)[:8], 16) % 10 < 6 and len(reasoning) <= 4000
    row = messages(CODE_PROMPT + '\n\n' + question, f'```python\n{code}\n```', reasoning if use_reasoning else '')
    meta = {'group': uid, 'source_id': uid, 'path': Path(record['source']).relative_to(base.ROOT).as_posix(),
            'family': 'python', 'repository': REPO, 'revision': REVISION, 'verification': 'paired assertions executed in WASI'}
    return row, meta, program + '\nprint("\\n__SFT_TEST_DONE__")', key, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=10000000)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base.init_worker()
    if not sandbox('print("\\n__SFT_TEST_DONE__")'):
        raise RuntimeError('WASI completion probe failed')
    parts = sorted(args.samples.glob('part-*.jsonl'))
    if not parts:
        raise ValueError('No sampled L3 shards')
    counts, screening, terms = Counter(), Counter(), Counter()
    seen = exclude_keys()
    source_counts = Counter()
    start = time.perf_counter()
    tokens = 0
    candidates = (record for stripe in zip_longest(*(base.rows(path) for path in parts))
                  for record in stripe if record is not None)
    output = args.output / 'code_identity_clean.jsonl'
    provenance = args.output / 'code.provenance.jsonl'
    with output.open('xb') as data, provenance.open('xb') as origin, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in base.batches(candidates, 256):
            prepared = [value for item in batch if (value := prepare(item, counts, seen)) is not None]
            if not prepared:
                continue
            checked, filters, hits, _, _ = base.screen_batch(('train', 'executed_code', row, meta)
                                                            for row, meta, _, _, _ in prepared)
            screening.update(filters['train', 'executed_code'])
            terms.update(hits)
            eligible = {meta['source_id']: (row, meta) for _, _, row, meta in checked}
            work = [(eligible[meta['source_id']], program, key, fingerprint)
                    for _, meta, program, key, fingerprint in prepared if meta['source_id'] in eligible]
            for ((row, meta), _, key, fingerprint), passed in zip(work, pool.map(sandbox, (p for _, p, _, _ in work)), strict=True):
                counts['wasi_executed'] += 1
                if not passed:
                    counts['execution_failed'] += 1
                    continue
                if key in seen[0] or fingerprint in seen[1] or meta['source_id'] in seen[2]:
                    counts['duplicate_in_batch'] += 1
                    continue
                seen[0].add(key)
                seen[1].add(fingerprint)
                seen[2].add(meta['source_id'])
                counts['accepted'] += 1
                meta['row'] = counts['accepted']
                tokens += meta['tokens']
                source_counts[meta['path']] += 1
                base.emit(data, row)
                base.emit(origin, meta)
            if counts['input'] % 2048 == 0 or tokens >= args.max_tokens:
                print(f'[L3] screened={counts["input"]} passed={counts["accepted"]} tokens={tokens}', flush=True)
                data.flush()
                origin.flush()
            if tokens >= args.max_tokens:
                break
    result = {'repository': REPO, 'revision': REVISION, 'license': 'Apache-2.0', 'tokens': tokens,
              'target_tokens': args.max_tokens, 'counts': dict(counts), 'screen_counts': dict(screening),
              'source_rows': dict(source_counts), 'identity_keyword_hits': dict(terms),
              'base_train_sha256': base.digest(BASE_DIR / 'train_identity_clean.jsonl'),
              'base_validation_sha256': base.digest(BASE_DIR / 'validation_identity_clean.jsonl'),
              'code_sha256': base.digest(output), 'provenance_sha256': base.digest(provenance),
              'elapsed_seconds': time.perf_counter() - start, 'sandbox_workers': args.workers}
    (args.output / 'code.report.json').write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(result).decode(), flush=True)


if __name__ == '__main__':
    main()
