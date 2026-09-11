"""Build a reproducible 50:30:20 continued-SFT mix from local files.

Competitive programming mixes complete reasoning with final-answer-only rows,
one sample per question/language. Explicit held-out problem
IDs are excluded across languages. T2T is intentional replay; UltraChat
uses only train_sft and excludes previously trained conversations/prompts.
Run from the repository root; no training process is started.
"""

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import random
import sys

import pyarrow.parquet as pq
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mix_sft_datasets import iter_jsonl


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def fingerprints(row):
    messages = row['conversations']
    plain = [{"role": m['role'], "content": m.get('content', '').strip()}
             for m in messages]
    prompt = next((m['content'] for m in plain if m['role'] == 'user'), '')
    return digest(plain), digest(prompt)


class LengthChecker:
    """Render the local HF chat template without importing torch/transformers."""
    def __init__(self, model_dir):
        config = json.loads((model_dir / 'tokenizer_config.json').read_text('utf-8'))
        self.tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
        self.template = env.from_string(config['chat_template'])

    def render(self, row):
        messages = [dict(m) for m in row['conversations']]
        tools = None
        for m in messages:
            if m.get('role') == 'system' and m.get('tools'):
                tools = json.loads(m['tools']) if isinstance(m['tools'], str) else m['tools']
            if isinstance(m.get('tool_calls'), str):
                m['tool_calls'] = json.loads(m['tool_calls'])
        return self.template.render(messages=messages, tools=tools,
                                    add_generation_prompt=False)

    def count(self, row):
        return len(self.tokenizer.encode(self.render(row), add_special_tokens=True).ids)

    def count_batch(self, rows):
        return [len(x.ids) for x in self.tokenizer.encode_batch(
            [self.render(row) for row in rows], add_special_tokens=True)]


def valid(row):
    messages = row.get('conversations')
    return (isinstance(messages, list) and bool(messages)
            and any(m.get('role') == 'user' for m in messages)
            and messages[-1].get('role') == 'assistant'
            and all(m.get('role') in {'user', 'assistant', 'system', 'tool'}
                    and isinstance(m.get('content'), str)
                    and (m['content'].strip() or m.get('tool_calls')) for m in messages))


class Sample:
    """Uniform random-priority sample among unique, in-limit rows."""
    def __init__(self, size, seed, checker, limit):
        self.size, self.rng, self.checker, self.limit = size, random.Random(seed), checker, limit
        self.heap = []
        self.seen = set()
        self.stats = Counter()

    def add(self, row, provenance):
        if not valid(row):
            self.stats['invalid'] += 1
            return
        fp, _ = fingerprints(row)
        if fp in self.seen:
            self.stats['duplicate'] += 1
            return
        self.seen.add(fp)
        self.stats['unique_candidates'] += 1
        priority = self.rng.getrandbits(128)
        if len(self.heap) == self.size and priority >= -self.heap[0][0]:
            return
        tokens = self.checker.count(row)
        self.stats['length_checked'] += 1
        if tokens > self.limit:
            self.stats['over_limit_among_checked'] += 1
            return
        entry = (-priority, fp, row, tokens, provenance)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, entry)
        else:
            heapq.heapreplace(self.heap, entry)

    def take(self, count):
        if len(self.heap) < count:
            raise ValueError(f'Only {len(self.heap)} eligible rows for requested {count}')
        return sorted(self.heap, reverse=True)[:count]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--total', type=int, default=20000)
    p.add_argument('--competitive-sample-rate', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--output', type=Path, default=ROOT / 'dataset/sft_competitive_t2t_ultrachat_50_30_20_16k.jsonl')
    args = p.parse_args()
    if args.total < 10 or args.total % 10 or args.max_tokens < 1:
        p.error('--total must be a positive multiple of 10; --max-tokens must be positive')
    if not 0 < args.competitive_sample_rate <= 1:
        p.error('--competitive-sample-rate must be in (0, 1]')
    output = args.output.resolve()
    report_path = output.with_suffix('.report.json')
    provenance_path = output.with_suffix('.sources.jsonl')
    if any(x.exists() for x in (output, report_path, provenance_path)):
        raise FileExistsError('Output/report/provenance already exists; choose another --output')
    output.parent.mkdir(parents=True, exist_ok=True)
    checker = LengthChecker(ROOT / 'model')
    previous = ROOT / 'dataset/sft_magicoder110k_mathinstruct_ultrachat20.jsonl'
    t2t = ROOT / 'dataset/sft_t2t_mini.jsonl'
    known, known_prompts = set(), set()
    print('Indexing previously trained mixture', flush=True)
    for row in iter_jsonl(previous):
        fp, prompt = fingerprints(row)
        known.add(fp)
        known_prompts.add(prompt)
    prior_count = len(known)
    print(f'Previous unique conversations: {prior_count:,}; sampling T2T replay', flush=True)
    replay = Sample(args.total * 3 // 10, args.seed, checker, args.max_tokens)
    for i, row in enumerate(iter_jsonl(t2t), 1):
        fp, prompt = fingerprints(row)
        known.add(fp)
        known_prompts.add(prompt)
        replay.add(row, {'source': 't2t_replay', 'file': str(t2t), 'row': i})
        if i % 100000 == 0:
            print(f'T2T scanned {i:,}', flush=True)

    print('Randomly sampling competitive coding; collecting final answers and complete reasoning', flush=True)
    best, best_cot, heldout = {}, {}, set()
    scan_rng = random.Random(args.seed + 4)
    code_stats = Counter()
    inputs = [previous, t2t]
    shards = sorted((ROOT / 'dataset/competitive-coding/data').glob('competitive_programming_*.jsonl'))
    if not shards:
        raise FileNotFoundError('Missing competitive programming shards')
    for path in shards:
        inputs.append(path)
        lang = 'python' if 'python' in path.name else 'cpp'
        def sampled_rows():
            # Stream every line but only parse/tokenize a fixed random fraction.
            with path.open('rb') as f:
                for line_number, line in enumerate(f, 1):
                    code_stats['raw_rows'] += 1
                    if line_number % 50000 == 0:
                        print(f'{path.name}: scanned {line_number:,}; candidates {len(best):,}', flush=True)
                    if scan_rng.random() < args.competitive_sample_rate:
                        yield line_number, json.loads(line)
        for i, raw in sampled_rows():
            code_stats['randomly_sampled_rows'] += 1
            qid = raw.get('question_id') or digest([raw.get('dataset'), raw.get('index')])
            split = raw.get('split')
            splits = split if isinstance(split, list) else [split]
            if not splits or any(s != 'train' for s in splits):
                heldout.add(qid)
                code_stats['non_train_rows'] += 1
                continue
            row = {'conversations': [{'role': m['role'], 'content': m['content'].strip()}
                                     for m in raw['messages']]}
            if not valid(row):
                code_stats['invalid'] += 1
                continue
            fp, prompt = fingerprints(row)
            if fp in known or prompt in known_prompts:
                code_stats['previously_trained'] += 1
                continue
            key = (qid, lang)
            chars = sum(len(m['content']) for m in row['conversations'])
            provenance = {'source': 'competitive', 'language': lang, 'file': str(path),
                          'row': i, 'question_id': qid, 'uuid': raw.get('uuid'),
                          'dataset': raw.get('dataset'), 'split': split,
                          'difficulty': raw.get('difficulty'), 'license': raw.get('license')}
            if key not in best or (chars, fp) < best[key][:2]:
                best[key] = (chars, fp, row, provenance)
            reasoning = raw['messages'][-1].get('reasoning_content')
            if isinstance(reasoning, str) and reasoning.strip():
                cot_chars = chars + len(reasoning.strip())
                # Prefer manageable traces for this small model; exact tokens checked below.
                if cot_chars <= args.max_tokens * 4:
                    cot_row = {'conversations': [dict(m) for m in row['conversations']]}
                    cot_row['conversations'][-1]['reasoning_content'] = reasoning.strip()
                    cot_provenance = dict(provenance, has_reasoning=True)
                    if key not in best_cot or (cot_chars, digest(cot_row)) < best_cot[key][:2]:
                        best_cot[key] = (cot_chars, digest(cot_row), cot_row, cot_provenance)
                else:
                    code_stats['cot_over_character_prefilter'] += 1
        print(f'Finished {path.name}', flush=True)
    cot = Sample(args.total // 5, args.seed + 5, checker, args.max_tokens)
    for (qid, lang), (_, _, row, provenance) in sorted(best_cot.items()):
        if qid not in heldout:
            cot.add(row, provenance)
    print(f'Complete reasoning samples available: {len(cot.heap):,}', flush=True)
    selected_cot_keys = {(e[4]['question_id'], e[4]['language']) for e in cot.heap}
    code = Sample(args.total // 2 - len(cot.heap), args.seed + 1, checker, args.max_tokens)
    for (qid, lang), (_, _, row, provenance) in sorted(best.items()):
        if qid not in heldout and (qid, lang) not in selected_cot_keys:
            code.add(row, dict(provenance, has_reasoning=False))
        else:
            code_stats['excluded_preselected_reasoning_or_heldout_candidates'] += 1
    code.heap.extend(cot.heap)
    code.seen.update(cot.seen)
    del best, best_cot
    print(f'Competitive candidates selected within limit: {len(code.heap):,}', flush=True)

    ultra = Sample(args.total // 5, args.seed + 2, checker, args.max_tokens)
    ultra_stats = Counter()
    for path in sorted((ROOT / 'dataset/ultrachat-200k/data').glob('train_sft-*.parquet')):
        inputs.append(path)
        i = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=['messages']):
            for raw in batch.to_pylist():
                i += 1
                ultra_stats['raw_rows'] += 1
                row = {'conversations': [{'role': m['role'], 'content': m['content'].strip()}
                                         for m in raw['messages']]}
                fp, prompt = fingerprints(row)
                if fp in known or prompt in known_prompts or fp in code.seen:
                    ultra_stats['previously_trained_or_code_overlap'] += 1
                    continue
                ultra.add(row, {'source': 'ultrachat_new', 'file': str(path), 'row': i})
        print(f'Finished {path.name}: {i:,} rows', flush=True)
    units = min(args.total // 10, len(code.heap) // 5,
                len(replay.heap) // 3, len(ultra.heap) // 2)
    if units == 0:
        raise ValueError('Insufficient eligible rows for the requested mixture')
    # Preserve the reasoning allocation even if the total must shrink.
    code_count = units * 5
    reasoning_count = min(units * 2, len(cot.heap))
    final_entries = sorted([e for e in code.heap if not e[4]['has_reasoning']], reverse=True)
    reasoning_count = max(reasoning_count, code_count - len(final_entries))
    selected_code = cot.take(reasoning_count) + final_entries[:code_count - reasoning_count]
    selected = selected_code + replay.take(units * 3) + ultra.take(units * 2)
    random.Random(args.seed + 3).shuffle(selected)
    totals, lengths, languages = Counter(), {}, Counter()
    seen = set()
    temp = output.with_suffix('.jsonl.tmp')
    temp_sources = provenance_path.with_suffix('.jsonl.tmp')
    output_hash = hashlib.sha256()
    with temp.open('wb') as f, temp_sources.open('w', encoding='utf-8', newline='\n') as meta:
        for _, fp, row, tokens, provenance in selected:
            if fp in seen:
                raise ValueError('Cross-source duplicate in final output')
            seen.add(fp)
            source = provenance['source']
            totals[source] += 1
            lengths.setdefault(source, []).append(tokens)
            if source == 'competitive':
                languages[provenance['language']] += 1
            encoded = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
            output_hash.update(encoded)
            f.write(encoded)
            meta.write(json.dumps(dict(provenance, tokens=tokens, fingerprint=fp), ensure_ascii=False) + '\n')
    token_stats = {}
    for source, values in lengths.items():
        values.sort()
        token_stats[source] = {'total': sum(values), 'mean': round(sum(values) / len(values), 1),
                               'p50': values[len(values)//2], 'p95': values[int(len(values)*.95)],
                               'max': max(values)}
    report = {'output': str(output), 'sha256': output_hash.hexdigest(), 'seed': args.seed,
              'requested_rows': args.total, 'total_rows': len(selected), 'max_tokens': args.max_tokens,
              'ratio_basis': 'conversation count', 'counts': totals, 'token_stats': token_stats,
              'competitive_languages': languages, 'competitive_scan': code_stats,
              'competitive_reasoning_rows': reasoning_count,
              'competitive_sample_rate': args.competitive_sample_rate,
              'competitive_cot_sampling': cot.stats,
              'competitive_sampling': code.stats, 'replay_sampling': replay.stats,
              'ultrachat_scan': ultra_stats, 'ultrachat_sampling': ultra.stats,
              'previous_mix_unique_rows': prior_count,
              'policy': {'competitive': 'random row subsample; at most one sample per question/language; shortest character-length candidate for each type; only explicitly train rows; exclude non-train question IDs observed in the random subsample; target 40% competitive rows with complete reasoning_content, remainder final answers only; CoT character prefilter <= 4 * max_tokens',
                         't2t': 'intentional replay of previously trained data; preserve message fields',
                         'ultrachat': 'train_sft only; exclude conversation and first-user-prompt overlap with both previous datasets',
                         'length': 'full local chat-template token count; discard over-limit samples, never truncate',
                         'validation': 'structural checks only; generated code not executed; no benchmark decontamination beyond source split labels',
                         'sampling': 'uniform random-priority sampling after stated filtering; no replacement; reduce total to preserve exact 50:30:20 if needed'},
              'tokenizer_sha256': hashlib.sha256((ROOT / 'model/tokenizer.json').read_bytes()).hexdigest(),
              'template_sha256': hashlib.sha256((ROOT / 'model/tokenizer_config.json').read_bytes()).hexdigest(),
              'inputs': [{'path': str(x), 'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns} for x in inputs]}
    temp.replace(output)
    temp_sources.replace(provenance_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
