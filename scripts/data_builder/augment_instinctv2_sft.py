"""Fill an audited InstinctV2 SFT token budget while keeping validation separate."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import math
import os
from pathlib import Path
import random
import re
import shutil

import orjson

import build_instinctv2_sft as base
from scripts.data_builder.instinctv2_sft_scenarios import IDENTITY, tool_episode

ORIGINAL = base.DATA / 'instinctv2_sft_v1'
OUTPUT = base.DATA / 'instinctv2_sft_v2'
TARGET = 36_000_000
OFFICIAL_CODE = 'https://huggingface.co/datasets/openbmb/UltraData-SFT-2605'
IDENTITY_CONTEXTS = (
    ('给模型信息面板写一条答复。', 'Write a response for a model information panel. '),
    ('这条答复要用于新手引导。', 'This answer is for first-time users. '),
    ('请回答产品文档里的这个问题。', 'Answer this question from the deployment manual. '),
    ('只根据你的模型身份和当前上下文回答。', 'Answer from your model identity and the current context. '),
    ('请为安装向导回答这个问题。', 'Answer this question for the setup wizard. '),
    ('我在核对模型卡上的基本信息。', 'I am checking the basic information on the model card. '),
)


def source_pairs(directory, split):
    return zip(base.rows(directory / f'{split}_identity_clean.jsonl'),
               base.rows(directory / f'{split}.provenance.jsonl.gz'), strict=True)


def new_candidates(used_groups):
    for category, folder, sidecar in (
        ('verified_math', 'verified_cot_sft', 'provenance.jsonl.gz'),
        ('verified_logic', 'reasoning_sft_strict', 'verification_records.jsonl.gz'),
        ('instruction', 'instruction_understanding_sft', 'provenance.jsonl.gz'),
    ):
        source = base.DATA / folder / 'train.jsonl'
        provenance = base.DATA / folder / sidecar
        meta = {item['row']: item for item in base.rows(provenance) if item['file_split'] == 'train'}
        counts = Counter()
        for number, row in enumerate(base.rows(source)):
            item = meta[number]
            group = item.get('group_id', item.get('semantic_hash'))
            if (category, group) in used_groups or counts[group] >= 2:
                continue
            counts[group] += 1
            yield 'train', category, row, {'group': group, 'family': item.get('family', item.get('kind')),
                                          'language': item['language'], 'source_row': number + 1,
                                          'path': source.relative_to(base.ROOT).as_posix()}
    for index in range(5200, 7800):
        row, meta = tool_episode(index, 'train')
        yield 'train', 'tools', row, meta


def supplement(path, workers):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    groups = {(meta['category'], meta['group']) for split in ('train', 'validation')
              for _, meta in source_pairs(ORIGINAL, split)}
    counts, terms, cpu_seconds = defaultdict(Counter), Counter(), Counter()
    with path.open('xb') as stream, ProcessPoolExecutor(max_workers=workers, initializer=base.init_worker) as pool:
        for window in base.batches(base.batches(new_candidates(groups), 256), 2 * workers):
            for screened, filters, hits, pid, elapsed in pool.map(base.screen_batch, window):
                for _, category, row, meta in screened:
                    base.emit(stream, {'row': row, 'meta': {'category': category, **meta}})
                for (_, category), value in filters.items():
                    counts[category].update(value)
                terms.update(hits)
                cpu_seconds[str(pid)] += elapsed
    report = {'screen_counts': dict(counts), 'identity_keyword_hits': dict(terms),
              'worker_cpu_seconds': dict(cpu_seconds), 'sha256': base.digest(path),
              'source_sha256': {name: base.digest(base.DATA / name / 'train.jsonl') for name in (
                  'verified_cot_sft', 'reasoning_sft_strict', 'instruction_understanding_sft')}}
    path.with_suffix('.report.json').write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(report).decode(), flush=True)


def budget(log, target, hours):
    windows = [int(value) for value in re.findall(r'window_tokens/s=(\d+)', log)]
    supervised = [int(value) for value in re.findall(r'useful_tokens/s=(\d+)', log)]
    if len(windows) < 4 or '(78/78)' not in log:
        raise ValueError('Calibration did not finish or lacks warm throughput windows')
    # Round down the slowest measured warm window, then keep a wall-time reserve.
    rate = min(windows) // 100 * 100
    reserve = .15
    useful_rate = rate * .9916
    return {'calibration': {'gpu': 'RTX 5070 Ti', 'params': 1010278000, 'dtype': 'BF16',
                            'param_dtype': 'FP32', 'optimizer': 'Adafactor', 'activation_checkpoint': 2,
                            'batch': 1, 'accumulation': 8, 'sequence_length': 2048, 'packing': 'fixed',
                            'warmup_microbatches': 8, 'packed_fill': .9916, 'measured_window_input_tps': windows,
                            'measured_supervised_tps': supervised},
            'training_hours_assumption': hours, 'wall_time_reserve_fraction': reserve,
            'conservative_padded_input_tps': rate, 'estimated_nonpadding_input_tps': useful_rate,
            'nonpadding_token_capacity': {f'{h:g}': int(h * 3600 * useful_rate * (1 - reserve))
                                         for h in sorted({4, 8, 12, hours})},
            'unique_target_tokens': target,
            'policy': 'One fresh-data pass, then compare held-out metrics before any second pass. Capacity includes reserve; tokens are rendered inputs, not supervised labels.'}


def take(candidates, seen, limit):
    selected, tokens = [], 0
    for row, meta in candidates:
        key = meta['first_user_sha256']
        length = meta['tokens']
        if key in seen or tokens + length > limit:
            continue
        selected.append((row, meta))
        seen.add(key)
        tokens += length
        if limit - tokens < 12:
            break
    return selected, tokens


def added_identity():
    for prefix, context in enumerate(IDENTITY_CONTEXTS):
        for topic, variants in enumerate(IDENTITY):
            for i, question in enumerate(variants[:4]):
                language = i // 2
                yield 'train', 'identity', {'conversations': [
                    {'role': 'user', 'content': context[language] + question},
                    {'role': 'assistant', 'content': variants[4 + language]},
                ]}, {'group': f'identity/augmentation/{prefix}/{topic}', 'language': ('zh', 'en')[language]}


def assemble(args):
    existing = orjson.loads((ORIGINAL / 'build.report.json').read_bytes())
    calibration = args.calibration.read_text(encoding='utf-8-sig')
    plan = budget(calibration, args.target_tokens, args.hours)
    calibration_save = orjson.loads(args.calibration.with_name('checkpoint_verification.json').read_bytes())
    if calibration_save != {'epoch': 1, 'microbatches': 78, 'archive_crc': True, 'matching_export': True}:
        raise ValueError('Calibration checkpoint was not verified')
    plan['calibration']['checkpoint_verified'] = True
    old = list(source_pairs(ORIGINAL, 'train'))
    validation = list(source_pairs(ORIGINAL, 'validation'))
    seen = {meta['first_user_sha256'] for _, meta in old + validation}
    baseline_tokens = existing['splits']['train']['tokens']
    capacity = plan['nonpadding_token_capacity'][f'{args.hours:g}']
    target = min(args.target_tokens, capacity)
    if target <= baseline_tokens:
        raise ValueError('The requested budget does not fit the audited baseline corpus')
    gap = target - baseline_tokens
    old_code_tokens = sum(existing['splits']['train']['source_tokens'][key] for key in ('magicoder', 'executed_code'))
    code_budget = min(gap, max(0, math.floor(target * .35) - old_code_tokens))
    code_report = orjson.loads((args.code / 'code.report.json').read_bytes())
    source_verification = orjson.loads(args.source_verification.read_bytes())
    verified_paths = {item['path'] for item in source_verification['files']}
    if (source_verification['repo'] != code_report['repository']
            or source_verification['revision'] != code_report['revision']
            or not code_report['source_rows'].keys() <= verified_paths):
        raise ValueError('L3 source manifest does not cover the selected shards')
    code_path = args.code / 'code_identity_clean.jsonl'
    code_meta = args.code / 'code.provenance.jsonl'
    for path, key in ((code_path, 'code_sha256'), (code_meta, 'provenance_sha256')):
        if base.digest(path) != code_report[key]:
            raise ValueError('L3 candidate hash mismatch')
    if code_report['base_train_sha256'] != existing['splits']['train']['sha256'] or code_report['base_validation_sha256'] != existing['splits']['validation']['sha256']:
        raise ValueError('L3 exclusions do not match the selected baseline')
    fresh = [(row, {'category': 'openbmb_l3_code', **meta})
             for row, meta in zip(base.rows(code_path), base.rows(code_meta), strict=True)]
    rng = random.Random(f'{base.SEED}/sft-augmentation')
    rng.shuffle(fresh)
    additions, code_tokens = take(fresh, seen, code_budget)
    pools = defaultdict(list)
    supplements = orjson.loads(args.supplement.with_suffix('.report.json').read_bytes())
    if base.digest(args.supplement) != supplements['sha256']:
        raise ValueError('Supplement hash mismatch')
    for item in base.rows(args.supplement):
        pools[item['meta']['category']].append((item['row'], item['meta']))
    base.init_worker()
    new_anchors, _, _, _, _ = base.screen_batch(added_identity())
    anchors, anchor_tokens = take(((row, {'category': 'identity', **meta})
                                  for _, _, row, meta in new_anchors), seen, gap - code_tokens)
    additions.extend(anchors)
    remaining = gap - code_tokens - anchor_tokens
    quota = {'verified_math': int(remaining * .27), 'verified_logic': int(remaining * .21),
             'tools': int(remaining * .13)}
    quotas_filled = {'identity': anchor_tokens}
    for category, limit in quota.items():
        picked, spent = take(base.ordered_by_family(pools[category], rng), seen, limit)
        additions.extend(picked)
        quotas_filled[category] = spent
    picked, spent = take(base.ordered_by_family(pools['instruction'], rng), seen, gap - code_tokens - sum(quotas_filled.values()))
    additions.extend(picked)
    quotas_filled['instruction'] = spent
    # Redistribute any constrained category's residual to the other audited pools.
    for category in ('verified_logic', 'tools', 'verified_math', 'instruction'):
        remainder = gap - code_tokens - sum(quotas_filled.values())
        if remainder < 12:
            break
        picked, spent = take(base.ordered_by_family(pools[category], rng), seen, remainder)
        additions.extend(picked)
        quotas_filled[category] += spent
    rows = old + additions
    rng.shuffle(rows)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    train = out / 'train_identity_clean.jsonl'
    provenance = out / 'train.provenance.jsonl.gz'
    with train.open('xb') as data, provenance.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as origin:
        for index, (row, meta) in enumerate(rows, 1):
            base.emit(data, row)
            base.emit(origin, {**meta, 'row': index})
    for name in ('validation_identity_clean.jsonl', 'validation.provenance.jsonl.gz'):
        shutil.copyfile(ORIGINAL / name, out / name)
    tokens_by_category, rows_by_category = Counter(), Counter()
    for _, meta in rows:
        tokens_by_category[meta['category']] += meta['tokens']
        rows_by_category[meta['category']] += 1
    realized = sum(tokens_by_category.values())
    plan.update({'target_after_capacity_cap': target, 'baseline_tokens': baseline_tokens, 'realized_unique_tokens': realized,
                 'unfilled_tokens': target - realized, 'code_target_tokens': code_budget, 'code_added_tokens': code_tokens,
                 'code_target_shortfall': code_budget - code_tokens, 'supplement_added_tokens': quotas_filled,
                 'estimated_microbatches_per_pass': math.ceil(realized / (2048 * .9916)),
                 'estimated_optimizer_updates_per_pass': math.ceil(realized / (2048 * .9916 * 8)),
                 'comparison_token_checkpoints': sorted({realized // 4, realized // 2, min(baseline_tokens, realized), realized}),
                 'comparison_protocol': 'Use the same base checkpoint and consumed input tokens. Track HumanEval+/code, instruction following, tool schema and identity on held-out data before selecting a checkpoint or repeating the mix.',
                 'one_pass_hours_with_reserve': realized / (plan['estimated_nonpadding_input_tps'] * .85 * 3600),
                 'two_pass_hours_with_reserve': 2 * realized / (plan['estimated_nonpadding_input_tps'] * .85 * 3600),
                 'calibration_log_sha256': base.digest(args.calibration)})
    report = {'identity': 'InstinctV2', 'status': 'jsonl_built', 'training_max_seq_len': 2048, 'budget': plan,
              'official_sft': {'url': OFFICIAL_CODE, 'revision': 'affda6aca75e7cff78e73f93ad08d4c3b01f097c',
                               'model_card': 'https://huggingface.co/openbmb/MiniCPM5-1B',
                               'documented_1b_deep_thinking_tokens': 200000000000,
                               'documented_1b_hybrid_thinking_tokens': 200000000000,
                               'anonymous_access': '401 GatedRepo', 'rows_included': 0},
              'l3_audit': code_report, 'l3_source_verification': source_verification, 'noncode_audit': supplements,
              'baseline_verification_sha256': base.digest(ORIGINAL / 'verification.report.json'),
              'splits': {'train': {'rows': len(rows), 'tokens': realized, 'sha256': base.digest(train),
                                   'provenance_sha256': base.digest(provenance), 'source_rows': dict(rows_by_category),
                                   'source_tokens': dict(tokens_by_category), 'added_rows': len(additions),
                                   'thinking_rows': sum(meta['thinking'] for _, meta in rows),
                                   'supervised_tokens': sum(meta['supervised_tokens'] for _, meta in rows),
                                   'max_tokens': max(meta['tokens'] for _, meta in rows)},
                         'validation': existing['splits']['validation']}}
    (out / 'build.report.json').write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    print(orjson.dumps({'budget': plan, 'train': report['splits']['train']}, option=orjson.OPT_INDENT_2).decode(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('supplement')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--workers', type=int, default=6)
    mix = sub.add_parser('assemble')
    mix.add_argument('--output', type=Path, default=OUTPUT)
    mix.add_argument('--code', type=Path, required=True)
    mix.add_argument('--supplement', type=Path, required=True)
    mix.add_argument('--calibration', type=Path, required=True)
    mix.add_argument('--source-verification', type=Path, required=True)
    mix.add_argument('--target-tokens', type=int, default=TARGET)
    mix.add_argument('--hours', type=float, default=8.0)
    args = parser.parse_args()
    if args.command == 'supplement':
        supplement(args.output, args.workers)
    else:
        assemble(args)


if __name__ == '__main__':
    main()
