"""Re-select a code-think SFT lane by *natural* think length instead of N(2048, 1024^2).

Why: the v4 code-think lane was resampled to a truncated normal around 2,048
reasoning tokens, which made "code problem -> ~2k tokens of thinking" the only
code-thinking behaviour the model ever saw, while the 4,096-token LCB budget
leaves roughly 3,650 for the answer (docs/reports/2026-11-20-sft-loss-performative-thinking.md,
section 4.5). This tool instead

* keeps every row's reasoning intact -- rows are *selected*, never truncated or
  rewritten, so the ``</think>`` closing signal is never cut off;
* drops rows above the top length bucket (selection, not truncation);
* guarantees a quota of genuinely short thinking (default 25% <= 256 tokens),
  so "block opened, then closed quickly" exists as a learnable pattern;
* optionally round-robins over a difficulty field inside each bucket;
* drops repetitive reasoning (same 5-gram heuristic as
  ``build_clean_easy_competitive_sft.py``) so loops are not taught;
* keeps ``--max-per-question`` generations per question (default 1).

It performs **no synthesis**. If the candidate pool has too few short traces the
shortfall is reported per bucket and the quota is redistributed to the other
buckets; source more short traces rather than inventing them.

The default bucket shares are a starting point chosen for this experiment, not a
validated recipe; tune ``--buckets`` and compare via the ablation in the report.

Empty think blocks are a separate switch handled at pack time: build the packed
cache with ``INSTINCT_EMPTY_THINK_RATIO`` between 0.2 and 0.4 instead of 0 (the
v4 packer forced 0 and asserted ``empty_thinking_blocks == 0``), and relax that
assertion for the new run.

Input: JSONL (optionally ``.gz``) rows whose last message is an assistant turn
carrying ``reasoning_content``. Both the repo SFT spelling (``conversations``)
and the raw UltraData spelling (``messages``) are accepted. Token counts use
``model/tokenizer.json`` (needs the ``tokenizers`` package).

    python scripts/data_builder/rebalance_code_think_lane.py \
        --input dataset/code_think_candidates.jsonl \
        --output dataset/code_think_lane_v5.jsonl --target-rows 7390
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKETS = '0-256:0.25,257-768:0.25,769-1536:0.30,1537-2500:0.20'
FENCE = re.compile(r'```[^\n`]*\n.*?```', re.S)


def parse_buckets(spec):
    buckets = []
    for part in spec.split(','):
        match = re.fullmatch(r'\s*(\d+)-(\d+)\s*:\s*([0-9.]+)\s*', part)
        if not match:
            raise ValueError(f'bad bucket {part!r}; expected lo-hi:share')
        lo, hi, share = int(match.group(1)), int(match.group(2)), float(match.group(3))
        if hi < lo or share <= 0:
            raise ValueError(f'bad bucket {part!r}')
        buckets.append({'lo': lo, 'hi': hi, 'share': share, 'name': f'{lo}-{hi}'})
    total = sum(b['share'] for b in buckets)
    for bucket in buckets:
        bucket['share'] /= total
    return buckets


def open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else open(path, encoding='utf-8')


def priority(seed, *parts):
    body = '\x1f'.join(map(str, (seed,) + parts)).encode('utf-8', 'replace')
    return int.from_bytes(hashlib.sha256(body).digest()[:8], 'big')


def get_path(row, dotted):
    value = row
    for key in dotted.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def messages_of(row):
    """Return the conversation list from either repo SFT or UltraData source rows.

    The packed SFT corpora use ``conversations`` while the raw UltraData code-think
    shards (``tmp/instinctv2_sft_2b_sources/data/think/Code``) use ``messages``;
    both put the reasoning in the assistant turn's ``reasoning_content``.
    """
    if not isinstance(row, dict):
        return None
    for key in ('conversations', 'messages'):
        value = row.get(key)
        if isinstance(value, list):
            return value
    return None


def assistant_turn(row):
    messages = messages_of(row)
    if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'assistant':
        return None
    if not any(m.get('role') == 'user' and isinstance(m.get('content'), str) for m in messages):
        return None
    return messages[-1]


def question_key(row):
    messages = messages_of(row) or []
    users = [m['content'] for m in messages if m.get('role') == 'user']
    return hashlib.sha256(re.sub(r'\s+', ' ', users[0]).strip().casefold().encode('utf-8', 'replace')).hexdigest()


def repetitive(text, threshold=0.88):
    words = re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text.lower())
    if len(words) < 30:
        return False
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    return len(set(grams)) / len(grams) < threshold


def rejection(row, require_code):
    turn = assistant_turn(row)
    if turn is None or not isinstance(turn.get('content'), str):
        return 'invalid'
    reasoning = turn.get('reasoning_content')
    if not isinstance(reasoning, str) or not reasoning.strip():
        return 'no_reasoning'
    if '<think>' in reasoning or '</think>' in reasoning or '<think>' in turn['content'] \
            or '</think>' in turn['content']:
        return 'think_tag_in_text'
    if not turn['content'].strip():
        return 'empty_answer'
    if require_code and not FENCE.search(turn['content']):
        return 'no_code_block'
    if repetitive(reasoning):
        return 'reasoning_repetition'
    return None


def allocate(buckets, available, target, max_overshoot=1.5):
    """Per-bucket take counts: share-proportional, then water-fill any shortfall.

    A bucket may exceed its requested quota by at most ``max_overshoot`` times, so
    a skewed candidate pool cannot silently flip the length profile (for example
    turning a 25% short-think quota into 85%). If the capped buckets cannot reach
    ``target`` the result is simply smaller than ``target``.
    """
    capacity = {b['name']: min(available[b['name']],
                               int(math.ceil(b['share'] * target * max_overshoot)))
                for b in buckets}
    take = {b['name']: min(capacity[b['name']], int(math.floor(b['share'] * target))) for b in buckets}
    guard = 0
    while sum(take.values()) < target and guard < 10_000:
        guard += 1
        spare = [b for b in buckets if capacity[b['name']] > take[b['name']]]
        if not spare:
            break
        leftover = target - sum(take.values())
        weight = sum(b['share'] for b in spare)
        moved = 0
        for b in spare:
            extra = min(capacity[b['name']] - take[b['name']], int(leftover * b['share'] / weight))
            take[b['name']] += extra
            moved += extra
        if not moved:  # fewer rows left than buckets: hand out single rows by share
            for b in sorted(spare, key=lambda x: -x['share']):
                if sum(take.values()) < target and capacity[b['name']] > take[b['name']]:
                    take[b['name']] += 1
    return take


def select(rows, count_tokens, buckets, target_rows, seed=42, max_per_question=1,
           difficulty_field=None, require_code=True, max_overshoot=1.5):
    dropped = Counter()
    top = max(b['hi'] for b in buckets)
    by_question = defaultdict(list)
    for row in rows:
        reason = rejection(row, require_code)
        if reason:
            dropped[reason] += 1
            continue
        tokens = count_tokens(assistant_turn(row)['reasoning_content'])
        bucket = next((b for b in buckets if b['lo'] <= tokens <= b['hi']), None)
        if bucket is None:
            dropped['above_top_bucket' if tokens > top else 'below_buckets'] += 1
            continue
        key = question_key(row)
        by_question[key].append({'row': row, 'tokens': tokens, 'bucket': bucket['name'],
                                 'priority': priority(seed, key, len(by_question[key]), tokens)})
    # Cap generations per question, preferring the buckets that are scarcest overall.
    pool_size = Counter(item['bucket'] for items in by_question.values() for item in items)
    pool = []
    for items in by_question.values():
        items.sort(key=lambda x: (pool_size[x['bucket']], x['priority']))
        kept = items[:max_per_question]
        dropped['question_over_cap'] += len(items) - len(kept)
        pool.extend(kept)

    grouped = defaultdict(list)
    for item in pool:
        grouped[item['bucket']].append(item)
    available = {b['name']: len(grouped[b['name']]) for b in buckets}
    quotas = allocate(buckets, available, target_rows, max_overshoot)

    chosen = []
    for bucket in buckets:
        items = grouped[bucket['name']]
        if difficulty_field:
            ranks = defaultdict(int)
            ordered = []
            for item in sorted(items, key=lambda x: x['priority']):
                level = str(get_path(item['row'], difficulty_field))
                ordered.append((ranks[level], item['priority'], item))
                ranks[level] += 1
            items = [x[2] for x in sorted(ordered, key=lambda x: (x[0], x[1]))]
        else:
            items = sorted(items, key=lambda x: x['priority'])
        chosen.extend(items[:quotas[bucket['name']]])
    chosen.sort(key=lambda x: x['priority'])  # deterministic shuffle

    lengths = [x['tokens'] for x in chosen]
    pct = lambda q: (sorted(lengths)[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else None)
    report = {
        'target_rows': target_rows, 'selected_rows': len(chosen), 'seed': seed,
        'dropped': dict(dropped), 'questions': len(by_question),
        'buckets': [{'bucket': b['name'], 'share': round(b['share'], 4),
                     'quota_requested': int(round(b['share'] * target_rows)),
                     'available': available[b['name']], 'selected': quotas[b['name']],
                     'final_share': round(quotas[b['name']] / len(chosen), 4) if chosen else None,
                     'shortfall': max(0, int(round(b['share'] * target_rows)) - quotas[b['name']])}
                    for b in buckets],
        'think_tokens': {'sum': sum(lengths), 'mean': statistics.fmean(lengths) if lengths else None,
                         'p50': pct(0.5), 'p90': pct(0.9), 'max': max(lengths) if lengths else None},
        'share_le_256': (sum(x <= 256 for x in lengths) / len(lengths)) if lengths else None,
        'notes': ['No rows were truncated, rewritten or synthesized.',
                  'Pack with INSTINCT_EMPTY_THINK_RATIO in 0.2-0.4 (v4 used 0) and relax the '
                  'empty_thinking_blocks == 0 assertion for this run.'],
    }
    if len(chosen) < target_rows:
        report['notes'].append(
            f'pool cannot satisfy the profile within max_overshoot={max_overshoot}: selected '
            f'{len(chosen)} < target {target_rows}; see per-bucket shortfall')
    short = report['buckets'][0]
    if short['shortfall']:
        report['notes'].append(
            f"short-think shortfall in bucket {short['bucket']}: requested {short['quota_requested']}, "
            f"got {short['selected']}; source more short traces instead of synthesizing them")
    return [x['row'] for x in chosen], report


class TokenCounter:
    def __init__(self, tokenizer_json):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise SystemExit('the `tokenizers` package is required to count reasoning tokens') from error
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def __call__(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def main(argv=None, count_tokens=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', required=True, nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', default=None, help='default: <output>.report.json')
    parser.add_argument('--tokenizer', default=str(ROOT / 'model' / 'tokenizer.json'))
    parser.add_argument('--target-rows', type=int, default=7390, help='v4 lane size, for comparable token mass')
    parser.add_argument('--buckets', default=DEFAULT_BUCKETS, help='lo-hi:share,... in reasoning tokens')
    parser.add_argument('--max-per-question', type=int, default=1)
    parser.add_argument('--max-overshoot', type=float, default=1.5,
                        help='a bucket may exceed its requested quota by at most this factor when '
                             'redistributing another bucket\'s shortfall (fewer rows beat a skewed mix)')
    parser.add_argument('--difficulty-field', default=None, help='dotted path, e.g. metadata.difficulty')
    parser.add_argument('--allow-no-code', action='store_true', help='do not require a fenced block in the answer')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    def rows():
        for path in args.input:
            with open_text(path) as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    counter = count_tokens or TokenCounter(args.tokenizer)
    selected, report = select(rows(), counter, parse_buckets(args.buckets), args.target_rows,
                              args.seed, args.max_per_question, args.difficulty_field,
                              not args.allow_no_code, args.max_overshoot)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    report_path = Path(args.report) if args.report else output.with_suffix(output.suffix + '.report.json')
    report['args'] = {k: v for k, v in vars(args).items() if k != 'input'} | {'input': list(args.input)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"selected {report['selected_rows']}/{report['target_rows']} rows -> {output}")
    for bucket in report['buckets']:
        print(f"  {bucket['bucket']:>10}: available={bucket['available']} quota={bucket['quota_requested']} "
              f"selected={bucket['selected']} final_share={bucket['final_share']} shortfall={bucket['shortfall']}")
    for note in report['notes']:
        print('note:', note, file=sys.stderr)
    return report


if __name__ == '__main__':
    main()
