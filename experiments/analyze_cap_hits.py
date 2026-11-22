"""Classify why LiveCodeBench generations ran into the context cap.

Takes the per-task generations of a think-mode run and separates the cases that
a bigger budget cannot fix (repetition loops, not stopping after the answer was
already written) from the ones it can (still reasoning, answer drafted inside
the think block), optionally split by LiveCodeBench difficulty / platform.

Categories (capped generations only, first match wins):

    capped_after_answer       </think> closed and a final code block exists, yet
                              generation did not stop (EOS problem)
    capped_loop               tail repeats itself (period / n-gram / line repeats)
    capped_code_drafted_in_think
                              a complete fenced code block sits inside the think
                              body but the block never closed
    capped_cut_in_code        cut while a fenced block was still open
    capped_still_thinking     none of the above: long reasoning that never ended

Not capped: ``done_with_code`` / ``done_no_code``.

Input is JSONL / JSON list (optionally ``.gz``), one record per task. The
recorded harness (``eval/livecodebench/<run>/generate.py``) is not tracked in
this repository, so field names are detected, not assumed; override with
``--text-field`` / ``--id-field`` / ``--tokens-field`` if yours differ. A task
counts as *capped* by the first rule that applies:

  1. a recorded ``finish_reason`` (capped iff ``'length'``) or a boolean
     ``capped`` / ``hit_cap`` / ``reached_context_limit`` / ``context_limit_hit``
     field (the recorded LCB harness writes ``reached_context_limit``);
  2. ``--cap-tokens N`` together with a generated-token-count field >= N;
  3. fallback heuristic: no ``</think>`` AND no final code block (reported as
     ``likely`` and flagged in the summary, because it can over-count).

Usage:

    python experiments/analyze_cap_hits.py --outputs generations.jsonl \
        --meta lcb_v6_latest175.jsonl --json cap_hits.json --examples 3

``--meta`` needs ``question_id`` plus ``difficulty`` and/or ``platform`` (the
official ``code_generation_lite`` JSONL has them). Pure standard library.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CLOSE = '</think>'
FENCE_BLOCK = re.compile(r'```[^\n`]*\n.*?```', re.S)
TEXT_FIELDS = ('raw_response', 'response', 'output', 'text', 'completion', 'generation')
ID_FIELDS = ('question_id', 'task_id', 'id')
TOKEN_FIELDS = ('n_tokens', 'generated_tokens', 'num_tokens', 'completion_tokens', 'output_tokens')
CATEGORIES = ('capped_after_answer', 'capped_loop', 'capped_code_drafted_in_think',
              'capped_cut_in_code', 'capped_still_thinking', 'done_with_code', 'done_no_code')


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text[0] == '[':
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pick(record, names, override=None):
    for name in ([override] if override else names):
        if name and name in record and record[name] is not None:
            return record[name]
    return None


def response_text(record, override=None):
    value = pick(record, TEXT_FIELDS, override)
    if value is None and isinstance(record.get('output_list'), list) and record['output_list']:
        value = record['output_list'][0]
    return value if isinstance(value, str) else ''


def tail_period(text, minimum=4, maximum=800, repeats=3):
    """Smallest period p such that the text ends with `repeats` identical copies."""
    for p in range(minimum, maximum + 1):
        if len(text) < p * repeats:
            break
        unit = text[-p:]
        if unit.strip() and all(text[-(k + 1) * p:len(text) - k * p] == unit for k in range(1, repeats)):
            return p
    return None


def repetition(text, tail_chars=4000):
    tail = text[-tail_chars:]
    words = re.findall(r'\S+', tail)
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    dup = 1 - len(set(grams)) / len(grams) if len(grams) >= 50 else 0.0
    lines = [re.sub(r'\s+', ' ', x).strip() for x in tail.splitlines()]
    counts = Counter(x for x in lines if len(x) >= 12)
    return {
        'tail_period': tail_period(tail),
        'dup_4gram_ratio': round(dup, 4),
        'max_line_repeat': max(counts.values()) if counts else 0,
    }


def analyze(record, text_field=None, tokens_field=None, cap_tokens=None, think_mode='on',
            loop_dup=0.5, loop_line=5, tail_chars=4000):
    text = response_text(record, text_field)
    closed = CLOSE in text
    thinking = think_mode == 'on' or (think_mode == 'auto' and closed)
    think_body, answer = (text.partition(CLOSE)[0], text.partition(CLOSE)[2]) if closed else (
        (text, '') if thinking else ('', text))
    final_blocks = FENCE_BLOCK.findall(answer)
    think_blocks = FENCE_BLOCK.findall(think_body)
    scanned = answer if closed or not thinking else think_body
    open_fence = scanned.count('```') % 2 == 1
    rep = repetition(text, tail_chars)
    loop = (rep['tail_period'] is not None or rep['dup_4gram_ratio'] >= loop_dup
            or rep['max_line_repeat'] >= loop_line)

    tokens = pick(record, TOKEN_FIELDS, tokens_field)
    rule = None
    # Explicit per-task cap flags used by the recorded harness and by common
    # inference dumps. ``reached_context_limit`` is what
    # eval/livecodebench/<run>/generate.py writes next to ``responses.jsonl``.
    recorded = next((record[k] for k in ('capped', 'hit_cap', 'reached_context_limit',
                                        'hit_context_limit', 'context_limit_hit')
                     if record.get(k) is not None), None)
    if record.get('finish_reason') is not None:
        capped, rule = record['finish_reason'] == 'length', 'recorded'
    elif recorded is not None:
        capped, rule = bool(recorded), 'recorded'
    elif cap_tokens and isinstance(tokens, (int, float)):
        capped, rule = tokens >= cap_tokens, 'token_count'
    else:
        capped, rule = (not closed and not final_blocks), 'likely'

    if not capped:
        category = 'done_with_code' if (final_blocks or (not thinking and think_blocks)) else 'done_no_code'
    elif closed and final_blocks:
        category = 'capped_after_answer'
    elif loop:
        category = 'capped_loop'
    elif think_blocks and not closed:
        category = 'capped_code_drafted_in_think'
    elif open_fence:
        category = 'capped_cut_in_code'
    else:
        category = 'capped_still_thinking'
    return {
        'id': str(pick(record, ID_FIELDS) or ''), 'category': category, 'capped': capped, 'cap_rule': rule,
        'closed_think': closed, 'chars': len(text), 'tokens': tokens,
        'final_code_blocks': len(final_blocks), 'think_code_blocks': len(think_blocks),
        'open_fence': open_fence, 'loop': loop, **rep, 'tail': text[-300:],
    }


def load_meta(path):
    meta = {}
    if path:
        for row in read_records(path):
            if isinstance(row, dict) and 'question_id' in row:
                meta[str(row['question_id'])] = row
    return meta


def summarize(results, meta, group_key):
    groups = defaultdict(list)
    for item in results:
        value = meta.get(item['id'], {}).get(group_key) if group_key else None
        groups[str(value) if value is not None else ('unknown' if group_key else 'all')].append(item)
    table = {}
    for name, items in sorted(groups.items()):
        capped = [x for x in items if x['capped']]
        lengths = [x['tokens'] if isinstance(x['tokens'], (int, float)) else x['chars'] for x in items]
        table[name] = {
            'n': len(items), 'capped': len(capped),
            'capped_rate': round(len(capped) / len(items), 4) if items else None,
            'closed_think': sum(x['closed_think'] for x in items),
            'categories': {c: sum(x['category'] == c for x in items) for c in CATEGORIES},
            'median_length': statistics.median(lengths) if lengths else None,
            'length_unit': 'tokens' if all(isinstance(x['tokens'], (int, float)) for x in items) else 'chars',
        }
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--meta', default=None, help='JSON/JSONL with question_id + difficulty/platform')
    parser.add_argument('--text-field', default=None)
    parser.add_argument('--id-field', default=None)
    parser.add_argument('--tokens-field', default=None)
    parser.add_argument('--cap-tokens', type=int, default=None, help='generated-token cap, if recorded per task')
    parser.add_argument('--think-mode', choices=('auto', 'on', 'off'), default='on',
                        help="'on' (default): the prompt already opened <think>, so text before the first "
                             "</think> is the think body and an unclosed response is all think body")
    parser.add_argument('--loop-dup', type=float, default=0.5)
    parser.add_argument('--loop-line', type=int, default=5)
    parser.add_argument('--tail-chars', type=int, default=4000)
    parser.add_argument('--examples', type=int, default=0, help='keep the last 300 chars of N tasks per category')
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)

    records = read_records(args.outputs)
    if args.id_field:
        for record in records:
            record['question_id'] = record.get(args.id_field)
    meta = load_meta(args.meta)
    results = [analyze(r, args.text_field, args.tokens_field, args.cap_tokens, args.think_mode,
                       args.loop_dup, args.loop_line, args.tail_chars) for r in records]
    rules = Counter(x['cap_rule'] for x in results)
    report = {
        'tasks': len(results), 'cap_rules': dict(rules),
        'warning': ('capped = heuristic "likely" (no </think> and no final code) for some tasks; '
                    'pass per-task finish_reason or --cap-tokens for an exact count')
        if rules.get('likely') else None,
        'overall': summarize(results, meta, None)['all'],
        'by_difficulty': summarize(results, meta, 'difficulty') if meta else None,
        'by_platform': summarize(results, meta, 'platform') if meta else None,
    }
    if args.examples:
        kept = defaultdict(list)
        for item in results:
            if len(kept[item['category']]) < args.examples:
                kept[item['category']].append({k: item[k] for k in ('id', 'tokens', 'chars', 'tail')})
        report['examples'] = dict(kept)

    print(f"tasks={report['tasks']} cap_rules={report['cap_rules']}")
    if report['warning']:
        print('WARNING:', report['warning'])
    for title, block in (('overall', {'all': report['overall']}),
                         ('difficulty', report['by_difficulty'] or {}),
                         ('platform', report['by_platform'] or {})):
        for name, row in block.items():
            cats = ' '.join(f'{k.replace("capped_", "c:")}={v}' for k, v in row['categories'].items() if v)
            print(f"[{title}:{name}] n={row['n']} capped={row['capped']} ({row['capped_rate']}) "
                  f"closed={row['closed_think']} | {cats}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
