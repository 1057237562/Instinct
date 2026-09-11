"""Audit all output rows, then independently check sampled lengths with HF.

Use the normal data Python for the default audit; use a Python environment
with working transformers for --hf-check. Neither mode downloads models.
"""
import argparse
from collections import Counter
import hashlib
import heapq
from itertools import zip_longest
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset/sft_competitive_t2t_ultrachat_16k_reasoning80.jsonl')
    parser.add_argument('--hf-check', action='store_true')
    args = parser.parse_args()
    report_path = args.output.with_suffix('.report.json')
    report = json.loads(report_path.read_text('utf-8'))
    sample_path = ROOT / '.cache/continued_sft_large/validation_sample.jsonl'
    if args.hf_check:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'model'), local_files_only=True)
        checked = 0
        for line in sample_path.read_text('utf-8').splitlines():
            item = json.loads(line)
            assert item['output_sha256'] == report['sha256']
            messages = [dict(m) for m in item['row']['conversations']]
            tools = None
            for m in messages:
                if m.get('role') == 'system' and m.get('tools'):
                    tools = json.loads(m['tools']) if isinstance(m['tools'], str) else m['tools']
                if isinstance(m.get('tool_calls'), str):
                    m['tool_calls'] = json.loads(m['tool_calls'])
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
            actual = len(tokenizer(rendered).input_ids)
            assert actual == item['tokens'], (item['index'], actual, item['tokens'])
            assert actual <= report['max_tokens']
            checked += 1
        report['verification']['independent_hf_length_checks'] = checked
        print(f'PASS: {checked} HF tokenizer checks, including the 32 longest rows', flush=True)
    else:
        from scripts.mix_large_continued_sft import fingerprints, has_reasoning, valid
        seen, groups = set(), Counter()
        counts, thought_counts, token_totals, byte_totals = Counter(), Counter(), Counter(), Counter()
        h = hashlib.sha256()
        rng, reservoir, longest, first_by_source = random.Random(42), [], [], {}
        n, max_tokens = 0, 0
        with args.output.open('rb') as rows, args.output.with_suffix('.sources.jsonl').open('rb') as provenance:
            for i, (line, meta_line) in enumerate(zip_longest(rows, provenance)):
                assert line is not None and meta_line is not None, 'Provenance length mismatch'
                h.update(line)
                row, meta = json.loads(line), json.loads(meta_line)
                assert valid(row)
                fp, _ = fingerprints(row)
                assert fp.hex() == meta['fingerprint'] and fp not in seen, f'Duplicate or bad fingerprint at {i}'
                seen.add(fp)
                thought = bool(has_reasoning(row))
                assert thought == meta['has_reasoning']
                source, tokens = meta['source'], meta['tokens']
                assert len(line) == meta['bytes']
                assert 0 < tokens <= report['max_tokens']
                if source.startswith('competitive'):
                    splits = meta['split'] if isinstance(meta['split'], list) else [meta['split']]
                    assert splits and all(s == 'train' for s in splits)
                    groups[(meta['question_id'], meta['language'])] += 1
                if source == 'ultrachat_new':
                    assert Path(meta['file']).name.startswith('train_sft-')
                if source.endswith('_cot'):
                    assert thought
                if source == 't2t_final':
                    assert not thought
                counts[source] += 1
                thought_counts[source] += int(thought)
                token_totals[source] += tokens
                byte_totals[source] += len(line)
                max_tokens = max(max_tokens, tokens)
                item = {'index': i, 'row': row, 'tokens': tokens, 'output_sha256': report['sha256']}
                first_by_source.setdefault(source, item)
                if len(reservoir) < 256:
                    reservoir.append(item)
                else:
                    replacement = rng.randrange(i + 1)
                    if replacement < 256:
                        reservoir[replacement] = item
                entry = (tokens, i, item)
                if len(longest) < 32:
                    heapq.heappush(longest, entry)
                elif entry[:2] > longest[0][:2]:
                    heapq.heapreplace(longest, entry)
                n += 1
                if n % 50000 == 0:
                    print(f'Audited {n:,} rows', flush=True)
        assert counts == report['counts']
        assert thought_counts == report['nonempty_reasoning_rows']
        assert n == report['total_rows']
        assert sum(thought_counts.values()) * 2 > n
        assert max(groups.values()) <= 3
        assert sum(token_totals.values()) == report['total_tokens']
        assert h.hexdigest() == report['sha256']
        assert args.output.stat().st_size == sum(byte_totals.values()) == report['bytes']
        assert report.get('min_bytes', 1_600_000_000) <= report['bytes'] <= report.get('max_bytes', 5_000_000_000)
        for source in counts:
            assert token_totals[source] == report['token_stats'][source]['total']
            assert byte_totals[source] == report['selection_stats'][source]['bytes']
        sample = {item['index']: item for item in reservoir + [e[2] for e in longest] + list(first_by_source.values())}
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text('\n'.join(json.dumps(sample[i], ensure_ascii=False) for i in sorted(sample)) + '\n', encoding='utf-8')
        report['verification'] = {'all_rows_checked': n, 'unique_conversations': len(seen),
                                  'sha256_verified': True, 'max_tokens': max_tokens,
                                  'reasoning_fraction': sum(thought_counts.values()) / n,
                                  'source_bytes': dict(byte_totals),
                                  'max_answers_per_question_language': max(groups.values()),
                                  'hf_sample_rows': len(sample)}
        print(json.dumps(report['verification'], ensure_ascii=False, indent=2), flush=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
