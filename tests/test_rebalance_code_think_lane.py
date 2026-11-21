import json

import pytest

from scripts.data_builder.rebalance_code_think_lane import (
    allocate, main, parse_buckets, rejection, select,
)

CODE = '```python\nprint(1)\n```'


def words(text):  # stand-in tokenizer: one token per whitespace-separated word
    return len(text.split())


def make_row(index, think_tokens, difficulty='easy', question=None, answer=None):
    # distinct words so the repetition filter does not fire
    reasoning = ' '.join(f'w{index}x{i}' for i in range(think_tokens))
    return {
        'metadata': {'difficulty': difficulty},
        'conversations': [
            {'role': 'user', 'content': question or f'solve problem {index}'},
            {'role': 'assistant', 'content': answer or f'done\n{CODE}', 'reasoning_content': reasoning},
        ],
    }


BUCKETS = parse_buckets('1-10:0.5,11-30:0.5')


def test_parse_buckets_normalizes_shares_and_rejects_bad_specs():
    assert [round(b['share'], 3) for b in parse_buckets('0-5:1,6-10:3')] == [0.25, 0.75]
    for bad in ('0-5', '5-0:1', '0-5:0', 'a-b:1'):
        with pytest.raises(ValueError):
            parse_buckets(bad)


def test_rows_are_selected_not_truncated_and_top_bucket_is_a_hard_cap():
    rows = [make_row(i, 5) for i in range(10)] + [make_row(100 + i, 20) for i in range(10)]
    rows.append(make_row(999, 500))  # above the top bucket
    selected, report = select(rows, words, BUCKETS, 10)
    assert report['dropped']['above_top_bucket'] == 1
    assert len(selected) == 10
    originals = {r['conversations'][0]['content']: r for r in rows}
    for row in selected:  # byte-for-byte untouched
        assert row == originals[row['conversations'][0]['content']]
    assert report['think_tokens']['max'] <= 30
    assert report['share_le_256'] == 1.0


def test_quota_follows_shares_and_reports_shortfall_with_redistribution():
    rows = [make_row(i, 5) for i in range(2)] + [make_row(100 + i, 20) for i in range(50)]
    selected, report = select(rows, words, BUCKETS, 20)
    short, long_ = report['buckets']
    assert short['available'] == 2 and short['selected'] == 2 and short['shortfall'] == 8
    # the other bucket absorbs part of the shortfall, but only up to 1.5x its quota (10 -> 15)
    assert long_['selected'] == 15
    assert len(selected) == 17 < 20
    assert any('short-think shortfall' in note for note in report['notes'])
    assert any('cannot satisfy the profile' in note for note in report['notes'])


def test_skewed_pool_cannot_flip_the_length_profile():
    # 300 short rows but almost no long ones: without the cap the short bucket would grow to ~85%
    rows = [make_row(i, 5) for i in range(300)] + [make_row(1000 + i, 20) for i in range(3)]
    _, report = select(rows, words, BUCKETS, 40)
    short, long_ = report['buckets']
    assert short['selected'] <= 30            # ceil(0.5 * 40 * 1.5)
    assert long_['selected'] == 3
    assert short['final_share'] < 0.95
    _, loose = select(rows, words, BUCKETS, 40, max_overshoot=100)
    assert loose['buckets'][0]['selected'] == 37  # the old behaviour, now opt-in


def test_allocate_handles_scarce_pool_without_overshooting():
    take = allocate(BUCKETS, {'1-10': 1, '11-30': 2}, 20)
    assert take == {'1-10': 1, '11-30': 2}
    take = allocate(BUCKETS, {'1-10': 100, '11-30': 100}, 7)
    assert sum(take.values()) == 7


def test_rejections():
    assert rejection(make_row(1, 5), True) is None
    bad_tag = make_row(2, 5)
    bad_tag['conversations'][1]['reasoning_content'] += ' </think>'
    assert rejection(bad_tag, True) == 'think_tag_in_text'
    assert rejection(make_row(3, 5, answer='no fence here'), True) == 'no_code_block'
    assert rejection(make_row(3, 5, answer='no fence here'), False) is None
    empty = make_row(4, 5)
    empty['conversations'][1]['reasoning_content'] = '   '
    assert rejection(empty, True) == 'no_reasoning'
    loop = make_row(5, 5)
    loop['conversations'][1]['reasoning_content'] = 'check the sample input again now ' * 20
    assert rejection(loop, True) == 'reasoning_repetition'
    assert rejection({'conversations': []}, True) == 'invalid'


def test_max_per_question_and_difficulty_round_robin():
    same = [make_row(i, 5, question='same question') for i in range(4)]
    _, report = select(same, words, BUCKETS, 10, max_per_question=1)
    assert report['selected_rows'] == 1 and report['dropped']['question_over_cap'] == 3
    rows = [make_row(i, 5, 'easy') for i in range(20)] + [make_row(100 + i, 5, 'hard') for i in range(4)]
    one_bucket = parse_buckets('1-10:1')
    selected, _ = select(rows, words, one_bucket, 8, difficulty_field='metadata.difficulty')
    levels = [r['metadata']['difficulty'] for r in selected]
    assert levels.count('hard') == 4 and levels.count('easy') == 4  # not 8 easy + 0 hard


def test_selection_is_deterministic_per_seed():
    rows = [make_row(i, 5 + i % 20) for i in range(60)]
    first, _ = select(rows, words, BUCKETS, 20, seed=1)
    again, _ = select(rows, words, BUCKETS, 20, seed=1)
    other, _ = select(rows, words, BUCKETS, 20, seed=2)
    assert first == again and first != other


def test_cli_writes_rows_and_report(tmp_path, capsys):
    source = tmp_path / 'in.jsonl'
    source.write_text('\n'.join(json.dumps(make_row(i, 5 + i % 25)) for i in range(40)), encoding='utf-8')
    out = tmp_path / 'lane.jsonl'
    report = main(['--input', str(source), '--output', str(out), '--target-rows', '12',
                   '--buckets', '1-10:0.5,11-30:0.5'], count_tokens=words)
    written = [json.loads(line) for line in out.read_text('utf-8').splitlines()]
    assert len(written) == report['selected_rows'] == 12
    saved = json.loads((tmp_path / 'lane.jsonl.report.json').read_text('utf-8'))
    assert saved['args']['target_rows'] == 12
    assert 'selected 12/12' in capsys.readouterr().out
