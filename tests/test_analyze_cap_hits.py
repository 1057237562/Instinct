import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))

import analyze_cap_hits as cap  # noqa: E402

CODE = '```python\nprint(1)\n```'


def one(text, **extra):
    return cap.analyze({'question_id': 'q', 'raw_response': text, **extra})


def test_done_with_code_when_closed_and_not_capped():
    result = one(f'reasoning here</think>\n{CODE}', finish_reason='stop')
    assert result['category'] == 'done_with_code' and not result['capped']


def test_capped_after_answer_means_eos_problem():
    result = one(f'short thought</think>\n{CODE}\nand then more text', finish_reason='length')
    assert result['category'] == 'capped_after_answer'


def test_repeating_tail_is_a_loop():
    text = 'Let me think. ' + 'Wait, check the constraints again. ' * 40
    assert cap.tail_period(text) == len('Wait, check the constraints again. ')
    result = one(text, finish_reason='length')
    assert result['category'] == 'capped_loop' and result['loop']


def test_code_drafted_inside_think_but_never_closed():
    text = f'First idea.\n{CODE}\nHmm, but maybe a different approach is better, so'
    result = one(text, finish_reason='length')
    assert result['category'] == 'capped_code_drafted_in_think'


def test_cut_in_open_fence_and_still_thinking():
    cut = one('Plan.\n```python\ndef f(x):\n    return x', finish_reason='length')
    assert cut['category'] == 'capped_cut_in_code'
    still = one('We need to consider many cases. ' + ' '.join(f'case{i} differs from case{i + 1}' for i in range(60)),
                finish_reason='length')
    assert still['category'] == 'capped_still_thinking' and not still['loop']


def test_recorded_context_limit_flag_from_lcb_harness():
    # eval/livecodebench/<run>/generate.py writes reached_context_limit per task.
    result = cap.analyze({'question_id': 'q', 'raw_response': 'still going', 'reached_context_limit': True})
    assert result['capped'] and result['cap_rule'] == 'recorded'
    assert result['category'] == 'capped_still_thinking'
    done = cap.analyze({'question_id': 'q', 'raw_response': CODE, 'reached_context_limit': False})
    assert not done['capped']


def test_capped_decided_by_token_count_and_fallback_heuristic():
    by_tokens = cap.analyze({'question_id': 'q', 'raw_response': 'x ' * 50, 'n_tokens': 4000}, cap_tokens=4000)
    assert by_tokens['capped'] and by_tokens['cap_rule'] == 'token_count'
    not_capped = cap.analyze({'question_id': 'q', 'raw_response': 'x ' * 50, 'n_tokens': 3999}, cap_tokens=4000)
    assert not not_capped['capped']
    likely = one('still reasoning without any end')
    assert likely['capped'] and likely['cap_rule'] == 'likely'


def test_output_list_fallback_and_cli_grouping(tmp_path, capsys):
    outputs = tmp_path / 'out.jsonl'
    rows = [
        {'question_id': 'a', 'output_list': [f'ok</think>\n{CODE}'], 'finish_reason': 'stop'},
        {'question_id': 'b', 'output_list': ['Wait, again. ' * 60], 'finish_reason': 'length'},
        {'question_id': 'c', 'output_list': ['Wait, again. ' * 60], 'finish_reason': 'length'},
    ]
    outputs.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    meta = tmp_path / 'meta.jsonl'
    meta.write_text('\n'.join(json.dumps(x) for x in (
        {'question_id': 'a', 'difficulty': 'easy', 'platform': 'leetcode'},
        {'question_id': 'b', 'difficulty': 'hard', 'platform': 'atcoder'},
        {'question_id': 'c', 'difficulty': 'hard', 'platform': 'atcoder'})), encoding='utf-8')
    report = cap.main(['--outputs', str(outputs), '--meta', str(meta), '--examples', '1',
                       '--json', str(tmp_path / 'r.json')])
    assert report['overall']['capped'] == 2
    assert report['by_difficulty']['hard']['capped'] == 2
    assert report['by_difficulty']['hard']['categories']['capped_loop'] == 2
    assert report['by_difficulty']['easy']['capped'] == 0
    assert report['warning'] is None
    assert json.loads((tmp_path / 'r.json').read_text('utf-8'))['tasks'] == 3
    assert 'capped_loop' in report['examples']
    assert 'hard' in capsys.readouterr().out
