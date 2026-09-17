import json

import pytest

from eval_pass_k import calculate, parse_k
from scripts.eval_webui_utils import build_command


def test_humaneval_macro_average_and_insufficient_samples(tmp_path):
    path = tmp_path / 'results.jsonl'
    rows = ([{'task_id': 'a', 'passed': x} for x in [True, False, False]]
            + [{'task_id': 'b', 'passed': x} for x in [True, True]])
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    report = calculate(path, [1, 2, 3])
    assert report['metrics']['pass@1'] == pytest.approx(2 / 3)
    assert report['metrics']['pass@2'] == pytest.approx(5 / 6)
    assert report['skipped_k'] == [3]
    assert report['num_tasks'] == 2


def test_livecodebench_graded_list(tmp_path):
    path = tmp_path / 'output_eval_all.json'
    path.write_text(json.dumps([{'question_id': 'a', 'graded_list': [False, True]},
                                {'question_id': 'b', 'graded_list': [False, False]}]))
    assert calculate(path, [1, 2])['metrics'] == {'pass@1': .25, 'pass@2': .5}


def test_reject_ungraded_samples_and_string_booleans(tmp_path):
    path = tmp_path / 'results.jsonl'
    for row in [{'task_id': 'a', 'completion': 'pass'}, {'task_id': 'a', 'passed': 'false'}]:
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError, match='passed'):
            calculate(path, [1])


def test_parse_and_webui_k_validation(tmp_path):
    assert parse_k('10, 1, 5, 1') == [1, 5, 10]
    with pytest.raises(ValueError):
        parse_k('0')
    config = dict(benchmark='LiveCodeBench', mode='all', k='1,5', num_samples=1,
                  output=str(tmp_path / 'new.json'))
    with pytest.raises(ValueError, match='至少 5'):
        build_command(config)
    config.update(num_samples=5, temperature=.8)
    command = build_command(config)
    index = command.index('--lcb_k')
    assert command[index + 1:index + 3] == ['1', '5']


def test_ui_computes_metrics_from_file(tmp_path):
    from streamlit.testing.v1 import AppTest
    from scripts.eval_webui_utils import ROOT
    path = tmp_path / 'results.jsonl'
    path.write_text('\n'.join(json.dumps({'task_id': 'a', 'passed': value})
                              for value in [True, False]))
    app = AppTest.from_file(str(ROOT / 'scripts/eval_webui.py'), default_timeout=15).run()
    app.selectbox(key='pass_k_source').select('自定义…').run()
    app.text_input(key='pass_k_source_custom').set_value(str(path))
    app.selectbox(key='pass_k_values').select('1,2,10')
    next(button for button in app.button if button.label == '计算 pass@K').click().run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric} == {'pass@1': '50.00%', 'pass@2': '100.00%'}
    assert any('样本不足' in warning.value for warning in app.warning)
