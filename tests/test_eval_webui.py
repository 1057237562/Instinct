import json
from pathlib import Path
import sys
import time

import pytest
from streamlit.testing.v1 import AppTest

from scripts.eval_webui_utils import JobManager, ROOT, build_command, history, tail_log


def config(tmp_path, benchmark='HumanEval', **overrides):
    return dict(benchmark=benchmark, mode='generate', output=str(tmp_path / 'answers.jsonl'),
                temperature=0.8, num_samples=1, **overrides)


def test_humaneval_scoring_omits_model(tmp_path):
    output = tmp_path / 'answers.jsonl'
    output.write_text('')
    c = config(tmp_path)
    c.update(mode='evaluate', load_from='model', k='1, 10', problem_file='local.jsonl')
    command = build_command(c)
    assert '--load_from' not in command
    assert command[command.index('--k') + 1:] == ['1', '10']
    assert command[command.index('--problem_file') + 1] == 'local.jsonl'


def test_lcb_flags_and_constraints(tmp_path):
    c = config(tmp_path, 'LiveCodeBench', release_version='release_v6', resume=True)
    c['mode'] = 'all'
    command = build_command(c)
    assert '--lcb_evaluate' in command
    assert '--lcb_resume' in command
    assert '--mode' not in command
    c['limit'] = 3
    with pytest.raises(ValueError, match='limit'):
        build_command(c)


@pytest.mark.parametrize('benchmark,flag', [('HumanEval', '--batch_size'), ('LiveCodeBench', '--lcb_batch_size')])
def test_batch_size_forwarding(tmp_path, benchmark, flag):
    command = build_command(config(tmp_path, benchmark, batch_size=8))
    assert command[command.index(flag) + 1] == '8'


def test_eval_kv_cache_policy_forwarding(tmp_path):
    command = build_command(config(tmp_path, 'HumanEval', eval_kv_cache_dtype='configured'))
    assert command[command.index('--eval_kv_cache_dtype') + 1] == 'configured'


def test_no_shell_and_no_overwrite(tmp_path):
    c = config(tmp_path, weight='weight with spaces; echo test', load_from='model')
    command = build_command(c)
    assert c['weight'] in command
    Path(c['output']).write_text('')
    with pytest.raises(ValueError, match='已存在'):
        build_command(c)


def test_jobs_capture_logs_status_and_stop(tmp_path):
    manager = JobManager(tmp_path)
    manager.start([sys.executable, '-u', '-c', 'print("hello")'], {'benchmark': 'HumanEval'})
    manager.process.wait(timeout=10)
    row = manager.status()
    assert row['status'] == 'completed'
    assert 'hello' in tail_log(Path(row['directory']) / 'run.log')
    assert history(tmp_path)[0]['exit_code'] == 0
    manager.start([sys.executable, '-u', '-c', 'import time; time.sleep(60)'], {'benchmark': 'HumanEval'})
    with pytest.raises(ValueError, match='运行中'):
        manager.start([], {'benchmark': 'HumanEval'})
    manager.stop()
    assert manager.status()['status'] == 'stopped'
    assert manager.process.poll() is not None


def test_automatic_input(tmp_path):
    manager = JobManager(tmp_path)
    manager.start([sys.executable, '-u', '-c', 'print("mode="+input())'], {'benchmark': 'ToolCall'})
    manager.process.wait(timeout=10)
    assert 'mode=0' in tail_log(Path(manager.status()['directory']) / 'run.log')


@pytest.mark.parametrize('benchmark', ['HumanEval', 'LiveCodeBench', '推理自动测试', 'ToolCall'])
def test_ui_renders_each_benchmark(benchmark):
    app = AppTest.from_file(str(ROOT / 'scripts' / 'eval_webui.py'), default_timeout=15).run()
    app.sidebar.selectbox[0].select(benchmark).run()
    assert not app.exception
    assert app.title[0].value == 'Instinct 评测工作台'
    if benchmark in ('HumanEval', 'LiveCodeBench'):
        app.radio[0].set_value('evaluate').run()
        assert not app.exception
        assert not any(widget.label == '模型路径' for widget in app.text_input)
