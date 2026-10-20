"""Command parity, profile application and detached WebUI job lifecycle."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import autotune_panel as panel


def test_snapshot_preserves_training_arguments_and_isolates_output(tmp_path):
    argv = [sys.executable, 'trainer/train_pretrain.py', '--from_weight', 'none',
            '--data_path', 'dataset/a file.parquet', '--dtype', 'bfloat16',
            '--epochs', '2', '--learning_rate', '0.0005', '--use_grad_checkpoint', '1',
            '--bucket_token_budget', '16384', '--save_weight', 'formal', '--from_resume', '0']
    spec = panel.make_spec(argv, tmp_path / 'model.json', 'audit.json')
    assert spec['args']['data_path'] == 'dataset/a file.parquet'
    assert spec['args']['learning_rate'] == .0005
    assert spec['args']['bucket_token_budget'] == 16384
    assert spec['args']['bucket_gpu_memory_gb'] == 'auto'
    assert spec['args']['config_path'] == str(tmp_path / 'model.json')
    assert 'save_weight' not in spec['args'] and 'from_resume' not in spec['args']
    assert 'search' not in spec


def test_snapshot_does_not_require_or_invent_reports(tmp_path):
    spec = panel.make_spec(['--from_weight', 'none', '--data_path', 'data.parquet'],
                           tmp_path / 'model.json')
    assert spec['args']['data_path'] == 'data.parquet'
    assert 'audit' not in spec and 'provenance' not in spec


@pytest.mark.parametrize('from_weight,resume', [('out/pretrain.pth', '0'), ('none', '1')])
def test_no_existing_weight_calibration(from_weight, resume):
    with pytest.raises(ValueError):
        panel.make_spec(['--from_weight', from_weight, '--from_resume', resume], 'model.json', 'audit.json')


def test_optional_target_is_not_a_machine_default():
    command = panel.tuning_command('spec.json', 'output', {})
    assert '--target-tokens-per-second' not in command
    assert '--peak-tflops' not in command
    assert '--max-trials' in command
    assert '--target-tokens-per-second' in panel.tuning_command('s', 'o', {'autotune_target_tokens_per_second': 25081})
    assert '--plan-only' in panel.tuning_command('s', 'o', {}, plan_only=True)


def test_webui_removes_saved_deadlines_and_launches_plan_without_time_limits(tmp_path):
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st
        from scripts.autotune_panel import render
        render(st, {}, lambda *a: {}, False)

    old_state = dict(autotune_time_limit=1, autotune_trial_timeout=1, autotune_baseline_timeout=1)
    view = AppTest.from_function(app)
    for key, value in old_state.items():
        view.session_state[key] = value
    view.run()
    assert not view.exception
    assert not any(item.key in old_state for item in view.number_input)
    assert all(key not in view.session_state for key in old_state)
    view.number_input(key='autotune_max_trials').set_value(12).run()
    assert not view.exception
    assert any('--max-trials 12' in item.value for item in view.code)
    assert all('--time-limit' not in item.value and '-timeout' not in item.value for item in view.code)
    spec = tmp_path / 'spec.json'
    panel.write_json(spec, {'args': {'from_weight': 'none', 'sequence_packing': 1,
                                   'sequence_packing_mode': 'bucket'}})
    state = dict(old_state, autotune_max_trials=view.session_state['autotune_max_trials'])
    command = panel.tuning_command(spec, tmp_path, state, plan_only=True)
    assert not {'--time-limit', '--trial-timeout', '--baseline-timeout'}.intersection(command)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            encoding='utf-8', timeout=20, env=dict(os.environ, PYTHONUTF8='1'))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['max_trials'] == 12
    assert plan['time_limit_seconds'] is None
    assert plan['saves_formal_weights'] is False


def test_profile_removes_performance_conflicts_preserves_math(tmp_path, monkeypatch):
    from trainer import autotune_config
    monkeypatch.setattr(autotune_config, 'source_fingerprint', lambda: 'current')
    profile = tmp_path / 'profile.json'
    panel.write_json(profile, dict(schema=1, status='validated', hardware={'source': 'current'},
        settings={'args': {'bucket_token_budget': 16384, 'use_grad_checkpoint': 2,
                           'bucket_gpu_memory_gb': 15.89, 'use_compile': 1}, 'env': {}}))
    argv = ['python', 'train_pretrain.py', '--use_grad_checkpoint', '1', '--bucket_token_budget', '8192',
            '--bucket_gpu_memory_gb', '16', '--use_compile', '0', '--dtype', 'bfloat16',
            '--from_resume', '1', '--save_weight', 'keep_checkpoint']
    cmd = panel.profile_command(argv, profile, 'pretrain', {})
    assert '--use_grad_checkpoint' not in cmd and '--bucket_token_budget' not in cmd
    assert '--bucket_gpu_memory_gb' not in cmd and '--use_compile' not in cmd
    assert panel.cli_values(cmd)['dtype'] == 'bfloat16'
    assert panel.cli_values(cmd)['from_resume'] == 1
    assert panel.cli_values(cmd)['save_weight'] == 'keep_checkpoint'
    assert cmd[-2:] == ['--autotune_profile', str(profile)]
    monkeypatch.setattr(autotune_config, 'source_fingerprint', lambda: 'changed')
    with pytest.raises(ValueError, match='训练代码'):
        panel.profile_command(argv, profile, 'pretrain', {})


def test_unvalidated_and_other_trainer_profiles_rejected(tmp_path):
    profile = tmp_path / 'profile.json'
    panel.write_json(profile, {'schema': 1, 'status': 'running'})
    with pytest.raises(ValueError, match='验收'):
        panel.profile_command([], profile, 'pretrain', {})
    with pytest.raises(ValueError, match='预训练'):
        panel.profile_command([], profile, 'full_sft', {})


def test_manual_profile_loads_without_fabricated_measurements(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from trainer import autotune_config

    monkeypatch.setattr(autotune_config, 'source_fingerprint', lambda: 'current')
    profile = tmp_path / 'manual' / 'profile.json'
    profile.parent.mkdir()
    panel.write_json(profile, dict(schema=1, status='manual', hardware={'source': 'current'},
        validation={'performed': False}, settings={
            'args': {'bucket_token_budget': 5344},
            'env': {'INSTINCT_MUON_WORKSPACE_MB': '256', 'INSTINCT_MUON_BATCH_SIZE': '16'},
        }))
    command = panel.profile_command(['python', 'train_pretrain.py',
        '--muon_workspace_mb', '256', '--muon_batch_size', '16'], profile, 'pretrain', {})
    assert '--muon_workspace_mb' not in command and '--muon_batch_size' not in command
    assert command[-2:] == ['--autotune_profile', str(profile)]

    def app():
        import streamlit as st
        from scripts.autotune_panel import render
        render(st, {}, lambda *a: {}, False)

    view = AppTest.from_function(app)
    view.session_state['autotune_output'] = str(tmp_path / 'other_cache')
    view.session_state['autotune_profile_path'] = str(profile)
    view.session_state['autotune_use_profile'] = True
    view.run()
    assert not view.exception
    assert any('手动固定配置' in item.value for item in view.info)
    assert not view.error
    assert not any('tokens/s' in item.label for item in view.metric)


def test_pipeline_keeps_pretrain_profile_without_leaking_into_sft(tmp_path, monkeypatch):
    from scripts.pipeline_panel import snapshot
    from trainer import autotune_config
    monkeypatch.setattr(autotune_config, 'source_fingerprint', lambda: 'current')
    profile = tmp_path / 'profile.json'
    panel.write_json(profile, dict(schema=1, status='validated', hardware={'source': 'current'},
        settings={'args': {'bucket_token_budget': 16384, 'use_grad_checkpoint': 2,
                           'bucket_gpu_memory_gb': 15.89, 'use_compile': 1}, 'env': {}}))
    state = dict(data_path_pretrain='data.parquet', data_path_full_sft='sft.jsonl',
                 autotune_use_profile=True, autotune_profile_path=str(profile),
                 sequence_packing=True, sequence_packing_mode='bucket', dataset_streaming='on')
    model = dict(hidden_size=512, num_hidden_layers=32, use_moe=True)
    args = snapshot(state, model, 'pretrain')['args']
    assert args['autotune_profile'] == str(profile)
    assert args['num_workers'] == 0
    assert 'bucket_gpu_memory_gb' not in args and 'use_grad_checkpoint' not in args
    assert 'autotune_profile' not in snapshot(state, model, 'full_sft')['args']


def test_changed_training_contract_rejected_before_launch(tmp_path, monkeypatch):
    from trainer import autotune_config
    monkeypatch.setattr(autotune_config, 'source_fingerprint', lambda: 'current')
    profile = tmp_path / 'profile.json'
    panel.write_json(profile, dict(schema=1, status='validated', hardware={'source': 'current'},
        settings={'args': {'use_grad_checkpoint': 1}, 'env': {}},
        contract={'arguments': {'param_dtype': 'fp32'}, 'model': {'num_hidden_layers': 32}}))
    with pytest.raises(ValueError, match='param_dtype'):
        panel.profile_command(['--param_dtype', 'bf16'], profile, 'pretrain', {'num_hidden_layers': 32})
    with pytest.raises(ValueError, match='num_hidden_layers'):
        panel.profile_command(['--param_dtype', 'fp32'], profile, 'pretrain', {'num_hidden_layers': 20})


@pytest.mark.parametrize('change', [dict(train_type='full_sft'), dict(from_resume=True),
                                  dict(use_grad_checkpoint=0), dict(sequence_packing=False),
                                  dict(sequence_packing_mode='fixed'), dict(dataset_streaming='off')])
def test_unsupported_pipeline_explained(change):
    state = dict(train_type='pretrain', use_grad_checkpoint=1, sequence_packing=True,
                 sequence_packing_mode='bucket', dataset_streaming='on', data_path_pretrain='a')
    assert panel.eligibility(state, {}) is None
    assert panel.eligibility(dict(state, **change), {})


def test_pid_reuse_is_not_recovered_as_owned_job(tmp_path, monkeypatch):
    import psutil
    panel.write_json(tmp_path / 'run.json', dict(status='running', pid=123, create_time=1.))
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda: 2., cmdline=lambda: ['autotune_webui_worker.py']))
    assert panel.run_status(tmp_path)['status'] == 'interrupted'
    assert not panel.active({'autotune_run_dir': str(tmp_path)})


def test_trial_log_follows_last_launch_including_paths_with_spaces(tmp_path):
    probe = tmp_path / 'probe' / 'trial.log'
    trial = tmp_path / 'cache with spaces' / 'candidate' / 'trial.log'
    controller = (f'[Autotune] first-probe: {{}}; log={probe}\r\n'
                  '[Autotune] probe, useful tokens/s=None\r\n'
                  f'[Autotune] candidate: {{}}; log={trial}\r\n')
    assert panel.current_trial_log(controller) == trial
    assert panel.current_trial_log('waiting for environment\n') is None


def test_log_tail_is_bounded_and_preserves_recent_unicode(tmp_path):
    log = tmp_path / 'trial.log'
    log.write_text('old line\n' * 10000 + '编译完成\n', encoding='utf-8')
    tail = panel.read_log_tail(log, limit=128)
    assert len(tail.encode('utf-8')) <= 128
    assert tail.splitlines()[-1] == '编译完成'
    assert '\ufffd' not in tail


def test_trial_log_contents_follow_running_job_and_candidate_changes(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st
        from scripts.autotune_panel import render
        render(st, {}, lambda *a: {}, False)

    monkeypatch.setattr(panel, 'run_status', lambda run_dir:
                        {'status': 'running'} if run_dir else {})
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    probe = tmp_path / 'probe' / 'trial.log'
    probe.parent.mkdir()
    probe.write_text('old probe contents\n', encoding='utf-8')
    trial = tmp_path / 'candidate one' / 'trial.log'
    trial.parent.mkdir()
    trial.write_text('SingleProcess AUTOTUNE benchmarking\n', encoding='utf-8')
    panel.write_json(trial.parent / 'result.json', {'status': 'running'})
    panel.write_json(trial.parent / 'memory.json', {
        'trial_peak_reserved_gib': 9., 'samples': [{
            'phase': 'initial_validation_after_cache_release',
            'allocated_gib': 2., 'reserved_gib': 3., 'unallocated_reserved_gib': 1.,
            'inactive_split_gib': .25, 'released_reserved_gib': 5.,
        }],
    })
    controller = run_dir / 'controller.log'
    controller.write_text(f'[Autotune] probe: {{}}; log={probe}\n'
                          f'[Autotune] candidate: {{}}; log={trial}\n', encoding='utf-8')
    view = AppTest.from_function(app)
    view.session_state['autotune_run_dir'] = str(run_dir)
    view.session_state['autotune_output'] = str(tmp_path)
    view.run()
    assert not view.exception
    assert any('SingleProcess AUTOTUNE' in item.value for item in view.code)
    assert all('old probe contents' not in item.value for item in view.code)
    assert any('每 2 秒自动刷新' in item.value for item in view.caption)
    assert next(item for item in view.metric if item.label == '实际分配').value == '2.00 GiB'
    assert next(item for item in view.metric if item.label == '全程预留峰值').value == '9.00 GiB'
    assert view.button(key='btn_autotune_stop')

    trial.write_text('Epoch:[1/1](20/60), loss: 2.1\n', encoding='utf-8')
    view.button(key='btn_autotune_progress').click().run()
    assert not view.exception
    assert any('Epoch:[1/1](20/60)' in item.value for item in view.code)

    next_trial = tmp_path / 'candidate two' / 'trial.log'
    next_trial.parent.mkdir()
    # The controller announces a candidate before its log file is created.
    with controller.open('a', encoding='utf-8') as handle:
        handle.write(f'[Autotune] next: {{}}; log={next_trial}\n')
    view.run()
    assert not view.exception
    assert any('等待子进程写入日志' in item.value for item in view.info)
    assert all('Epoch:[1/1](20/60)' not in item.value for item in view.code)
    next_trial.write_text('Traceback: simulated failure\n', encoding='utf-8')
    panel.write_json(next_trial.parent / 'result.json',
                     {'status': 'process_error', 'error': 'simulated failure'})
    view.run()
    assert not view.exception
    assert any('Traceback: simulated failure' in item.value for item in view.code)
    assert any('simulated failure' in item.value for item in view.error)


def test_supervisor_runs_and_records_success_without_training(tmp_path):
    panel.write_json(tmp_path / 'request.json', {'command': [sys.executable, '-c', 'print("search plan")']})
    # Run in a subdirectory to accommodate the local root-directory sharing lock.
    result = subprocess.run([sys.executable, str(panel.ROOT / 'scripts/autotune_webui_worker.py'),
                             '--run-dir', str(tmp_path)], cwd=tmp_path, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert panel.run_status(tmp_path)['status'] == 'success'
    assert 'search plan' in (tmp_path / 'controller.log').read_text()


def test_supervisor_cancel_reaps_owned_tree(tmp_path):
    import psutil
    import time
    marker = tmp_path / 'child.json'
    code = ('import subprocess,sys,time,json,os; from pathlib import Path; '
            'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
            f'Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)')
    panel.write_json(tmp_path / 'request.json', {'command': [sys.executable, '-c', code]})
    process = subprocess.Popen([sys.executable, str(panel.ROOT / 'scripts/autotune_webui_worker.py'),
                               '--run-dir', str(tmp_path)], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        until = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < until:
            time.sleep(.1)
        assert marker.exists()
        pids = panel.read_json(marker)
        (tmp_path / 'stop').touch()
        _, stderr = process.communicate(timeout=25)
        assert process.returncode == 0, stderr.decode(errors='replace')
        assert panel.run_status(tmp_path)['status'] == 'cancelled'
        assert all(not psutil.pid_exists(pid) for pid in pids)
    finally:
        (tmp_path / 'stop').touch()
        process.wait(timeout=25)


def test_recovery_ignores_autotune_training_children(monkeypatch):
    import psutil
    source = panel.ROOT / 'scripts/config_webui.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'_training_script_from_argv', '_training_mode_from_argv', '_find_running_train_process'}
    import re
    namespace = dict(os=os, sys=sys, re=re)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), str(source), 'exec'), namespace)
    processes = [SimpleNamespace(info=dict(pid=1, name='python.exe', cmdline=['python', 'train_pretrain.py', '--autotune_trial', 'scratch/request.json'])),
                 SimpleNamespace(info=dict(pid=2, name='python.exe', cmdline=['python', 'train_full_sft.py']))]
    monkeypatch.setattr(psutil, 'process_iter', lambda fields: processes)
    assert namespace['_find_running_train_process']() == (2, 'train_full_sft')
