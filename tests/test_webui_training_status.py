"""Regression tests for recovered WebUI training-process status."""

import ast
import json
import os
import re
import time
from pathlib import Path

import pytest


class _SessionState(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def _load_status_helpers(checkpoints_dir):
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_training_log_is_complete",
        "_training_log_is_stopped",
        "_training_progress_from_log",
        "parse_training_metrics",
        "_completed_checkpoint_exists",
        "_detached_training_status",
        "_pause_checkpoint_saved",
        "_paused_state_path",
        "_read_paused_state",
        "_training_script_from_argv",
        "_training_mode_from_argv",
        "_training_activity_from_log",
    }
    functions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "os": os,
        "re": re,
        "time": time,
        "json": json,
        "_checkpoints_dir": lambda: str(checkpoints_dir),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace


def _write_log(path, progress):
    path.write_text(progress, encoding="utf-8")
    return path


def test_detached_run_stays_running_while_process_exists(tmp_path):
    helpers = _load_status_helpers(tmp_path / "checkpoints")
    log_path = _write_log(tmp_path / "train_full_sft_run.log", "starting\n")

    assert helpers["_detached_training_status"](
        str(log_path), True, None
    ) == "running"


def test_stopped_run_does_not_become_completed_or_paused(tmp_path):
    helpers = _load_status_helpers(tmp_path / 'checkpoints')
    log_path = _write_log(tmp_path / 'train_pretrain_run.log',
        '# epochs: 1\nEpoch:[1/1](10/10), loss: 1.0\n'
        '[PAUSED] Training paused — resume checkpoint saved.\n'
        '[STOPPED] Training terminated by user. No additional checkpoint saved.\n')
    assert helpers['_detached_training_status'](str(log_path), False, None) == 'stopped'
    assert helpers['_detached_training_status'](str(log_path), True, None) == 'running'
    with log_path.open('a', encoding='utf-8') as log:
        log.write('# epochs: 2\ninitializing\n')
    assert helpers['_detached_training_status'](str(log_path), False, None) == 'running'


def test_streaming_progress_uses_tokens_not_estimated_steps(tmp_path):
    helpers = _load_status_helpers(tmp_path)
    log = 'Epoch:[1/1](400000/~465048), tokens: 11999000000/12000000000, loss: 1.2'
    progress = helpers['_training_progress_from_log'](log)
    assert progress['fraction'] == 11999000000 / 12000000000
    assert not progress['complete']
    assert 'Tokens' in progress['caption']
    assert '465,048' not in progress['caption']
    assert helpers['parse_training_metrics'](log)[0]['loss'] == 1.2


def test_streaming_token_total_alone_does_not_claim_saved_completion(tmp_path):
    helpers = _load_status_helpers(tmp_path)
    progress = helpers['_training_progress_from_log'](
        'Epoch:[2/2](105/~100), tokens: 1000/1000, loss: 1.2'
    )
    assert progress['fraction'] == 1.0
    assert not progress['complete']
    assert not progress['checkpoint_saved']


def test_streaming_multi_epoch_progress(tmp_path):
    helpers = _load_status_helpers(tmp_path)
    progress = helpers['_training_progress_from_log'](
        'Epoch:[2/3](20/~90), tokens: 250/1000, loss: 1.2'
    )
    assert progress['fraction'] == 1.25 / 3


def test_streaming_completion_after_saved_checkpoint_and_resume_reset(tmp_path):
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    helpers = _load_status_helpers(checkpoints)
    checkpoint = checkpoints / 'pretrain_run_512_moe_resume.pth'
    checkpoint.write_bytes(b'saved')
    log = (
        '# epochs: 1\n'
        'Epoch:[1/1](400000/~465048), tokens: 11999000000/11999990666, loss: 1.2\n'
        'Epoch:[1/1](400052/400052), chunks_complete: 35, epoch_time: 0.0min, '
        'checkpoint_saved: 1, Streaming epoch complete (all chunks consumed).\n'
    )
    log_path = _write_log(tmp_path / 'train_pretrain_run.log', log)
    os.utime(checkpoint, (log_path.stat().st_mtime - 1, log_path.stat().st_mtime - 1))
    assert helpers['_detached_training_status'](str(log_path), False, None) == 'success'
    assert helpers['_training_progress_from_log'](log)['fraction'] == 1.0
    metrics = helpers['parse_training_metrics'](log)
    assert len(metrics) == 2
    assert metrics[-1]['epoch_time'] == 0
    assert 'loss' not in metrics[-1]
    log_path.write_text(log + '# epochs: 2\ninitializing\n', encoding='utf-8')
    assert not helpers['_training_log_is_complete'](str(log_path))
    assert helpers['_training_progress_from_log'](log + '# epochs: 2\n') is None


def test_pretraining_activity_distinguishes_background_prefetch(tmp_path):
    helper = _load_status_helpers(tmp_path / "checkpoints")[
        "_training_activity_from_log"
    ]
    log = (
        "[Streaming Chunk] epoch=1, chunk=1/35\n"
        "[Streaming Prefetch] first training batch completed; "
        "packing chunk=2/35 in background thread\n"
        "Tokenizing pretrain for sequence buckets:  73%|progress\n"
    )

    assert helper(log) == (
        "Training warmup + background prefetch chunk 2/35: tokenizing 73%"
    )


def test_training_process_detection_requires_a_script_argv(tmp_path):
    helper = _load_status_helpers(tmp_path / "checkpoints")[
        "_training_script_from_argv"
    ]

    assert helper([
        r"E:\miniconda3\python.exe",
        "-u",
        str(Path('repo') / 'trainer' / 'train_full_sft.py'),
        "--from_resume",
        "1",
    ]) == "train_full_sft"
    assert helper([
        r"E:\miniconda3\python.exe",
        "-c",
        "print('train_full_sft.py')",
    ]) is None


def test_cpt_process_is_distinguished_from_scratch_pretraining(tmp_path):
    helper = _load_status_helpers(tmp_path / "checkpoints")[
        "_training_mode_from_argv"
    ]
    argv = [
        r"E:\miniconda3\python.exe", "-u",
        str(Path('repo') / 'trainer' / 'train_pretrain.py'),
        "--save_weight", "cpt_20260916_768",
    ]

    assert helper(argv) == "train_cpt"
    assert helper(argv[:-1] + ["pretrain_20260916_768"]) == "train_pretrain"


def test_detached_run_becomes_success_after_final_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    helpers = _load_status_helpers(checkpoints)
    log_path = _write_log(
        tmp_path / "train_full_sft_20260903_010116.log",
        "Epoch:[1/1](18401/18401), loss: 1.5830\n",
    )
    checkpoint = checkpoints / "full_sft_20260903_010116_768_resume.pth"
    checkpoint.write_bytes(b"complete")
    checkpoint_mtime = log_path.stat().st_mtime + 1
    os.utime(checkpoint, (checkpoint_mtime, checkpoint_mtime))

    assert helpers["_detached_training_status"](
        str(log_path), False, None
    ) == "success"


def test_final_log_without_new_checkpoint_is_not_success(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    helpers = _load_status_helpers(checkpoints)
    log_path = _write_log(
        tmp_path / "train_full_sft_run.log",
        "Epoch:[1/1](10/10), loss: 1.0\n",
    )
    stale_checkpoint = checkpoints / "full_sft_run_768_resume.pth"
    stale_checkpoint.write_bytes(b"periodic")
    stale_mtime = log_path.stat().st_mtime - 10
    os.utime(stale_checkpoint, (stale_mtime, stale_mtime))

    assert helpers["_detached_training_status"](
        str(log_path), False, None, now=log_path.stat().st_mtime + 61
    ) == "failed"


def test_pause_request_without_save_confirmation_is_not_paused(tmp_path):
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    (checkpoints / 'pretrain_run_512_moe_resume.pth').write_bytes(b'periodic')
    helpers = _load_status_helpers(checkpoints)
    log_path = _write_log(tmp_path / "train_pretrain_run.log", "starting\n")

    assert helpers["_detached_training_status"](
        str(log_path), False, {"train_type": "pretrain", "weight": "pretrain_run"},
        now=log_path.stat().st_mtime + 61,
    ) == "failed"


def test_saved_pause_requires_matching_checkpoint_and_latest_launch(tmp_path):
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    helpers = _load_status_helpers(checkpoints)
    log_path = _write_log(tmp_path / 'train_pretrain_run.log',
        '# epochs: 1\n[PAUSED] Training paused — resume checkpoint saved.\n')
    paused = {'train_type': 'pretrain', 'weight': 'pretrain_run'}
    status = helpers['_detached_training_status']
    now = log_path.stat().st_mtime + 61
    assert status(str(log_path), False, paused, now=now) == 'failed'
    (checkpoints / 'pretrain_other_512_moe_resume.pth').write_bytes(b'other run')
    assert status(str(log_path), False, paused, now=now) == 'failed'
    checkpoint = checkpoints / 'pretrain_run_512_moe_resume.pth'
    checkpoint.write_bytes(b'saved')
    assert status(str(log_path), False, paused, now=now) == 'paused'
    # CLI pauses also recover from the trainer's confirmation without a UI record.
    assert status(str(log_path), False, None, now=now) == 'paused'
    log_path.write_text(log_path.read_text(encoding='utf-8') + '# epochs: 2\nstarting\n',
                        encoding='utf-8')
    assert status(str(log_path), False, paused, now=log_path.stat().st_mtime + 61) == 'failed'


@pytest.mark.parametrize('record', [[], {}, {'weight': None}, {'weight': ''},
                                   {'weight': '../other'}])
def test_unidentified_legacy_pause_records_are_ignored(tmp_path, record):
    (tmp_path / '.paused.json').write_text(json.dumps(record), encoding='utf-8')
    assert _load_status_helpers(tmp_path)['_read_paused_state']() is None


def test_recovery_ignores_untracked_historical_log(tmp_path):
    state = _SessionState()
    recover = _load_recovery(state, tmp_path)
    recover()
    assert 'train_status' not in state
    assert 'train_log_path' not in state


def test_recovery_clears_stale_paused_browser_session(tmp_path):
    state = _SessionState(train_status='paused', train_log_path='old.log')
    _load_recovery(state, tmp_path)()
    assert 'train_status' not in state
    assert 'train_log_path' not in state


def test_recovery_keeps_explicitly_tracked_run(tmp_path):
    state = _SessionState(train_log_path='owned.log')
    _load_recovery(state, tmp_path)()
    assert state['train_status'] == 'failed'
    assert state['train_log_path'] == 'owned.log'


@pytest.mark.parametrize('matching_checkpoint', [True, False])
def test_pause_recovery_requires_checkpoint_for_selected_model(tmp_path, matching_checkpoint):
    state = _SessionState(train_type='pretrain')
    recover = _load_recovery(state, tmp_path)
    paused = {'train_type': 'pretrain', 'weight': 'pretrain_saved',
              'log_path': str(tmp_path / 'train_pretrain_saved.log')}
    recover.__globals__.update({
        '_read_paused_state': lambda: paused,
        'build_config_dict': lambda: {'hidden_size': 1536, 'use_moe': False},
        '_arch_tag': lambda: '', '_matches_arch_tag': lambda *_: True,
        '_resume_checkpoint_exists': lambda *_: matching_checkpoint,
        '_pause_checkpoint_saved': lambda *_: True,
        '_detached_training_status': lambda *_: 'paused',
    })
    recover()
    if matching_checkpoint:
        assert state['train_status'] == 'paused'
        assert state['save_prefix_pretrain'] == 'pretrain_saved'
        assert state['train_log_path'] == paused['log_path']
    else:
        assert 'train_status' not in state


@pytest.mark.parametrize('pause_record', [False, True])
@pytest.mark.parametrize('control', ['btn_resume_train', 'btn_resume_train_monitor'])
def test_resume_button_pins_paused_run_and_enables_resume(tmp_path, monkeypatch, pause_record, control):
    import subprocess
    from streamlit.testing.v1 import AppTest
    from scripts import config_templates

    monkeypatch.setattr(config_templates, 'TEMPLATES_FILE', tmp_path / 'templates.json')
    trainer = tmp_path / 'trainer'
    logs = trainer / 'logs'
    logs.mkdir(parents=True)
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    paused_weight = 'pretrain_20261020_010101'
    (checkpoints / f'{paused_weight}_1536_resume.pth').write_bytes(b'saved pause')
    # A newer unrelated checkpoint must never win the dedicated resume action.
    newer = checkpoints / 'pretrain_20261020_020202_1536_resume.pth'
    newer.write_bytes(b'unrelated newer run')
    os.utime(newer, (time.time() + 10, time.time() + 10))
    log_path = _write_log(logs / f'train_{paused_weight}_2.log',
        f'# command: python trainer/train_pretrain.py --save_weight {paused_weight}\n'
        '# epochs: 1\n'
        'Epoch:[1/1](100/1000), loss: 3.0, lr: 0.0005\n'
        'Epoch:[1/1](200/1000), loss: 2.8, lr: 0.0005\n'
        '[PAUSED] Training paused — resume checkpoint saved.\n')
    original_log = log_path.read_text(encoding='utf-8')
    if pause_record:
        (checkpoints / '.paused.json').write_text(json.dumps({
            'weight': paused_weight, 'train_type': 'pretrain', 'log_path': str(log_path),
        }), encoding='utf-8')
    data = tmp_path / 'data.jsonl'
    data.write_text('{"text": "test data"}\n', encoding='utf-8')
    (trainer / 'train_pretrain.py').write_text('# Only a fake launcher target\n', encoding='utf-8')
    state_file = trainer / 'webui_state.json'
    state_file.write_text(json.dumps({
        'preset': 'Instinct V2', 'from_resume': False, 'autotune_use_profile': False,
        'auto_refresh_log': False, 'shutdown_after_train': False,
        'train_type': 'pretrain', 'train_log_path': str(log_path),
        'data_path_pretrain': str(data), 'use_compile': False,
    }), encoding='utf-8')
    source_path = Path(__file__).parents[1] / 'scripts/config_webui.py'
    source = source_path.read_text(encoding='utf-8').replace(
        'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")',
        repr(str(trainer)))
    source = source.replace('\n_try_recover_training_state()\n',
        f'\n_checkpoints_dir = lambda: {str(checkpoints)!r}\n'
        '_find_running_train_process = lambda: (None, None)\n'
        '_maybe_arm_shutdown_monitor = lambda *_: None\n'
        '_try_recover_training_state()\n')
    launches = []
    original_popen = subprocess.Popen

    def launch(command, *args, **kwargs):
        if isinstance(command, list) and any(str(x).endswith('train_pretrain.py') for x in command):
            launches.append(command)
            kwargs['stdout'].write(
                'Epoch [1/1]: resume global step 201, chunk=1, chunk_step=200\n'
                'Epoch:[1/1](300/1000), loss: 2.6, lr: 0.0005\n')
            kwargs['stdout'].flush()
            return type('FakeTrainer', (), {'pid': 987654, 'poll': lambda _: None})()
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', launch)
    app = AppTest.from_string(f'__file__ = {str(source_path)!r}\n' + source,
                             default_timeout=60).run()
    assert not app.exception
    assert app.session_state['train_status'] == 'paused'
    assert not app.session_state['from_resume']
    button = app.button(key=control)
    assert button.label == '继续训练' and not button.disabled
    button.click().run()
    assert not app.exception
    assert len(launches) == 1
    command = launches[0]
    assert command[command.index('--from_resume') + 1] == '1'
    assert command[command.index('--save_weight') + 1] == paused_weight
    assert app.session_state['from_resume']
    assert app.session_state['train_status'] == 'running'
    assert Path(app.session_state['train_log_path']).resolve() == log_path.resolve()
    resumed_log = log_path.read_text(encoding='utf-8')
    assert resumed_log.startswith(original_log)
    assert resumed_log.count('# command:') == 2
    assert not (logs / f'train_{paused_weight}.log').exists()
    metrics = _load_status_helpers(checkpoints)['parse_training_metrics'](resumed_log)
    assert [row['_epoch_step'] for row in metrics] == [100, 200, 300]
    assert not any(button.key in ('btn_resume_train', 'btn_resume_train_monitor') for button in app.button)
    assert newer.read_bytes() == b'unrelated newer run'
    assert not (checkpoints / '.paused.json').exists()


def test_resume_callback_refuses_changed_target_or_live_process(tmp_path):
    source_path = Path(__file__).parents[1] / 'scripts/config_webui.py'
    function = next(n for n in ast.parse(source_path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_request_resume')
    state = _SessionState(train_type='pretrain', from_resume=False)
    namespace = {
        'st': type('FakeStreamlit', (), {'session_state': state})(),
        'build_config_dict': lambda: {},
        '_paused_resume_prefix': lambda *_: 'pretrain_saved',
        '_find_running_train_process': lambda: (12345, 'train_pretrain'),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
    namespace['_request_resume']('pretrain', 'pretrain_other')
    assert not state.get('train_triggered') and not state.from_resume
    namespace['_request_resume']('pretrain', 'pretrain_saved')
    assert not state.get('train_triggered') and not state.from_resume


def _load_recovery(session_state, tmp_path, *, process=(None, None)):
    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    module = ast.parse(source_path.read_text(encoding='utf-8'))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_try_recover_training_state')
    namespace = {
        '__file__': str(source_path), 'os': os, 're': re, 'PAUSE_EXIT_CODE': 42,
        'st': type('FakeStreamlit', (), {'session_state': session_state})(),
        '_latest_train_log': lambda _: str(tmp_path / 'historical.log'),
        '_find_running_train_process': lambda: process,
        '_read_paused_state': lambda: None,
        '_pause_checkpoint_saved': lambda *_: False,
        '_detached_training_status': lambda *_: 'failed',
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'),
         namespace)
    return namespace['_try_recover_training_state']


def test_app_does_not_show_paused_for_legacy_unidentified_record(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from scripts import config_templates

    monkeypatch.setattr(config_templates, 'TEMPLATES_FILE', tmp_path / 'templates.json')
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    (checkpoints / '.paused.json').write_text(json.dumps({
        'train_type': 'pretrain', 'paused_at': '2026-09-27 12:45:28', 'weight': None,
    }), encoding='utf-8')
    historical_log = _write_log(tmp_path / 'train_pretrain_old.log',
        '[PAUSED] Training paused — resume checkpoint saved.\n')
    state_file = tmp_path / 'webui_state.json'
    state_file.write_text(json.dumps({'preset': 'Instinct V2'}), encoding='utf-8')
    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    source = source_path.read_text(encoding='utf-8')
    source = source.replace('os.path.join(trainer_dir, "webui_state.json")', repr(str(state_file)))
    source = source.replace('\n_try_recover_training_state()\n',
        f'\n_checkpoints_dir = lambda: {str(checkpoints)!r}\n'
        f'_latest_train_log = lambda _: {str(historical_log)!r}\n'
        '_find_running_train_process = lambda: (None, None)\n'
        '_try_recover_training_state()\n')
    app = AppTest.from_string(f'__file__ = {str(source_path)!r}\n' + source,
                              default_timeout=60).run()
    assert not app.exception
    assert app.session_state['preset'] == 'Instinct V2'
    assert all('训练已暂停' not in item.value for item in app.warning)
    assert any(item.value == '就绪 · 等待启动' for item in app.sidebar.info)
    assert not next(button for button in app.button if button.key == 'btn_start_train').disabled


def test_live_process_overrides_stale_failed_session(tmp_path):
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_try_recover_training_state"
    )
    session_state = _SessionState(train_status="failed")
    namespace = {
        "__file__": str(source_path),
        "os": os,
        "st": type("FakeStreamlit", (), {"session_state": session_state})(),
        "_latest_train_log": lambda _trainer_dir: str(tmp_path / "active.log"),
        "_find_running_train_process": lambda: (60104, "train_full_sft"),
        "_read_paused_state": lambda: None,
        "_detached_training_status": lambda *_args, **_kwargs: "failed",
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )

    namespace["_try_recover_training_state"]()

    assert session_state["train_status"] == "running"
    assert session_state["train_type"] == "full_sft"
    assert session_state["train_log_path"] == str(tmp_path / "active.log")


@pytest.mark.parametrize('control', ['btn_stop_train', 'btn_stop_train_monitor'])
def test_stop_button_controls_recovered_run_and_survives_reload(tmp_path, monkeypatch, control):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from scripts import config_templates, training_control

    monkeypatch.setattr(config_templates, 'TEMPLATES_FILE', tmp_path / 'templates.json')
    calls = []

    def stop(pid, root):
        calls.append((pid, root))
        st.session_state['_test_train_running'] = False
        return True

    monkeypatch.setattr(training_control, 'stop_training_process', stop)
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    checkpoint = checkpoints / 'existing_resume.pth'
    checkpoint.write_bytes(b'previous saved progress')
    log_path = _write_log(tmp_path / 'train_pretrain_run.log', '# epochs: 1\ninitializing\n')
    state_file = tmp_path / 'webui_state.json'
    state_file.write_text(json.dumps({'preset': 'Instinct V2', 'shutdown_after_train': True,
                                     'auto_refresh_log': False,
                                     'train_log_path': str(log_path)}), encoding='utf-8')
    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    source = source_path.read_text(encoding='utf-8')
    source = source.replace('os.path.join(trainer_dir, "webui_state.json")', repr(str(state_file)))
    source = source.replace('\n_try_recover_training_state()\n',
        f'\n_checkpoints_dir = lambda: {str(checkpoints)!r}\n'
        f'_latest_train_log = lambda _: {str(log_path)!r}\n'
        '_find_running_train_process = lambda: (60104, "train_pretrain") '
        'if st.session_state.get("_test_train_running", True) else (None, None)\n'
        'def _trigger_auto_shutdown():\n'
        '    raise AssertionError("Stopping training must never schedule shutdown")\n'
        '_try_recover_training_state()\n')
    app_source = f'__file__ = {str(source_path)!r}\n' + source
    app = AppTest.from_string(app_source, default_timeout=60).run()
    assert not app.exception
    assert app.session_state['train_status'] == 'running'
    assert app.session_state['_train_pid'] == 60104
    assert next(button for button in app.button if button.key == 'btn_start_train').disabled
    assert {'btn_stop_train', 'btn_stop_train_monitor'} <= {button.key for button in app.button}
    next(button for button in app.button if button.key == control).click().run()
    assert not app.exception
    assert calls == [(60104, str(source_path.parents[1]))]
    assert app.session_state['train_status'] == 'stopped'
    assert any(item.value == '训练已终止' for item in app.sidebar.info)
    assert checkpoint.read_bytes() == b'previous saved progress'
    assert not (checkpoints / '.pause_request').exists()
    assert '[STOPPED]' in log_path.read_text(encoding='utf-8')
    assert not any(button.key in ('btn_stop_train', 'btn_stop_train_monitor') for button in app.button)
    # A new browser session must recover the explicit stop without triggering
    # success/auto-shutdown from an earlier completion line or a fresh log.
    reloaded = AppTest.from_string(app_source, default_timeout=60)
    reloaded.session_state['_test_train_running'] = False
    reloaded.run()
    assert not reloaded.exception
    assert reloaded.session_state['train_status'] == 'stopped'


@pytest.mark.parametrize('exit_code,stopped,expected', [
    (None, False, 'running'), (0, False, 'success'),
    (42, False, 'paused'), (1, False, 'failed'), (0, True, 'stopped'),
])
def test_fragment_poll_preserves_training_exit_and_shutdown_rules(exit_code, stopped, expected):
    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    module = ast.parse(source_path.read_text(encoding='utf-8'))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_refresh_training_status')
    state = _SessionState(train_status='running',
                          train_proc=type('Process', (), {'poll': lambda _: exit_code})())
    shutdowns = []
    namespace = {
        'st': type('Streamlit', (), {'session_state': state})(), 'PAUSE_EXIT_CODE': 42,
        '_training_log_is_stopped': lambda _: stopped,
        '_trigger_auto_shutdown': lambda: shutdowns.append(True),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
    assert namespace['_refresh_training_status']() == (expected != 'running')
    assert state.train_status == expected
    assert shutdowns == ([True] if expected == 'success' else [])
    assert not namespace['_refresh_training_status']()
    assert shutdowns == ([True] if expected == 'success' else [])


@pytest.mark.parametrize('exit_code', [0, 1, 42])
def test_fragment_poll_observes_pipeline_exit_once(exit_code):
    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    function = next(node for node in ast.parse(source_path.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_refresh_training_status')
    process = type('Process', (), {'returncode': None, 'poll': lambda self: self.returncode})()
    state = _SessionState(_pipeline_proc=process)
    shutdowns = []
    namespace = {'st': type('Streamlit', (), {'session_state': state})(),
                 '_trigger_auto_shutdown': lambda: shutdowns.append(True)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
    poll = namespace['_refresh_training_status']
    assert poll() and state._pipeline_running
    assert not poll()
    process.returncode = exit_code
    assert poll() and not state.get('_pipeline_running', False)
    assert not poll()
    assert shutdowns == ([True] if exit_code == 0 else [])


@pytest.mark.parametrize('remaining', [120, -1])
def test_shutdown_countdown_fragment_finishes_and_can_be_cancelled(remaining):
    from streamlit.testing.v1 import AppTest

    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    function = next(node for node in ast.parse(source_path.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_render_training_activity')
    source = (
        'import streamlit as st\nimport time\n'
        'if "_test_initialized" not in st.session_state:\n'
        '    st.session_state._test_initialized = True\n'
        f'    st.session_state.shutdown_deadline = time.time() + {remaining}\n'
        '_refresh_training_status = lambda: False\n'
        '_cancel_auto_shutdown = lambda: st.session_state.pop("shutdown_deadline", None)\n'
        + ast.unparse(function) + '\n_render_training_activity()\n'
    )
    app = AppTest.from_string(source).run()
    assert not app.exception
    if remaining > 0:
        assert any('系统将在约' in warning.value for warning in app.warning)
        app.button(key='btn_cancel_shutdown').click().run()
        assert not app.exception
    assert 'shutdown_deadline' not in app.session_state
    assert not app.warning


def test_live_monitor_finishes_runs_with_auto_refresh_and_changing_metrics(tmp_path, monkeypatch):
    """The sleep/rerun loop used to prevent AppTest (and stale-node cleanup) finishing."""
    from streamlit.testing.v1 import AppTest
    from scripts import config_templates

    monkeypatch.setattr(config_templates, 'TEMPLATES_FILE', tmp_path / 'templates.json')
    trainer_dir = tmp_path / 'trainer'
    logs = trainer_dir / 'logs'
    logs.mkdir(parents=True)
    current = _write_log(logs / 'current.log', '# epochs: 1\ninitializing\n')
    historical = _write_log(logs / 'historical.log',
                            'Epoch:[1/1](1/10) loss: 4.0 lr: 0.0003 epoch_time: 1\n')
    import builtins
    original_open = builtins.open
    historical_reads = []

    def counted_open(file, mode='r', *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == historical and 'r' in mode:
            historical_reads.append(str(file))
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', counted_open)
    state_file = trainer_dir / 'webui_state.json'
    state_file.write_text(json.dumps({'preset': 'Instinct V2', 'auto_refresh_log': True,
                                     'train_log_path': str(current)}), encoding='utf-8')
    source_path = Path(__file__).parents[1] / 'scripts' / 'config_webui.py'
    source = source_path.read_text(encoding='utf-8').replace(
        'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")',
        repr(str(trainer_dir)))
    source = source.replace('\n_try_recover_training_state()\n',
        f'\n_latest_train_log = lambda _: {str(current)!r}\n'
        '_find_running_train_process = lambda: (60104, "train_pretrain") '
        'if st.session_state.get("_test_train_running", True) else (None, None)\n'
        '_read_paused_state = lambda: None\n'
        '_trigger_auto_shutdown = lambda: None\n'
        '_try_recover_training_state()\n')
    app = AppTest.from_string(f'__file__ = {str(source_path)!r}\n' + source,
                              default_timeout=30).run()
    assert not app.exception
    assert app.session_state['train_status'] == 'running'
    assert app.checkbox(key='auto_refresh_log').value
    assert [item.value for item in app.info].count('Waiting for training metrics…') == 1
    assert [item.value for item in app.caption].count('Loss') == 1
    assert len(historical_reads) == 1

    # Even a full page rerun must not reread unchanged historical files.
    app.run()
    assert not app.exception
    assert len(historical_reads) == 1

    current.write_text('Epoch:[1/1](2/10) loss: 3.0 lr: 0.0002 epoch_time: 2\n', encoding='utf-8')
    app.run()
    assert not app.exception
    assert [item.value for item in app.caption].count('Loss') == 2
    assert all(item.value != 'Waiting for training metrics…' for item in app.info)
    assert len(historical_reads) == 1
    historical.write_text('Epoch:[1/1](2/10) loss: 2.5 lr: 0.0003 epoch_time: 1\n', encoding='utf-8')
    app.run()
    assert not app.exception
    assert len(historical_reads) == 2
    assert any('loss: 2.5' in item.value for item in app.code)
    # Clearing the log removes current charts while retaining the historical charts.
    app.button(key='btn_clear_train_log').click().run()
    assert not app.exception
    assert current.read_text(encoding='utf-8') == ''
    assert [item.value for item in app.caption].count('Loss') == 1
    app.checkbox(key='auto_refresh_log').uncheck().run()
    assert not app.exception
    assert json.loads(state_file.read_text(encoding='utf-8'))['auto_refresh_log'] is False
    app.checkbox(key='auto_refresh_log').check().run()
    assert not app.exception

    # A recovered trainer's exit must still update controls and stop polling.
    current.write_text('RuntimeError: simulated training failure\n', encoding='utf-8')
    os.utime(current, (time.time() - 120, time.time() - 120))
    app.session_state['_test_train_running'] = False
    app.run()
    assert not app.exception
    assert app.session_state['train_status'] == 'failed'
    assert any(item.value == '训练失败' for item in app.sidebar.info)
    assert not any(button.key == 'btn_stop_train_monitor' for button in app.button)
