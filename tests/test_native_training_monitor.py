"""Native run discovery/control stays CPU-only and separate from panel presets."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import native_training_monitor as monitor


@pytest.fixture
def registered(tmp_path, monkeypatch):
    resume = tmp_path / 'production_1536_resume.npz'
    progress = dict(steps=16, processed_tokens=192000, parquet_cursor=dict(group=0, batch=16))
    metadata = dict(format='instinct-cublas-v1', progress=progress,
                    optimizers=dict(model=dict(updates=2, accumulated=0)))
    np.savez(resume, __metadata__=np.asarray(json.dumps(metadata)))
    status = dict(status='paused', pid=12345, process_created=100., steps=16,
                  processed_valid_tokens=192000, updates=2, accumulated=0,
                  parquet_cursor=progress['parquet_cursor'], resume_path=str(resume))
    status_path = tmp_path / 'run.status.json'
    monitor.write_json(status_path, status)
    manifest = tmp_path / 'training_run.json'
    monitor.register(manifest, runtime_root=tmp_path, python=Path(__import__('sys').executable),
                     status_path=status_path, stdout_path=tmp_path / 'run.stdout.log',
                     stderr_path=tmp_path / 'run.stderr.log', pause_path=tmp_path / 'run.pause',
                     resume_path=resume, planned_tokens=24_000_000_000)
    monkeypatch.setattr(monitor, 'process_matches', lambda *_: False)
    return manifest, monitor.snapshot(manifest)


def test_paused_checkpoint_and_full_corpus_progress(registered):
    manifest, state = registered
    assert state['phase'] == 'paused' and not state['alive']
    assert state['record']['planned_tokens'] == 24_000_000_000
    assert monitor.verify_resume(state['record'], state['status'])['steps'] == 16
    with pytest.raises(ValueError, match='一致'):
        monitor.verify_resume(state['record'], dict(state['status'], steps=24))


def test_stale_process_is_interrupted_and_cannot_pause(registered):
    manifest, state = registered
    monitor.write_json(state['record']['status_path'], dict(state['status'], status='running'))
    assert monitor.snapshot(manifest)['phase'] == 'interrupted'
    with pytest.raises(ValueError, match='可暂停'):
        monitor.request_pause(manifest)
    assert not Path(state['record']['pause_path']).exists()


def test_pause_uses_registered_marker_not_torch_global(registered, monkeypatch):
    manifest, state = registered
    monkeypatch.setattr(monitor, 'process_matches', lambda *_: True)
    monitor.write_json(state['record']['status_path'], dict(state['status'], status='running'))
    monitor.request_pause(manifest)
    assert Path(state['record']['pause_path']).exists()
    assert not (manifest.parent / '.pause_request').exists()


def test_pid_reuse_does_not_own_run(registered, monkeypatch):
    import psutil
    _, state = registered
    monkeypatch.undo()
    monkeypatch.setattr(psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 200.))
    assert not monitor.process_matches(state['record'], state['status'])


def test_resume_uses_exact_native_entry_and_preserves_status(registered, monkeypatch):
    import psutil
    manifest, state = registered
    launches = []
    monkeypatch.setattr(psutil, 'process_iter', lambda *_: [])
    monkeypatch.setattr(psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 300.))
    def launch(command, **kwargs):
        launches.append((command, kwargs))
        # Child publishes first: parent must not replace it with "starting".
        monitor.write_json(state['record']['status_path'], dict(state['status'], status='running', pid=24680))
        return SimpleNamespace(pid=24680)
    monkeypatch.setattr(monitor.subprocess, 'Popen', launch)
    assert monitor.resume_run(manifest) == 24680
    command, options = launches[0]
    assert command[1:] == ['-u', '-m', 'experiments.run_native_1b_acceptance',
                           '--resume-path', state['record']['resume_path']]
    assert options['cwd'] == state['record']['runtime_root']
    assert options['env']['PYTHONUTF8'] == '1'
    assert json.loads(Path(state['record']['status_path']).read_text())['status'] == 'running'
    assert not manifest.with_suffix('.launch.lock').exists()


def test_launch_lock_not_removed_by_second_browser(registered):
    manifest, _ = registered
    lock = manifest.with_suffix('.launch.lock')
    lock.touch()
    with pytest.raises(FileExistsError):
        monitor.resume_run(manifest)
    assert lock.exists()


def test_resume_passes_registered_shard_directory(registered, monkeypatch):
    import psutil
    manifest, state = registered
    record = dict(state['record'], data_path=str(manifest.parent/'shards'))
    monitor.write_json(manifest, record)
    monkeypatch.setattr(psutil, 'process_iter', lambda *_: [])
    monkeypatch.setattr(psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda:300.))
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=24680)
    monkeypatch.setattr(monitor.subprocess, 'Popen', launch)
    assert monitor.resume_run(manifest) == 24680
    assert commands[0][-2:] == ['--data-path', record['data_path']]


def test_new_launch_is_visible_before_observer_writes_status(registered, monkeypatch):
    manifest, _ = registered
    monitor.write_json(manifest.with_suffix('.launch.json'),
                       dict(pid=24680, process_created=300., updated_at=301.))
    monkeypatch.setattr(monitor, 'process_matches', lambda _, status: status['pid'] == 24680)
    state = monitor.snapshot(manifest)
    assert state['alive'] and state['phase'] == 'starting'
    assert state['status']['steps'] == 16


def test_actual_monitor_app_displays_paused_native_run(registered):
    from streamlit.testing.v1 import AppTest
    manifest, state = registered
    Path(state['record']['stdout_path']).write_text('native training log', encoding='utf-8')
    source = f'''
import streamlit as st
from scripts import native_training_monitor as monitor
def read_log(path):
    from pathlib import Path
    return (Path(path).read_text(encoding='utf-8') if Path(path).exists() else ''), []
monitor.render(st, {str(manifest)!r}, read_log=read_log, render_charts=lambda _: None)
'''
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert any('已保存并暂停' in item.value for item in app.info)
    assert any('24,000,000,000' in item.proto.text for item in app.get('progress'))
    assert not app.button(key='native_resume').disabled
    assert any('native training log' in item.value for item in app.code)


def test_monitor_estimates_remaining_time_from_live_cursor(registered, monkeypatch):
    from streamlit.testing.v1 import AppTest

    manifest, state = registered
    status_path = state['record']['status_path']
    status = dict(state['status'], status='running', updated_at=200.)
    monitor.write_json(status_path, status)
    monkeypatch.setattr(monitor, 'process_matches', lambda *_: True)
    source = f'''
import streamlit as st
from scripts import native_training_monitor as monitor
monitor.render(st, {str(manifest)!r}, read_log=lambda _: ('', [{{'epoch_time': 0.5, 'loss': 1.0}}]),
               render_charts=lambda metrics: st.write(metrics))
'''
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert any('等待下一次 token 游标更新' in item.value for item in app.caption)
    assert all('epoch_time' not in item.value for item in app.json)

    monitor.write_json(status_path, dict(status, processed_valid_tokens=792000, updated_at=230.))
    for _ in range(2):
        app.run()
        assert not app.exception
        assert any('最近 30 秒平均 20,000 token/s · 预计剩余 333.3 小时' in item.value
                   for item in app.caption)

    monitor.write_json(status_path, dict(status, pid=23456, process_created=240., updated_at=250.))
    app.run()
    assert any('等待下一次 token 游标更新' in item.value for item in app.caption)


def test_config_webui_connects_external_run_without_overwriting_preset(registered, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    from scripts import config_templates
    manifest, state = registered
    checkpoint_dir = tmp_path / 'checkpoints'
    destination = checkpoint_dir / 'native_cublas' / 'training_run.json'
    monitor.write_json(destination, state['record'])
    trainer_dir = tmp_path / 'trainer'
    trainer_dir.mkdir()
    (trainer_dir / 'webui_state.json').write_text(json.dumps(dict(
        preset='Instinct V2', auto_refresh_log=False, shutdown_after_train=False)), encoding='utf-8')
    monkeypatch.setattr(config_templates, 'TEMPLATES_FILE', tmp_path / 'templates.json')
    source_path = Path(__file__).parents[1] / 'scripts/config_webui.py'
    source = source_path.read_text(encoding='utf-8').replace(
        'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")', repr(str(trainer_dir)))
    source = source.replace('\n_try_recover_training_state()\n',
        f'\n_checkpoints_dir = lambda: {str(checkpoint_dir)!r}\n'
        '_find_running_train_process = lambda: (None, None)\n'
        '_try_recover_training_state()\n')
    app = AppTest.from_string(f'__file__ = {str(source_path)!r}\n' + source, default_timeout=30).run()
    assert not app.exception
    assert any('已保存并暂停' in item.value for item in app.info)
    assert not app.button(key='native_resume').disabled
    assert app.session_state['preset'] == 'Instinct V2'
