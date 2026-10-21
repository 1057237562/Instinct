"""ETA regressions: startup cost, changing throughput, resume and GPU timing."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from trainer.training_eta import RecentThroughputETA


def test_startup_is_excluded_and_recent_rates_are_weighted():
    eta = RecentThroughputETA()
    assert eta.update(100, 900, 1000) is None
    assert eta.update(200, 910, 1000) == pytest.approx(800 / 10 / 60)
    assert eta.update(500, 930, 1000) == pytest.approx(500 / (400 / 30) / 60)
    eta.update(600, 940, 1000)
    assert eta.update(900, 950, 1000) == pytest.approx(100 / (700 / 40) / 60)
    assert eta.window_progress == 700
    assert eta.window_seconds == 40


def test_recurring_waits_count_but_old_slow_intervals_expire():
    eta = RecentThroughputETA()
    eta.update(100, 1000, 1000)
    eta.update(200, 1100, 1000)  # Includes a checkpoint/data wait.
    assert eta.rate == 1
    eta.update(300, 1110, 1000)
    eta.update(400, 1120, 1000)
    eta.update(500, 1130, 1000)
    assert eta.rate == 10  # Old wait no longer biases the rate.
    assert eta.update(1000, 1180, 1000) == 0
    assert eta.update(1001, 1181, 1000) == 0


def test_restart_duplicate_and_no_progress_do_not_invent_a_rate():
    eta = RecentThroughputETA()
    assert eta.update(100, 10, 1000) is None
    assert eta.update(100, 10, 1000) is None
    assert eta.update(100, 20, 1000) is None
    assert eta.update(200, 30, 1000) == pytest.approx(800 / 5 / 60)
    assert eta.update(10, 1, 1000) is None
    assert eta.update(20, 3, 1000) == pytest.approx(980 / 5 / 60)


def _ui_helpers():
    source = Path(__file__).parents[1] / 'scripts/config_webui.py'
    names = {'parse_training_metrics', '_format_training_duration', 'render_log_metrics_charts'}
    nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'re': re, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


def test_legacy_chart_recomputes_token_eta_and_resets_between_epochs_and_resumes():
    parse = _ui_helpers()['parse_training_metrics']
    lines = [
        'Epoch:[1/2](100/~999), tokens: 100/1000, loss: 2, elapsed_time: 10.0min, epoch_time: 900min',
        'Epoch:[1/2](200/~999), tokens: 300/1000, loss: 2, elapsed_time: 10.5min, epoch_time: 800min',
        'Epoch:[2/2](100/~999), tokens: 100/1000, loss: 2, elapsed_time: 1.0min, epoch_time: 700min',
        'Epoch:[2/2](200/~999), tokens: 400/1000, loss: 2, elapsed_time: 1.5min, epoch_time: 600min',
        'Epoch [2/2]: resume global step 201, chunk=1, chunk_step=200',
        'Epoch:[2/2](300/~999), tokens: 500/1000, loss: 2, elapsed_time: 5.0min, epoch_time: 500min',
        'Epoch:[2/2](400/~999), tokens: 800/1000, loss: 2, elapsed_time: 5.5min, epoch_time: 400min',
    ]
    rows = parse('\n'.join(lines))
    assert ['epoch_time' in r for r in rows] == [False, True, False, True, False, True]
    assert [rows[i]['epoch_time'] for i in (1, 3, 5)] == pytest.approx([1.75, 1, 1/3])
    # Never import an earlier launch's rate into an appended resume segment.
    assert 'epoch_time' not in parse('\n'.join(lines) + '\n# epochs: 2\n' + lines[0])[0]


def test_new_eta_survives_partial_log_reads_and_no_eta_is_not_zero():
    parse = _ui_helpers()['parse_training_metrics']
    ready = ('Epoch:[1/1](100/~999), tokens: 300/1000, loss: 2, epoch_time: 1.7min, '
             'elapsed_seconds: 630.0, eta_basis: recent_tokens')
    assert parse(ready)[0]['epoch_time'] == 1.7
    pending = ready.replace('epoch_time: 1.7min', 'epoch_time: estimating')
    assert 'epoch_time' not in parse(pending)[0]
    old_without_clock = 'Epoch:[1/1](100/999), loss: 2, epoch_time: 3.0min'
    assert parse(old_without_clock)[0]['epoch_time'] == 3


@pytest.mark.parametrize('resume_flags', [
    '--save_weight pretrain_run --from_resume 1',
    '--save_weight="pretrain_run" --from_resume="1"',
])
def test_resume_chart_retains_history_but_resets_throughput_window(resume_flags):
    parse = _ui_helpers()['parse_training_metrics']
    log = (
        '# command: python train.py --save_weight pretrain_run\n# epochs: 1\n'
        'Epoch:[1/1](100/1000), loss: 3, elapsed_seconds: 600\n'
        'Epoch:[1/1](200/1000), loss: 2, elapsed_seconds: 630\n'
        '[PAUSED] Training paused — resume checkpoint saved.\n'
        f'# command: python train.py {resume_flags}\n# epochs: 1\n'
        'initializing\n'
    )
    # Compilation/loading after resume must not make the previous curve vanish.
    assert [row['_epoch_step'] for row in parse(log)] == [100, 200]
    log += (
        'Epoch [1/1]: resume global step 201, chunk=1, chunk_step=200\n'
        'Epoch:[1/1](300/1000), loss: 1.9, elapsed_seconds: 100\n'
        'Epoch:[1/1](400/1000), loss: 1.8, elapsed_seconds: 160\n'
    )
    rows = parse(log)
    assert [row['_epoch_step'] for row in rows] == [100, 200, 300, 400]
    assert ['epoch_time' in row for row in rows] == [False, True, False, True]
    assert rows[-1]['epoch_time'] == pytest.approx(6.0)


@pytest.mark.parametrize('header', [
    '# command: python train.py --save_weight pretrain_run --from_resume 0\n',
    '# command: python train.py --save_weight pretrain_other --from_resume 1\n',
    '',  # Older unidentified appended launches keep the conservative behavior.
])
def test_new_or_unidentified_launch_does_not_merge_chart_history(header):
    log = (
        '# command: python train.py --save_weight pretrain_run\n# epochs: 1\n'
        'Epoch:[1/1](100/1000), loss: 3\n'
        'Epoch:[1/1](200/1000), loss: 2\n'
        f'{header}# epochs: 1\n'
        'Epoch:[1/1](100/1000), loss: 4\n'
    )
    rows = _ui_helpers()['parse_training_metrics'](log)
    assert [row['_epoch_step'] for row in rows] == [100]
    assert rows[0]['loss'] == 4


@pytest.mark.parametrize('cursor', [
    'Epoch [1/1]: resume global step 201, chunk=1, chunk_step=200',
    'Epoch [1/1]: 跳过前200个step，从step 201开始',
])
def test_chart_discards_only_replayed_suffix_when_resuming_older_checkpoint(cursor):
    log = (
        '# command: python train.py --lora_name lora_run\n# epochs: 1\n'
        'Epoch:[1/1](100/1000), loss: 3\n'
        'Epoch:[1/1](200/1000), loss: 2\n'
        'Epoch:[1/1](300/1000), loss: 1\n'
        '# command: python train.py --lora_name lora_run --from_resume 1\n# epochs: 1\n'
        f'{cursor}\n'
    )
    parse = _ui_helpers()['parse_training_metrics']
    assert [row['_epoch_step'] for row in parse(log)] == [100, 200]
    rows = parse(log + 'Epoch:[1/1](300/1000), loss: 1.5\n')
    assert [row['_epoch_step'] for row in rows] == [100, 200, 300]
    assert rows[-1]['loss'] == 1.5


def test_eta_chart_renders_pending_and_ready_points():
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st
        from tests.test_training_eta import _ui_helpers
        helpers = _ui_helpers()
        helpers['st'] = st
        helpers['render_log_metrics_charts']([
            {'_global_step': 100, 'loss': 2, 'elapsed_time': 59},
            {'_global_step': 200, 'loss': 1.9, 'elapsed_time': 61.5, 'epoch_time': 1501.2},
        ])

    view = AppTest.from_function(app).run()
    assert not view.exception
    assert any(x.value == 'Epoch Remaining' for x in view.caption)
    assert any(x.value == 'Elapsed: 1 h 1.5 min · Epoch Remaining: 1 d 1 h 1.2 min'
               for x in view.caption)


@pytest.mark.parametrize('streaming', [False, True])
def test_real_pretrain_log_samples_clock_after_loss_sync(monkeypatch, streaming):
    import torch
    import trainer.train_pretrain as trainer
    from trainer.training_profiler import TrainingProfiler

    clock = SimpleNamespace(now=0.0)
    sync_costs = iter([600, 10, 30])
    real_item = torch.Tensor.item

    def synchronized_item(tensor, *args, **kwargs):
        if tensor.requires_grad:
            clock.now += next(sync_costs)  # Represents asynchronous CUDA work.
        return real_item(tensor, *args, **kwargs)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.25))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(loss=self.weight.square(), aux_loss=self.weight.new_zeros(()))

    args = SimpleNamespace(device='cpu', profile='off', early_exit=0, epochs=2,
                           accumulation_steps=1, grad_clip=1., log_interval=1,
                           save_interval=10000, use_compile=0)
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    data_config = {'streaming_processed_tokens': 12300, 'streaming_chunks': 10}
    for key, value in dict(args=args, model=model, optimizer=optimizer,
                           scaler=torch.amp.GradScaler('cpu', enabled=False),
                           profiler=TrainingProfiler(args), autocast_ctx=nullcontext(),
                           lm_config=SimpleNamespace(use_moe=False), data_config=data_config).items():
        monkeypatch.setattr(trainer, key, value, raising=False)
    monkeypatch.setattr(trainer.time, 'perf_counter', lambda: clock.now)
    monkeypatch.setattr(torch.Tensor, 'item', synchronized_item)
    monkeypatch.setattr(trainer, 'set_cosine_lr', lambda *a: None)
    monkeypatch.setattr(trainer, 'set_cosine_lr_progress', lambda *a: None)
    monkeypatch.setattr(trainer, '_pause_after_update', lambda *a: None)
    logs, metrics = [], []
    monkeypatch.setattr(trainer, 'Logger', logs.append)
    batches = [(torch.ones((1, n), dtype=torch.long), torch.ones((1, n), dtype=torch.long))
               for n in (4, 8, 16)]
    trainer.train_epoch(1, batches, 1000, 40, wandb=SimpleNamespace(log=metrics.append),
                        token_schedule={'total_tokens': 20000} if streaming else None)
    assert 'epoch_time: estimating' in logs[0]
    assert [m['elapsed_seconds'] for m in metrics] == [600, 610, 640]
    assert metrics[-1]['eta_window_seconds'] == 40
    expected = (20000 - 12328) / (24 / 40) / 60 if streaming else (1000 - 43) / (2 / 40) / 60
    assert metrics[-1]['epoch_time'] == pytest.approx(expected)
