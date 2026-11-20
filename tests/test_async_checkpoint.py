from threading import Event
from types import SimpleNamespace

import pytest
import torch

from trainer import trainer_utils as checkpoints


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'out'
    output.mkdir()
    args = SimpleNamespace(save_dir=str(output), pause_file=str(tmp_path / 'pause'))
    config = SimpleNamespace(use_moe=False, hidden_size=2, to_dict=lambda: {'hidden_size': 2})
    model = torch.nn.Linear(2, 2)
    model.lora = torch.nn.Linear(2, 2, bias=False)
    model.tied = model.weight
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = dict(weight='test', model=model, optimizer=optimizer, epoch=0)
    yield args, config, state
    checkpoints.wait_for_checkpoint()


def test_training_can_mutate_state_while_writer_waits(run, monkeypatch):
    args, config, state = run
    entered, release = Event(), Event()
    write = checkpoints._write_checkpoint

    def blocked_write(*args):
        entered.set()
        assert release.wait(10)
        write(*args)

    monkeypatch.setattr(checkpoints, '_write_checkpoint', blocked_write)
    model = state['model']
    optimizer = state['optimizer']
    momentum = optimizer.state[model.weight]['momentum_buffer']
    saved_weights, saved_momentum = model.weight.detach().clone(), momentum.clone()
    cursor = {'chunk': [3, 1]}
    scaler = torch.amp.GradScaler('cpu', init_scale=128)
    checkpoints.save_training_checkpoint(
        args, config, step=7, data_config=cursor, scaler=scaler, lora_save=True, **state,
    )
    try:
        assert entered.wait(10)
        assert checkpoints.lm_checkpoint(config, weight='test') is None
        with torch.no_grad():
            model.weight.add_(10)
            momentum.add_(20)
        cursor['chunk'][1] = 2
        scaler.load_state_dict(dict(scaler.state_dict(), scale=256))
    finally:
        release.set()
        checkpoints.wait_for_checkpoint()

    saved = checkpoints.lm_checkpoint(config, weight='test')
    assert saved['step'] == 7 and saved['data_config'] == {'chunk': [3, 1]}
    assert saved['scaler']['scale'] == 128
    torch.testing.assert_close(saved['model']['weight'], saved_weights)
    assert saved['model']['weight'].data_ptr() == saved['model']['tied'].data_ptr()
    param_id = optimizer.state_dict()['param_groups'][0]['params'][0]
    torch.testing.assert_close(saved['optimizer']['state'][param_id]['momentum_buffer'], saved_momentum)
    assert set(torch.load('out/test_2.pth')) == {'lora.weight'}


def test_failed_pause_preserves_previous_checkpoint_and_can_retry(run, monkeypatch, capsys, tmp_path):
    args, config, state = run
    checkpoints.save_training_checkpoint(args, config, step=4, wait=True, **state)
    saved_output = (tmp_path / 'out' / 'test_2.pth').read_bytes()
    with torch.no_grad():
        state['model'].weight.add_(1)

    save = torch.save

    def fail_resume(value, handle):
        if 'optimizer' in value:
            handle.write(b'partial checkpoint')
            raise OSError('disk full')
        save(value, handle)

    marker = tmp_path / 'pause'
    profiler = SimpleNamespace(finish=lambda: None)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'save', fail_resume)
        marker.touch()
        checkpoints.checkpoint_after_update(args, config, profiler, periodic=False, step=8, **state)
    log = capsys.readouterr().out
    assert 'Pause save failed: disk full' in log
    assert '[PAUSED]' not in log
    assert checkpoints.lm_checkpoint(config, weight='test')['step'] == 4
    assert (tmp_path / 'out' / 'test_2.pth').read_bytes() == saved_output
    assert not list(tmp_path.rglob('*.tmp'))

    marker.touch()
    with pytest.raises(SystemExit) as stop:
        checkpoints.checkpoint_after_update(args, config, profiler, periodic=True, step=12, **state)
    assert stop.value.code == 42
    assert checkpoints.lm_checkpoint(config, weight='test')['step'] == 12
