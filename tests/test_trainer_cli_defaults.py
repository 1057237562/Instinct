from types import SimpleNamespace

import pytest

from trainer import trainer_cli
from trainer.trainer_cli import build_trainer_parser


def test_gradient_accumulation_defaults_to_one():
    args = build_trainer_parser("default-test").parse_args([])
    assert args.accumulation_steps == 1


@pytest.mark.parametrize('mode', [3, 4])
def test_dense_selective_checkpoint_cli_choice(mode):
    args = build_trainer_parser('checkpoint-test').parse_args(['--use_grad_checkpoint', str(mode)])
    assert args.use_grad_checkpoint == mode


def test_bucket_vram_setting_caps_cuda_allocator(monkeypatch):
    calls = []
    monkeypatch.setattr(trainer_cli.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        trainer_cli.torch.cuda, 'get_device_properties',
        lambda _device: SimpleNamespace(total_memory=24 * 1024 ** 3),
    )
    monkeypatch.setattr(
        trainer_cli.torch.cuda, 'set_per_process_memory_fraction',
        lambda fraction, device=None: calls.append((fraction, str(device))),
    )
    args = SimpleNamespace(
        sequence_packing=1, sequence_packing_mode='bucket',
        device='cuda:0', bucket_gpu_memory_gb=16.0,
    )

    limit = trainer_cli._apply_bucket_cuda_memory_limit(args)

    assert limit == pytest.approx(15.5)
    assert calls == [(pytest.approx(15.5 / 24.0), 'cuda:0')]


@pytest.mark.parametrize('budget_gb,expected', [(14.,13.),(.25,None)])
def test_bucket_allocator_respects_driver_budget(monkeypatch,budget_gb,expected):
    from trainer import windows_gpu_memory
    monkeypatch.setattr(trainer_cli.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(trainer_cli.torch.cuda,'get_device_properties',
                        lambda d:SimpleNamespace(total_memory=16*2**30,name='test GPU'))
    monkeypatch.setattr(trainer_cli.torch.cuda,'memory_reserved',lambda d:3*2**30)
    monkeypatch.setattr(windows_gpu_memory,'local_memory_budget',
                        lambda name:dict(budget=int(budget_gb*2**30),current_usage=4*2**30))
    calls=[]
    monkeypatch.setattr(trainer_cli.torch.cuda,'set_per_process_memory_fraction',
                        lambda fraction,device:calls.append(fraction))
    args=SimpleNamespace(sequence_packing=1,sequence_packing_mode='bucket',device='cuda:0',
                         bucket_gpu_memory_gb=16.)
    if expected is None:
        with pytest.raises(RuntimeError,match='safety reserve'):
            trainer_cli._apply_bucket_cuda_memory_limit(args)
        assert not calls
    else:
        assert trainer_cli._apply_bucket_cuda_memory_limit(args)==expected
        assert calls==[expected/16.]
        assert args.bucket_driver_budget_gb==budget_gb
        assert args.bucket_gpu_memory_gb==16.  # Keep the user's requested setting.
