from types import SimpleNamespace

import pytest

from trainer import trainer_cli
from trainer.trainer_cli import build_trainer_parser


def test_gradient_accumulation_defaults_to_one():
    args = build_trainer_parser("default-test").parse_args([])
    assert args.accumulation_steps == 1


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
