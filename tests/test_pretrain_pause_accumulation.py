"""A checkpointed pause must not discard pending microbatch gradients."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import trainer.train_pretrain as trainer
from trainer.training_profiler import TrainingProfiler


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(loss=(self.weight - input_ids.float().mean()).square(),
                               aux_loss=self.weight.new_zeros(()))


@pytest.mark.parametrize('accumulation,start,count,request_at,expected_step', [
    (8, 0, 12, 3, 8), (8, 8, 12, 1, 16), (1, 0, 4, 1, 1),
    (8, 0, 3, 2, 3),  # End of a short epoch must flush before saving.
])
def test_pause_saves_applied_gradients(monkeypatch, tmp_path, accumulation, start,
                                      count, request_at, expected_step):
    args = SimpleNamespace(device='cpu', profile='off', early_exit=0,
                           accumulation_steps=accumulation, grad_clip=1.,
                           log_interval=10000, save_interval=10000, use_compile=0,
                           save_weight='test', pause_file=str(tmp_path/'pause'))
    model = TinyModel()
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    for name, value in dict(args=args, model=model, optimizer=opt, scaler=scaler,
                            profiler=TrainingProfiler(args), autocast_ctx=nullcontext(),
                            lm_config=SimpleNamespace(use_moe=False), data_config={}).items():
        monkeypatch.setattr(trainer, name, value, raising=False)
    monkeypatch.setattr(trainer, 'set_cosine_lr', lambda *a: None)
    monkeypatch.setattr(trainer, 'prepare_lm_batch', lambda batch, device: (*batch, None))
    saved = []
    def save(*a, **kw):
        assert all(p.grad is None for p in model.parameters())
        saved.append(kw['step'])
    monkeypatch.setattr(trainer, 'pause_save_checkpoint', save)
    batches = [torch.full((1, 4), i + 1, dtype=torch.long) for i in range(count)]
    def loader():
        for index, tokens in enumerate(batches, 1):
            if index == request_at:
                (tmp_path/'pause').touch()
            yield tokens, tokens
    with pytest.raises(SystemExit) as stop:
        trainer.train_epoch(0, loader(), 10000, start_step=start)
    assert stop.value.code == 42 and saved == [expected_step]
    assert not (tmp_path/'pause').exists()
    for tokens in batches[:expected_step-start]:
        (reference(tokens).loss/accumulation).backward()
    trainer.step_with_scaler(scaler, ref_opt, reference.parameters(), 1.)
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
