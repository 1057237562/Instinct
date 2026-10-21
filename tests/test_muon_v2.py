"""V2 shape routing, exact NS arithmetic, and complete optimizer state parity."""
import copy
from types import SimpleNamespace

import pytest
import torch

from trainer.batched_muon import (
    BatchedMuon, _NS_COEFFICIENTS, _batched_zeropower_reference, _ns_backend,
    _split_precise,
)


V2_SHAPES = [(1536, 1536), (384, 1536), (5504, 1536), (1536, 5504)]


def test_auto_routing_keeps_v1_graphs_and_v2_memory_headroom(monkeypatch):
    from model import cuda_fastpath

    monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND', 'auto')
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: True)
    for shape in V2_SHAPES:
        assert _ns_backend(SimpleNamespace(shape=(1, *shape), device='cuda')) == 'triton'
    assert _ns_backend(SimpleNamespace(shape=(4, 512, 1664), device='cuda')) == 'triton_graph'
    assert _ns_backend(SimpleNamespace(shape=(1, 1024, 4096), device='cuda')) == 'torch'
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: False)
    for shape in V2_SHAPES:
        assert _ns_backend(SimpleNamespace(shape=(1, *shape), device='cpu')) == 'torch'
    monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND', 'torch')
    assert _ns_backend(SimpleNamespace(shape=(1, 1536, 5504), device='cuda')) == 'torch'


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [
    (1, 1536, 1536), (4, 1536, 1536),
    (1, 384, 1536), (16, 384, 1536),
    (1, 5504, 1536), (2, 5504, 1536),
    (1, 1536, 5504), (2, 1536, 5504),
])
@pytest.mark.parametrize('kind', ['random', 'zero', 'rank_one'])
def test_v2_ns_and_graph_match_reference(shape, kind):
    from trainer.muon_kernels import zeropower, ZeropowerGraphCache

    torch.manual_seed(53)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    if kind == 'zero':
        x.zero_()
    elif kind == 'rank_one':
        x = x[:, :, :1].expand(shape).contiguous()
    opts = dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7)
    expected = _batched_zeropower_reference(x, **opts)
    torch.testing.assert_close(zeropower(x, **opts), expected, rtol=0, atol=0)
    cache = ZeropowerGraphCache(limit=1)
    torch.testing.assert_close(cache(x, **opts), expected, rtol=0, atol=0)
    # Replay with a new allocation, including a different reduction trajectory.
    x = torch.randn_like(x)
    torch.testing.assert_close(cache(x, **opts), _batched_zeropower_reference(x, **opts),
                               rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('shape', V2_SHAPES)
@pytest.mark.parametrize('storage', ['fp32', 'mantissa_fp32', 'mantissa_bf16'])
@pytest.mark.parametrize('nesterov', [True, False])
def test_v2_auto_optimizer_trajectory_and_resume(monkeypatch, shape, storage, nesterov):
    from model.cuda_fastpath import blackwell_training_kernels

    if not blackwell_training_kernels('cuda'):
        pytest.skip('V2 auto fast path requires the validated Blackwell/PyTorch stack')
    torch.manual_seed(173)
    mantissa = storage != 'fp32'
    kwargs = dict(use_mantissa=mantissa, momentum_dtype='bf16' if storage == 'mantissa_bf16'
                  else 'fp32', nesterov=nesterov, workspace_mb=256,
                  adjust_lr_fn='match_rms_adamw')
    # Three matrices exercise both batches and tails; missing gradients change
    # the batch size without changing parameter/state identities.
    values = [torch.randn(shape, device='cuda') * .02 for _ in range(3)]
    pairs = [_split_precise(v) for v in values] if mantissa else None
    refs = [torch.nn.Parameter((pairs[i][0] if mantissa else v).clone())
            for i, v in enumerate(values)]
    params = [torch.nn.Parameter(p.detach().clone()) for p in refs]
    reference, candidate = BatchedMuon(refs, **kwargs), BatchedMuon(params, **kwargs)
    if mantissa:
        for p, q, (_, low) in zip(refs, params, pairs):
            reference.state[p]['mantissa'] = low.clone()
            candidate.state[q]['mantissa'] = low.clone()
    del values, pairs
    pointers = [p.data_ptr() for p in params]
    for step in range(6):
        for i, (p, q) in enumerate(zip(refs, params)):
            p.grad = None if step % 4 == i else torch.randn_like(p)
            q.grad = p.grad
        for optimizer in (reference, candidate):
            optimizer.param_groups[0]['lr'] = [5e-4, 0., 1e-6, 3e-4, 2e-4, 1e-5][step]
            optimizer.param_groups[0]['momentum'] = [.95, .5, .8, 0., 1., .95][step]
        monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND', 'torch')
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', '0')
        reference.step()
        monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND', 'auto')
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', 'auto')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            candidate.step()
        assert not hasattr(candidate, '_ns_graph_cache')
        assert pointers == [p.data_ptr() for p in params]
        for p, q in zip(refs, params):
            assert torch.equal(p, q)
            for key, value in reference.state[p].items():
                assert torch.equal(value, candidate.state[q][key]), (step, key)
                assert value.dtype == candidate.state[q][key].dtype
        if step == 2:
            saved = copy.deepcopy(candidate.state_dict())
            candidate.load_state_dict(saved)
