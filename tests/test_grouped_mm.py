"""Ragged GEMM correctness and device-only execution checks."""

import pytest
import torch

from model.grouped_mm import expert_grouped_mm, make_grouped_mm_plan


def _reference(x, weight, ends):
    chunks = []
    start = 0
    for expert, end in enumerate(ends):
        chunks.append(x[start:end] @ weight[expert])
        start = end
    return torch.cat(chunks, dim=0)


def _compare(device, backend, counts, inner, columns, transpose_weight=True):
    torch.manual_seed(13)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float64
    ends = torch.tensor(counts).cumsum(0).tolist()
    offsets = torch.tensor(ends, device=device, dtype=torch.int32)
    x = torch.randn(sum(counts), inner, device=device, dtype=dtype).requires_grad_()
    if transpose_weight:
        weight = torch.randn(len(counts), columns, inner, device=device, dtype=dtype).transpose(1, 2)
    else:
        weight = torch.randn(len(counts), inner, columns, device=device, dtype=dtype)
    weight.requires_grad_()
    plan = make_grouped_mm_plan(offsets, backend=backend)
    out = expert_grouped_mm(x, weight, plan)
    reference = _reference(x, weight, ends)
    # Explicitly test strided incoming gradients, as used by transpose consumers.
    grad = torch.randn(columns, sum(counts), device=device, dtype=dtype).T
    actual_grads = torch.autograd.grad(out, (x, weight), grad)
    reference_grads = torch.autograd.grad(reference, (x, weight), grad)
    comparisons = [(out, reference), *zip(actual_grads, reference_grads)]
    for actual, expected in comparisons:
        if device != 'cuda':
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        if expected.numel() and expected.float().norm() > 0:
            assert ((actual.float() - expected.float()).norm() / expected.float().norm()) < 0.006
    if device == 'cuda':
        # cuBLAS may use reduced-precision BF16 intermediate reductions. Near
        # cancellation those differ from FP32 accumulation by much more than
        # output-rounding error. Check against an independent FP32 oracle too,
        # rather than relaxing elementwise checks against that approximation.
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            xf = x.detach().float().requires_grad_()
            wf = weight.detach().float().requires_grad_()
            yf = _reference(xf, wf, ends)
            gf = torch.autograd.grad(yf, (xf, wf), grad.float())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        if backend == 'triton':
            for actual, expected in [(out, yf), *zip(actual_grads, gf)]:
                torch.testing.assert_close(actual.float(), expected,
                                           atol=0.003, rtol=0.004)
    for i, count in enumerate(counts):
        if count == 0:
            assert torch.count_nonzero(actual_grads[1][i]) == 0


@pytest.mark.parametrize('counts', [[0, 7, 0, 13, 31, 0, 9, 0], [0]*8, [137]+[0]*7])
def test_cached_plan_cpu_matches_forward_and_backward(counts):
    _compare('cpu', 'cached', counts, 16, 24)


def test_cached_plan_gradcheck_and_frozen_input():
    offsets = torch.tensor([0, 2, 7], dtype=torch.int32)
    plan = make_grouped_mm_plan(offsets, backend='cached')
    x = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    w = torch.randn(3, 3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: expert_grouped_mm(a, b, plan), (x, w))
    expert_grouped_mm(x.detach(), w, plan).sum().backward()
    assert w.grad is not None


def test_plan_caches_boundaries_once_and_rejects_bad_backend(monkeypatch):
    offsets = torch.tensor([0, 2, 7], dtype=torch.int32)
    plan = make_grouped_mm_plan(offsets, backend='cached')
    assert plan.ends == (0, 2, 7)
    # All projections use the identical immutable metadata, not new CPU copies.
    x = torch.randn(7, 3, requires_grad=True)
    w = torch.randn(3, 3, 4, requires_grad=True)
    def fail(*a, **kw):
        raise AssertionError('GEMM tried to reread routing metadata')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', fail)
        patch.setattr(torch.Tensor, 'tolist', fail)
        for _ in range(3):
            expert_grouped_mm(x, w, plan).sum().backward()
    with pytest.raises(ValueError, match='BACKEND'):
        make_grouped_mm_plan(offsets, backend='invalid')


def test_original_expert_state_dict_still_loads_strictly():
    from model.model_instinct import MOEFeedForward
    from tests.helpers import make_tiny_config

    original = MOEFeedForward(make_tiny_config(use_moe=True))
    state = original.state_dict()
    restored = MOEFeedForward(make_tiny_config(use_moe=True))
    restored.load_state_dict(state, strict=True)
    assert 'experts.0.gate_proj.weight' in state
    assert all(p.ndim == 2 for p in restored.parameters())
    assert not any('plan' in name or 'offsets' in name for name in state)


@pytest.mark.gpu
@pytest.mark.parametrize('backend', ['triton', 'cached'])
@pytest.mark.parametrize('counts,inner,columns', [
    ([0, 7, 0, 13, 31, 0, 9, 0], 32, 48),
    ([0]*8, 32, 48),
    ([137]+[0]*7, 48, 32),
    ([1, 63, 64, 65, 129, 0, 3, 27], 19, 23),
])
def test_cuda_ragged_gemm_matches_reference(backend, counts, inner, columns):
    if backend == 'triton':
        pytest.importorskip('triton')
    _compare('cuda', backend, counts, inner, columns)


@pytest.mark.gpu
def test_real_expert_shapes():
    pytest.importorskip('triton')
    for inner, columns in [(512, 1664), (1664, 512)]:
        _compare('cuda', 'triton', [3, 111, 127, 31, 64, 1, 0, 0], inner, columns)


@pytest.mark.gpu
def test_long_reduction_with_skewed_experts():
    pytest.importorskip('triton')
    _compare('cuda', 'triton', [100, 200, 2400, 0, 1800, 3300, 2000, 1912], 512, 1664)


@pytest.mark.gpu
def test_triton_forward_backward_have_no_host_readbacks():
    pytest.importorskip('triton')
    x = torch.randn(127, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(8, 32, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    ends = torch.tensor([0, 5, 5, 64, 64, 99, 127, 127], device='cuda', dtype=torch.int32)
    plan = make_grouped_mm_plan(ends, backend='triton')
    # Compile first; compilation/initialization is not steady-state routing.
    expert_grouped_mm(x, w, plan).sum().backward()
    x.grad = w.grad = None
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode('error')
        expert_grouped_mm(x, w, plan).sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert x.grad is not None and w.grad is not None
