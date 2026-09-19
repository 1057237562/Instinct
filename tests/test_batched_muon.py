"""Small numerical/state tests; CUDA coverage is explicitly marked."""

import copy

import pytest
import torch

from trainer.batched_muon import BatchedMuon


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _compare_native(device, *, nesterov=True, momentum=0.95, scaling='match_rms_adamw'):
    if not hasattr(torch.optim, 'Muon'):
        pytest.skip('requires native Muon for independent reference')
    torch.manual_seed(42)
    shapes = [(12, 8)] * 5 + [(8, 12)] * 3 + [(8, 8)]
    original = [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]
    batched = [torch.nn.Parameter(p.detach().clone()) for p in original]
    kwargs = dict(lr=5e-4, weight_decay=0.1, momentum=momentum,
                  nesterov=nesterov, adjust_lr_fn=scaling)
    native = torch.optim.Muon(original, **kwargs)
    candidate = BatchedMuon(batched, batch_size=3, **kwargs)
    for step in range(4):
        before = [p.detach().clone() for p in original]
        before_batched = [p.detach().clone() for p in batched]
        for i, (p, q) in enumerate(zip(original, batched)):
            # Missing gradients must skip both weight decay and momentum update.
            grad = None if i == step else torch.randn_like(p) * (i + 1)
            p.grad = grad
            q.grad = None if grad is None else grad.clone()
        native.step()
        candidate.step()
        for i, (p, q) in enumerate(zip(original, batched)):
            torch.testing.assert_close(q, p, atol=2e-5, rtol=2e-5)
            if p.grad is None:
                torch.testing.assert_close(q, before_batched[i], atol=0, rtol=0)
                continue
            torch.testing.assert_close(candidate.state[q]['momentum_buffer'],
                                       native.state[p]['momentum_buffer'], atol=1e-6, rtol=1e-6)
            expected = before[i] - p
            actual = before_batched[i] - q
            # Compare the small update too: close large weights alone can hide
            # an incorrect optimizer. BF16 GEMM/reduction order may differ.
            relative_error = (expected - actual).norm() / expected.norm().clamp_min(1e-12)
            assert relative_error.item() < 0.03


@pytest.mark.parametrize('nesterov,momentum,scaling', [
    (True, 0.95, 'match_rms_adamw'), (False, 0.8, 'original'), (False, 0.0, None),
])
def test_matches_native_cpu(nesterov, momentum, scaling):
    _compare_native('cpu', nesterov=nesterov, momentum=momentum, scaling=scaling)


@pytest.mark.gpu
def test_matches_native_cuda():
    _compare_native('cuda')


@pytest.mark.gpu
def test_real_expert_shapes_cuda_under_autocast():
    if not hasattr(torch.optim, 'Muon'):
        pytest.skip('requires native Muon')
    torch.manual_seed(73)
    shapes = [(1664, 512)] * 2 + [(512, 1664)] * 2 + [(512, 512)] * 2
    params = [torch.nn.Parameter(torch.zeros(s, device='cuda')) for s in shapes]
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    native = torch.optim.Muon(params, lr=5e-4, adjust_lr_fn='match_rms_adamw')
    new = BatchedMuon(copies, lr=5e-4, adjust_lr_fn='match_rms_adamw')
    for _ in range(3):
        for p, q in zip(params, copies):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        before = [p.detach().clone() for p in params]
        before_new = [p.detach().clone() for p in copies]
        native.step()
        with torch.autocast('cuda', dtype=torch.float16):
            new.step()
        for p, q, a, b in zip(params, copies, before, before_new):
            reference = a - p
            actual = b - q
            assert ((actual - reference).norm() / reference.norm()).item() < 0.03
            torch.testing.assert_close(new.state[q]['momentum_buffer'],
                                       native.state[p]['momentum_buffer'], atol=1e-6, rtol=1e-6)


def test_resume_native_and_batched_both_directions():
    if not hasattr(torch.optim, 'Muon'):
        pytest.skip('requires native Muon')
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(8, 12))
    native = torch.optim.Muon([p], lr=0.003, adjust_lr_fn='original')
    p.grad = torch.randn_like(p)
    native.step()
    q = torch.nn.Parameter(p.detach().clone())
    batched = BatchedMuon([q], batch_size=2, workspace_mb=1)
    batched.load_state_dict(copy.deepcopy(native.state_dict()))
    assert batched.batch_size == 2
    assert batched.workspace_bytes == 2**20
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    native.step()
    batched.step()
    torch.testing.assert_close(q, p, atol=3e-5, rtol=2e-5)

    r = torch.nn.Parameter(q.detach().clone())
    restored = torch.optim.Muon([r])
    restored.load_state_dict(copy.deepcopy(batched.state_dict()))
    q.grad = torch.randn_like(q)
    r.grad = q.grad.clone()
    batched.step()
    restored.step()
    torch.testing.assert_close(q, r, atol=3e-5, rtol=2e-5)


def test_legacy_state_without_coefficients_and_different_groups():
    from trainer.trainer_utils import MuonOptimizer

    p = torch.nn.Parameter(torch.ones(4, 8))
    legacy = MuonOptimizer([p], lr=0.003)
    p.grad = torch.ones_like(p)
    legacy.step()
    q = torch.nn.Parameter(p.detach().clone())
    new = BatchedMuon([q])
    new.load_state_dict(copy.deepcopy(legacy.state_dict()))
    assert new.param_groups[0]['ns_coefficients'] == (3.4445, -4.7750, 2.0315)
    q.grad = torch.zeros_like(q)
    new.step()
    assert torch.isfinite(q).all()

    a = torch.nn.Parameter(torch.ones(4, 8))
    b = torch.nn.Parameter(a.detach().clone())
    opt = BatchedMuon([{'params': [a], 'lr': 0.01}, {'params': [b], 'lr': 0.02}], weight_decay=0)
    a.grad = torch.ones_like(a)
    b.grad = torch.ones_like(b)
    opt.step()
    torch.testing.assert_close(1 - b, 2 * (1 - a))


def test_zero_gradients_budget_and_noncontiguous_parameters():
    params = [torch.nn.Parameter(torch.ones(8, 12).T) for _ in range(3)]
    opt = BatchedMuon(params, lr=0.01, weight_decay=0.1, workspace_mb=0.001)
    assert opt._chunk_size(params[0]) == 1
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()
    for p in params:
        torch.testing.assert_close(p, torch.full_like(p, 0.999))
        assert torch.count_nonzero(opt.state[p]['momentum_buffer']) == 0


def test_factory_combined_resume_and_native_escape_hatch(monkeypatch):
    from trainer.trainer_utils import CombinedOptimizer, build_optimizer

    monkeypatch.setenv('INSTINCT_MUON_BACKEND', 'native')
    old_model = torch.nn.Linear(8, 12)
    old = build_optimizer(old_model.named_parameters(), lr=5e-4, optimizer='muon')
    for p in old_model.parameters():
        p.grad = torch.ones_like(p)
    old.step()
    monkeypatch.setenv('INSTINCT_MUON_BACKEND', 'batched')
    new_model = copy.deepcopy(old_model)
    new = build_optimizer(new_model.named_parameters(), lr=5e-4, optimizer='muon')
    assert isinstance(new, CombinedOptimizer)
    assert isinstance(new.optimizers[0], BatchedMuon)
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    assert new.param_groups[0] is new.optimizers[0].param_groups[0]
    new.param_groups[0]['lr'] = 0.002
    assert new.optimizers[0].param_groups[0]['lr'] == 0.002
    for p in new_model.parameters():
        p.grad = torch.ones_like(p)
    new.step()
    assert len(new.optimizers[0].state) == 1


def test_rejects_sparse_gradients():
    p = torch.nn.Parameter(torch.ones(4, 4))
    opt = BatchedMuon([p])
    p.grad = torch.eye(4).to_sparse()
    with pytest.raises(RuntimeError, match='sparse'):
        opt.step()
