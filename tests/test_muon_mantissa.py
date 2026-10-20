"""Exactness tests for BatchedMuon's opt-in uint16-mantissa mode.

The split/reconstruct pair is a truncating bit split, so the reconstruction is a
bitwise identity. The core acceptance test drives a mantissa arm and an fp32
reference arm over identical seeded gradients and requires bitwise equality of
the reconstructed trajectory.
"""

import copy
import io
import types

import pytest
import torch

from trainer.batched_muon import BatchedMuon, _reconstruct_precise, _split_precise


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


_KWARGS = dict(lr=5e-4, weight_decay=0.1, adjust_lr_fn='match_rms_adamw',
               batch_size=2, workspace_mb=1)


def _mantissa_pair(shapes, device='cpu'):
    """Return (fp32 reference params, bf16 params, seeded mantissa lows)."""
    torch.manual_seed(11)
    reference = [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]
    precise, mantissas = [], []
    for p in reference:
        high, low = _split_precise(p.detach().clone())
        param = torch.nn.Parameter(high.clone())
        precise.append(param)
        mantissas.append(low)
    return reference, precise, mantissas


def _run_trajectory(device, *, nesterov, steps=6):
    """Drive fp32 reference and mantissa arms; return per-step equality results."""
    shapes = [(12, 8)] * 3 + [(8, 8)] * 2
    reference, precise, mantissas = _mantissa_pair(shapes, device=device)
    kwargs = dict(_KWARGS, nesterov=nesterov)
    ref_opt = BatchedMuon(reference, **kwargs)
    man_opt = BatchedMuon(precise, use_mantissa=True, **kwargs)
    for p, low in zip(precise, mantissas):
        man_opt.state[p]['mantissa'] = low

    results = []
    torch.manual_seed(99)
    for _ in range(steps):
        # Identical values with real BF16 .grad storage on the mantissa arm.
        grads = [torch.randn_like(p).bfloat16().float() for p in reference]
        for p, g in zip(reference, grads):
            p.grad = g.clone()
        for p, g in zip(precise, grads):
            p.grad = g.bfloat16()
        ref_opt.step()
        man_opt.step()
        for pa, pb in zip(reference, precise):
            reconstructed = _reconstruct_precise(pb, man_opt.state[pb]['mantissa'])
            eq_param = torch.equal(reconstructed, pa)
            eq_momentum = torch.equal(
                ref_opt.state[pa]['momentum_buffer'],
                man_opt.state[pb]['momentum_buffer'],
            )
            results.append((eq_param, eq_momentum))
    return reference, precise, man_opt, results


def test_split_reconstruct_bit_exact():
    torch.manual_seed(1234)
    values = [
        torch.randn(4096),
        torch.randn(4096) * 1e-3,
        torch.randn(4096) * 1e3,
        torch.randn(4096) * -5.0,
        torch.tensor([0.0, -0.0, 1e-30, -1e-30, 123.456, -123.456, 3.3895314e38]),
    ]
    for value in values:
        value = value.float().contiguous()
        high, low = _split_precise(value)
        assert high.dtype == torch.bfloat16
        assert low.dtype == torch.uint16
        assert torch.equal(_reconstruct_precise(high, low), value)

    # Truncating split: a value whose low bits would round the bf16 up must keep
    # the lower bf16 neighbour, and the low buffer must carry the remainder.
    precise = torch.tensor([1.0 + 2.0 ** -8], dtype=torch.float32)
    high, low = _split_precise(precise)
    assert high.item() == 1.0
    assert torch.equal(_reconstruct_precise(high, low), precise)


@pytest.mark.parametrize('nesterov', [True, False])
def test_mantissa_tracks_fp32_reference_bit_exact(nesterov):
    _, _, _, results = _run_trajectory('cpu', nesterov=nesterov)
    assert len(results) == 6 * 5
    assert all(eq_param for eq_param, _ in results)
    assert all(eq_momentum for _, eq_momentum in results)


def test_naive_bf16_params_lose_updates():
    torch.manual_seed(21)
    shapes = [(12, 8)] * 3 + [(8, 8)] * 2
    reference = [torch.nn.Parameter(torch.randn(s)) for s in shapes]
    naive = [torch.nn.Parameter(p.detach().to(torch.bfloat16)) for p in reference]
    ref_opt = BatchedMuon(reference, **_KWARGS)
    naive_opt = BatchedMuon(naive, **_KWARGS)

    torch.manual_seed(99)
    for _ in range(8):
        grads = [torch.randn_like(p).bfloat16().float() for p in reference]
        for p, g in zip(reference, grads):
            p.grad = g.clone()
        for p, g in zip(naive, grads):
            p.grad = g.clone().to(torch.bfloat16)
        ref_opt.step()
        naive_opt.step()

    num = sum(((pa - pb.float()).norm() ** 2 for pa, pb in zip(reference, naive))) ** 0.5
    den = sum((pa.norm() ** 2 for pa in reference)) ** 0.5
    relative = (num / den.clamp_min(1e-12)).item()
    assert relative > 1e-3


def test_mantissa_state_resume():
    torch.manual_seed(31)
    shapes = [(10, 6)] * 2 + [(6, 6)]
    base = [torch.nn.Parameter(torch.randn(s).to(torch.bfloat16)) for s in shapes]
    opt = BatchedMuon(base, use_mantissa=True, **_KWARGS)

    torch.manual_seed(101)
    grads = [[torch.randn_like(p) for p in base] for _ in range(5)]

    for batch in grads[:3]:
        for p, g in zip(base, batch):
            p.grad = g.clone()
        opt.step()
    snapshot = [p.detach().clone() for p in base]
    saved = copy.deepcopy(opt.state_dict())

    for batch in grads[3:]:
        for p, g in zip(base, batch):
            p.grad = g.clone()
        opt.step()

    resumed = [torch.nn.Parameter(p.clone()) for p in snapshot]
    resumed_opt = BatchedMuon(resumed, use_mantissa=True, **_KWARGS)
    resumed_opt.load_state_dict(copy.deepcopy(saved))
    for batch in grads[3:]:
        for p, g in zip(resumed, batch):
            p.grad = g.clone()
        resumed_opt.step()

    for p, q in zip(base, resumed):
        assert torch.equal(p, q)
        mantissa = opt.state[p]['mantissa']
        resumed_mantissa = resumed_opt.state[q]['mantissa']
        assert mantissa.dtype == torch.uint16
        assert resumed_mantissa.dtype == torch.uint16
        assert torch.equal(mantissa, resumed_mantissa)


def test_mantissa_resume_from_legacy_bf16_state():
    torch.manual_seed(51)
    shapes = [(8, 4)] * 2 + [(4, 4)]
    legacy = [torch.nn.Parameter(torch.randn(s).to(torch.bfloat16)) for s in shapes]
    legacy_opt = BatchedMuon(legacy, lr=5e-4)

    torch.manual_seed(151)
    for _ in range(2):
        for p in legacy:
            p.grad = torch.randn_like(p)
        legacy_opt.step()
    saved = copy.deepcopy(legacy_opt.state_dict())

    resumed = [torch.nn.Parameter(p.detach().clone()) for p in legacy]
    opt = BatchedMuon(resumed, lr=5e-4, use_mantissa=True)
    opt.load_state_dict(copy.deepcopy(saved))
    # A legacy checkpoint keeps bf16 momentum; it is promoted on the next step.
    for p in resumed:
        assert opt.state[p]['momentum_buffer'].dtype == torch.bfloat16

    for p in resumed:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in resumed:
        assert opt.state[p]['momentum_buffer'].dtype == torch.float32
        assert opt.state[p]['mantissa'].dtype == torch.uint16
        assert torch.isfinite(p).all()


def test_mantissa_lazy_init_from_legacy_state():
    torch.manual_seed(41)
    p = torch.nn.Parameter(torch.randn(8, 4).to(torch.bfloat16))
    opt = BatchedMuon([p], use_mantissa=True, **_KWARGS)
    assert 'mantissa' not in opt.state[p]
    p.grad = torch.randn_like(p)
    opt.step()
    assert 'mantissa' in opt.state[p]
    assert opt.state[p]['mantissa'].dtype == torch.uint16
    assert torch.isfinite(p).all()


def test_default_off_and_factory_env(monkeypatch):
    from trainer.trainer_utils import (
        CombinedOptimizer,
        _is_muon_hidden_matrix,
        build_optimizer,
        cast_training_parameters,
        estimate_training_persistent_bytes,
    )

    monkeypatch.delenv('INSTINCT_MUON_MANTISSA', raising=False)
    p = torch.nn.Parameter(torch.randn(8, 8))
    default = BatchedMuon([p], lr=5e-4)
    assert default.use_mantissa is False
    p.grad = torch.randn_like(p)
    default.step()
    assert 'mantissa' not in default.state[p]

    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    monkeypatch.setenv('INSTINCT_MUON_BACKEND', 'batched')
    model = cast_training_parameters(torch.nn.Linear(8, 12), 'bf16')
    opt = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
    assert isinstance(opt, CombinedOptimizer)
    assert isinstance(opt.optimizers[0], BatchedMuon)
    assert opt.optimizers[0].use_mantissa is True
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    opt.step()
    assert all(torch.isfinite(param).all() for param in model.parameters())

    args = types.SimpleNamespace(optimizer='muon')
    monkeypatch.delenv('INSTINCT_MUON_MANTISSA', raising=False)
    off = estimate_training_persistent_bytes(model, args)
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    on = estimate_training_persistent_bytes(model, args)
    expected_delta = 4 * sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.element_size() == 2 and _is_muon_hidden_matrix(name, param)
    )
    assert expected_delta > 0
    assert on > off
    assert on - off == expected_delta


@pytest.mark.parametrize('versioned', [False, True])
def test_resume_rejects_discarding_mantissa(versioned):
    p = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    opt = BatchedMuon([p], use_mantissa=True)
    p.grad = torch.ones_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    if not versioned:
        saved.pop('muon_precision')
    for dtype in (torch.bfloat16, torch.float32):
        q = torch.nn.Parameter(p.detach().to(dtype))
        target = BatchedMuon([q])
        with pytest.raises(ValueError, match='refusing to discard'):
            target.load_state_dict(saved)
        assert not target.state


def test_fresh_mantissa_checkpoint_also_records_mode():
    p = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    saved = BatchedMuon([p], use_mantissa=True).state_dict()
    with pytest.raises(ValueError, match='refusing to discard'):
        BatchedMuon([p]).load_state_dict(saved)


@pytest.mark.parametrize('corruption', ['dtype', 'shape', 'missing', 'format'])
def test_resume_rejects_corrupt_mantissa(corruption):
    p = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    opt = BatchedMuon([p], use_mantissa=True)
    opt.reserve_momentum()
    saved = copy.deepcopy(opt.state_dict())
    if corruption == 'dtype':
        saved['state'][0]['mantissa'] = saved['state'][0]['mantissa'].to(torch.bfloat16)
    elif corruption == 'shape':
        saved['state'][0]['mantissa'] = torch.zeros(1, dtype=torch.uint16)
    elif corruption == 'missing':
        del saved['state'][0]['mantissa']
    else:
        saved['muon_precision'] = 'unknown'
    with pytest.raises(ValueError):
        BatchedMuon([p], use_mantissa=True).load_state_dict(saved)


def test_serialized_resume_preserves_bits_and_missing_gradients():
    torch.manual_seed(19)
    params = [torch.nn.Parameter(torch.randn(8, 4).bfloat16()) for _ in range(2)]
    opt = BatchedMuon(params, use_mantissa=True)
    assert opt.reserve_momentum() == sum(p.numel() * 6 for p in params)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    stream = io.BytesIO()
    torch.save({'weights': [p.detach().clone() for p in params], 'opt': opt.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    copies = [torch.nn.Parameter(p.clone()) for p in saved['weights']]
    restored = BatchedMuon(copies, use_mantissa=True)
    restored.load_state_dict(saved['opt'])
    assert restored.reserve_momentum() == 0
    before = copies[1].detach().clone()
    for i, (p, q) in enumerate(zip(params, copies)):
        p.grad = torch.randn_like(p) if i == 0 else None
        q.grad = None if p.grad is None else p.grad.clone()
    opt.step()
    restored.step()
    for p, q in zip(params, copies):
        assert torch.equal(p, q)
        for key in ('mantissa', 'momentum_buffer'):
            assert torch.equal(opt.state[p][key], restored.state[q][key])
            assert opt.state[p][key].dtype == restored.state[q][key].dtype
    assert torch.equal(copies[1], before)


def test_factory_protects_auxiliary_params_and_small_lr(monkeypatch):
    from trainer.trainer_utils import build_optimizer, cast_training_parameters
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    model = torch.nn.Linear(8, 8).bfloat16()
    with pytest.raises(ValueError, match='AdamW parameters'):
        build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
    model = cast_training_parameters(torch.nn.Linear(8, 8), 'bf16')
    assert model.weight.dtype == torch.bfloat16
    assert model.bias.dtype == torch.float32
    opt = build_optimizer(model.named_parameters(), lr=1e-6, optimizer='muon')
    assert opt.optimizers[0].use_mantissa
    monkeypatch.setenv('INSTINCT_MUON_BACKEND', 'native')
    with pytest.raises(ValueError, match='BACKEND=batched'):
        build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', 'typo')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')


def test_combined_and_native_resume_reject_mantissa(monkeypatch):
    from trainer.trainer_utils import build_optimizer, cast_training_parameters
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    model = cast_training_parameters(torch.nn.Linear(8, 8), 'bf16')
    saved = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon').state_dict()
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '0')
    for backend in ('batched', 'native'):
        monkeypatch.setenv('INSTINCT_MUON_BACKEND', backend)
        target = build_optimizer(torch.nn.Linear(8, 8).named_parameters(), lr=5e-4, optimizer='muon')
        with pytest.raises(ValueError, match='refusing to discard'):
            target.load_state_dict(saved)
        with pytest.raises(ValueError, match='refusing to discard'):
            target.optimizers[0].load_state_dict(saved['optimizers'][0])
        changed = copy.deepcopy(saved)
        changed['optimizers'][0]['param_groups'][0]['params'].append(99)
        with pytest.raises(ValueError, match='parameter groups changed'):
            target.load_state_dict(changed)


@pytest.mark.gpu
@pytest.mark.parametrize('nesterov', [True, False])
def test_mantissa_tracks_fp32_reference_bit_exact_cuda(nesterov):
    _, _, _, results = _run_trajectory('cuda', nesterov=nesterov)
    assert all(eq_param for eq_param, _ in results)
    assert all(eq_momentum for _, eq_momentum in results)


@pytest.mark.gpu
def test_real_expert_shapes_cuda_under_autocast():
    torch.manual_seed(73)
    shapes = [(1664, 512)] * 2 + [(512, 1664)] * 2 + [(512, 512)] * 2
    reference, precise, mantissas = _mantissa_pair(shapes, device='cuda')
    ref_opt = BatchedMuon(reference, lr=5e-4, adjust_lr_fn='match_rms_adamw')
    man_opt = BatchedMuon(precise, use_mantissa=True, lr=5e-4,
                          adjust_lr_fn='match_rms_adamw')
    for p, low in zip(precise, mantissas):
        man_opt.state[p]['mantissa'] = low

    torch.manual_seed(211)
    for _ in range(3):
        grads = [torch.randn_like(p).bfloat16().float() for p in reference]
        for p, g in zip(reference, grads):
            p.grad = g.clone()
        for p, g in zip(precise, grads):
            p.grad = g.bfloat16()
        ref_opt.step()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            man_opt.step()
        for pa, pb in zip(reference, precise):
            assert torch.isfinite(pb).all()
            reconstructed = _reconstruct_precise(pb, man_opt.state[pb]['mantissa'])
            # Every op on the precise path is the same fp32 op as the reference,
            # so bitwise equality holds on CUDA too; keep a close tolerance as a
            # safety net if a future backend reorders reductions.
            if not torch.equal(reconstructed, pa):
                torch.testing.assert_close(reconstructed, pa, rtol=1e-6, atol=0)


def test_experimental_bf16_momentum_matches_value_only_simulation():
    from experiments.audit_muon_precision import BF16MomentumMuon
    torch.manual_seed(41)
    p = torch.nn.Parameter(torch.randn(8, 4).bfloat16())
    q = torch.nn.Parameter(p.detach().clone())
    simulated = BatchedMuon([p], use_mantissa=True)
    actual = BF16MomentumMuon([q])
    for _ in range(6):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        simulated.step()
        actual.step()
        momentum = simulated.state[p]['momentum_buffer']
        momentum.copy_(momentum.bfloat16())
        assert torch.equal(p, q)
        assert torch.equal(simulated.state[p]['mantissa'], actual.state[q]['mantissa'])
        assert torch.equal(momentum, actual.state[q]['momentum_buffer'].float())
        assert actual.state[q]['momentum_buffer'].dtype == torch.bfloat16
    # Research momentum checkpoints must not silently change precision on resume.
    with pytest.raises(ValueError, match='Unsupported Muon mantissa'):
        BatchedMuon([q], use_mantissa=True).load_state_dict(actual.state_dict())


def test_mixed_parameter_cast_preserves_tied_fp32_weights(monkeypatch):
    from trainer.trainer_utils import cast_training_parameters
    from model.model_instinct import InstinctForCausalLM
    from tests.helpers import make_tiny_config
    model = InstinctForCausalLM(make_tiny_config())
    before = model.model.embed_tokens.weight.detach().clone()
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    cast_training_parameters(model, 'bf16')
    assert model.model.embed_tokens.weight is model.lm_head.weight
    assert model.lm_head.weight.dtype == torch.float32
    assert torch.equal(model.lm_head.weight, before)
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
