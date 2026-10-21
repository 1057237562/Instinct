"""Numerical and lifecycle gates for the optional exact-order CUDA Muon path."""
import copy
import pytest
import torch
from trainer.batched_muon import BatchedMuon, _batched_zeropower_reference, _NS_COEFFICIENTS


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [(4,1664,512),(4,512,1664),(10,512,512),(16,128,512)])
@pytest.mark.parametrize('kind', ['random', 'zero', 'rank_one'])
def test_ns_kernel_matches_reference(shape, kind):
    from trainer.muon_kernels import zeropower, ZeropowerGraphCache
    torch.manual_seed(53)
    x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    if kind=='zero':
        x.zero_()
    elif kind=='rank_one':
        x=x[:,:,:1].expand(shape).contiguous()
    opts=dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7)
    expected=_batched_zeropower_reference(x,**opts)
    torch.testing.assert_close(zeropower(x,**opts),expected,rtol=0,atol=0)
    cache=ZeropowerGraphCache(limit=1)
    torch.testing.assert_close(cache(x,**opts),expected,rtol=0,atol=0)
    # Changed allocation/data and scalar settings must not reuse stale inputs.
    x=torch.randn_like(x)
    opts.update(ns_steps=3,eps=1e-6)
    torch.testing.assert_close(cache(x,**opts),_batched_zeropower_reference(x,**opts),rtol=0,atol=0)
    assert len(cache.entries)==1


@pytest.mark.gpu
@pytest.mark.parametrize('fused', ['0','1'])
def test_optimizer_graph_dynamic_lr_missing_grad_and_resume(monkeypatch, fused):
    torch.manual_seed(57)
    params=[torch.nn.Parameter(torch.randn(512,1664,device='cuda')) for _ in range(5)]
    copies=[torch.nn.Parameter(p.detach().clone()) for p in params]
    ref=BatchedMuon(params,lr=5e-4)
    new=BatchedMuon(copies,lr=5e-4)
    for step in range(6):
        for i,(p,q) in enumerate(zip(params,copies)):
            grad=None if i==step else torch.randn_like(p)
            p.grad=grad
            q.grad=None if grad is None else grad.clone()
        for opt in (ref,new):
            opt.param_groups[0]['lr']=[5e-4,1e-4,0,3e-4,2e-4,1e-5][step]
        monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND','torch')
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES','0')
        ref.step()
        monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND','triton_graph')
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES',fused)
        new.step()
        for p,q in zip(params,copies):
            torch.testing.assert_close(q,p,rtol=0,atol=0)
            if p in ref.state:
                torch.testing.assert_close(new.state[q]['momentum_buffer'],ref.state[p]['momentum_buffer'],rtol=0,atol=0)
        if step==2:
            state=copy.deepcopy(new.state_dict())
            assert set(state)=={'state','param_groups'}
            new.load_state_dict(state)


def test_torch_fallback_cpu_and_backend_validation(monkeypatch):
    from trainer.batched_muon import _batched_zeropower
    x=torch.randn(2,12,8)
    opts=dict(ns_coefficients=_NS_COEFFICIENTS,ns_steps=3,eps=1e-7)
    monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND','auto')
    torch.testing.assert_close(_batched_zeropower(x,**opts),_batched_zeropower_reference(x,**opts),rtol=0,atol=0)
    monkeypatch.setenv('INSTINCT_MUON_NS_BACKEND','wrong')
    with pytest.raises(ValueError,match='BACKEND'):
        _batched_zeropower(x,**opts)


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [(1664,512),(512,1664)])
def test_expert_pack_forward_backward_and_weight_replacement(shape):
    from model.expert_weight_pack import pack_weights
    weights=[torch.randn(shape,device='cuda',requires_grad=True) for _ in range(8)]
    for _ in range(2):
        actual=pack_weights(weights,torch.bfloat16)
        expected=torch.stack(weights).transpose(1,2).bfloat16()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        grad=torch.randn_like(actual.transpose(1,2)).transpose(1,2)
        ga=torch.autograd.grad(actual,weights,grad)
        ge=torch.autograd.grad(expected,weights,grad)
        for a,e in zip(ga,ge):
            torch.testing.assert_close(a,e,rtol=0,atol=0)
        weights=[(w.detach()+0.25).requires_grad_() for w in weights]


@pytest.mark.gpu
def test_swiglu_all_finite_bf16_inputs_and_strided_gradient():
    from model.swiglu_triton import swiglu
    # Enumerate the BF16 domain rather than testing only a small normal range.
    values=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gate=values[torch.isfinite(values)].reshape(-1,1).requires_grad_()
    torch.manual_seed(91)
    up=torch.randn_like(gate).requires_grad_()
    actual=swiglu(gate,up)
    expected=torch.nn.functional.silu(gate)*up
    gradient=torch.randn(gate.numel(),2,device='cuda',dtype=torch.bfloat16)[:,:1]
    ag=torch.autograd.grad(actual,(gate,up),gradient)
    eg=torch.autograd.grad(expected,(gate,up),gradient)
    for a,e in [(actual,expected),*zip(ag,eg)]:
        torch.testing.assert_close(a,e,rtol=0,atol=0,equal_nan=True)


@pytest.mark.gpu
@pytest.mark.parametrize('momentum', [0.,0.5,0.8,0.95,1.])
@pytest.mark.parametrize('nesterov', [True,False])
def test_fused_update_preserves_lerp_and_decay_rounding(momentum,nesterov):
    from trainer.muon_kernels import prepare_update,apply_update
    torch.manual_seed(37)
    grads=[torch.randn(1664,512,device='cuda') for _ in range(2)]
    buffers=[torch.randn_like(g) for g in grads]
    reference_buffers=[b.clone() for b in buffers]
    torch._foreach_lerp_(reference_buffers,grads,1-momentum)
    updates=(torch._foreach_lerp(grads,reference_buffers,momentum)
             if nesterov else reference_buffers)
    expected=torch.stack(updates).bfloat16()
    actual=prepare_update(grads,buffers,momentum=momentum,nesterov=nesterov)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,e in zip(buffers,reference_buffers):
        torch.testing.assert_close(a,e,rtol=0,atol=0)
    params=[torch.randn_like(g) for g in grads]
    reference=[p.clone() for p in params]
    torch._foreach_mul_(reference,0.99995)
    torch._foreach_add_(reference,list(expected.float().unbind()),alpha=-0.0012345)
    apply_update(params,actual,decay=0.99995,alpha=-0.0012345)
    for a,e in zip(params,reference):
        torch.testing.assert_close(a,e,rtol=0,atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('momentum', [0., 0.5, 0.8, 0.95, 1.])
@pytest.mark.parametrize('nesterov', [True, False])
def test_fused_bf16_grad_fp32_momentum_bitwise(momentum, nesterov):
    from trainer.muon_kernels import prepare_update
    torch.manual_seed(127)
    # Non-block-aligned size tests masked tail loads/stores too.
    grads = [torch.randn(19, 71, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    buffers = [torch.randn_like(g, dtype=torch.float32) for g in grads]
    expected = [b.clone() for b in buffers]
    promoted = [g.float() for g in grads]
    torch._foreach_lerp_(expected, promoted, 1-momentum)
    update = torch.stack(torch._foreach_lerp(promoted, expected, momentum) if nesterov else expected).bfloat16()
    actual = prepare_update(grads, buffers, momentum=momentum, nesterov=nesterov)
    assert torch.equal(actual, update)
    for a, b in zip(buffers, expected):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.gpu
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('decay,alpha', [(0.99995, -0.0012345), (1., 0.), (0., -1.)])
def test_fused_mantissa_preserves_raw_bits(transposed, decay, alpha):
    from trainer.batched_muon import _split_precise, _reconstruct_precise
    from trainer.muon_kernels import apply_mantissa_update
    torch.manual_seed(59)
    # Cover signs, exponent range, subnormals and both signed zeros. Exclude
    # NaN/Inf from arithmetic equality, whose payload is unspecified.
    bits = torch.randint(-(2**31), 2**31-1, (2, 131, 257), device='cuda', dtype=torch.int32)
    values = bits.view(torch.float32)
    values = torch.where(torch.isfinite(values), values, 0.)
    values.view(-1)[:6] = torch.tensor([0., -0., 1e-44, -1e-44, 1e-37, -1e-37], device='cuda')
    pairs = [_split_precise(v) for v in values.unbind()]
    params, lows = [v[0] for v in pairs], [v[1] for v in pairs]
    update = (torch.randn(2, 257, 131, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
              if transposed else torch.randn_like(values, dtype=torch.bfloat16))
    reference = [v.clone() for v in values.unbind()]
    torch._foreach_mul_(reference, decay)
    torch._foreach_add_(reference, list(update.float().unbind()), alpha=alpha)
    versions = [p._version for p in (*params, *lows)]
    apply_mantissa_update(params, lows, update, decay=decay, alpha=alpha)
    assert [p._version for p in (*params, *lows)] == [v+1 for v in versions]
    for p, low, expected in zip(params, lows, reference):
        actual = _reconstruct_precise(p, low)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@pytest.mark.parametrize('nesterov', [True, False])
def test_mantissa_fused_trajectory_resume_and_grad_none(monkeypatch, nesterov):
    from trainer.batched_muon import _split_precise
    torch.manual_seed(137)
    shapes = [(1664,512), (512,1664), (512,512), (128,512), (8,512)]
    shapes = shapes * 2
    originals = [_split_precise(torch.randn(s, device='cuda') * .02) for s in shapes]
    params = [torch.nn.Parameter(x[0].clone()) for x in originals]
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    ref = BatchedMuon(params, use_mantissa=True, nesterov=nesterov, batch_size=3)
    opt = BatchedMuon(copies, use_mantissa=True, nesterov=nesterov, batch_size=3)
    for p, q, (_, low) in zip(params, copies, originals):
        ref.state[p]['mantissa'] = low.clone()
        opt.state[q]['mantissa'] = low.clone()
    pointers = [p.data_ptr() for p in copies]
    for step in range(12):
        for i, (p, q) in enumerate(zip(params, copies)):
            p.grad = None if step % 4 == i % 4 else torch.randn_like(p)
            q.grad = p.grad.clone() if p.grad is not None else None
        for instance in (ref, opt):
            instance.param_groups[0]['lr'] = [0., 5e-4, 1e-6][step % 3]
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', '0')
        ref.step()
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', '1')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            opt.step()
        assert pointers == [p.data_ptr() for p in copies]
        for p, q in zip(params, copies):
            assert torch.equal(p.view(torch.int16), q.view(torch.int16))
            for key, value in ref.state[p].items():
                assert torch.equal(value, opt.state[q][key])
        if step == 5:
            opt.load_state_dict(copy.deepcopy(opt.state_dict()))
            opt.reserve_momentum()


@pytest.mark.gpu
def test_mantissa_noncontiguous_falls_back(monkeypatch):
    torch.manual_seed(139)
    p = torch.nn.Parameter(torch.randn(64, 128, device='cuda').bfloat16().T)
    q = torch.nn.Parameter(p.detach().clone())
    assert not p.is_contiguous()
    ref, opt = BatchedMuon([p], use_mantissa=True), BatchedMuon([q], use_mantissa=True)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', '0')
        ref.step()
        monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', '1')
        opt.step()
        assert torch.equal(p, q)
        for key, value in ref.state[p].items():
            assert torch.equal(value, opt.state[q][key])


@pytest.mark.gpu
def test_fused_mantissa_invalidates_inference_weight_stack(monkeypatch):
    from model.moe_dispatch import _stacked_expert_weights
    experts = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    for expert in experts:
        expert.gate_proj = torch.nn.Linear(16, 8, bias=False, device='cuda', dtype=torch.bfloat16)
    experts._inference_stack_cache = {}
    params = [e.gate_proj.weight for e in experts]
    with torch.no_grad():
        cached = _stacked_expert_weights(experts, 'gate_proj', torch.bfloat16)
    opt = BatchedMuon(params, use_mantissa=True, lr=.1, ns_steps=0)
    for p in params:
        p.grad = torch.ones_like(p)
    monkeypatch.setenv('INSTINCT_MUON_FUSED_UPDATES', '1')
    opt.step()
    with torch.no_grad():
        fresh = _stacked_expert_weights(experts, 'gate_proj', torch.bfloat16)
    assert fresh is not cached
    assert not torch.equal(cached, fresh)
    assert torch.equal(fresh, torch.stack(params).transpose(1, 2))
