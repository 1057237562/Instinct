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
