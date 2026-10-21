"""Numerical and checkpoint contracts for the optional top-2 CUDA fast paths."""
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize('tokens,top,skew', [(17,1,False),(17,2,False),
    (8192,2,False),(11712,2,False),(8192,2,True)])
def test_stable_routing_and_gradients(tokens,top,skew):
    from model.moe_routing_triton import permutation,gather,combine
    torch.manual_seed(171)
    ids=torch.rand(tokens,8,device='cuda').topk(top,dim=-1).indices
    if skew: ids=torch.arange(top,device='cuda').expand(tokens,-1).contiguous()
    order,inv,ends=permutation(ids,8)
    expected_order=ids.flatten().argsort(stable=True)
    assert torch.equal(order.long(),expected_order)
    assert torch.equal(inv[order.long()],torch.arange(tokens*top,device='cuda',dtype=torch.int32))
    assert torch.equal(ends,torch.bincount(ids.flatten(),minlength=8).cumsum(0))
    token=expected_order//top
    x=torch.randn(tokens,512,device='cuda',requires_grad=True)
    xx=x.detach().clone().requires_grad_(True)
    a=gather(x,order,inv,top); b=xx[token].bfloat16()
    assert torch.equal(a,b)
    dy=torch.randn_like(a); a.backward(dy); b.backward(dy)
    assert torch.equal(x.grad,xx.grad)
    y=torch.randn(tokens*top,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    yy=y.detach().clone().requires_grad_(True)
    w=torch.rand(tokens,top,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    ww=w.detach().clone().requires_grad_(True)
    a=combine(y,w,order,inv,torch.float32)
    b=torch.zeros(tokens,512,device='cuda').index_add_(0,token,(yy*ww.flatten()[expected_order,None]).float())
    assert torch.equal(a,b)
    dy=torch.randn_like(a); a.backward(dy); b.backward(dy)
    assert torch.equal(y.grad,yy.grad)
    assert torch.equal(w.grad,ww.grad)


@pytest.mark.gpu
@pytest.mark.parametrize('length',[128,513,4096])
def test_causal_flex_output_gradient_and_isolation(length):
    from model.packed_attention import causal_flex_attention
    torch.manual_seed(18)
    x=[torch.randn(1,length,h,32,device='cuda',dtype=torch.bfloat16,requires_grad=True) for h in (16,4,4)]
    ref=[t.detach().clone().requires_grad_(True) for t in x]
    a=causal_flex_attention(*x)
    # Direct SDPA oracle with compact heads expanded as in the historical path.
    q,k,v=ref
    b=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),
        k.repeat_interleave(4,dim=2).transpose(1,2),v.repeat_interleave(4,dim=2).transpose(1,2),
        is_causal=True).transpose(1,2)
    assert (a.float()-b.float()).norm()/b.float().norm()<.005
    g=torch.randn_like(a); a.backward(g); b.backward(g)
    for t,r in zip(x,ref):
        assert (t.grad.float()-r.grad.float()).norm()/r.grad.float().norm()<.02
    altered=[t.detach().clone() for t in x]
    for t in altered: t[:,length//2:]+=3
    c=causal_flex_attention(*altered)
    assert torch.equal(a[:,:length//2],c[:,:length//2])


@pytest.mark.gpu
@pytest.mark.parametrize('counts',[[2048]*8,[0,100,500,4000,2000,4000,3000,2784]])
def test_weight_gradient_direct_layout_is_exact(counts):
    from model.grouped_mm_triton import ragged_weight_grad
    torch.manual_seed(111)
    ends=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
    x=torch.randn(sum(counts),512,device='cuda',dtype=torch.bfloat16)
    dy=torch.randn(sum(counts),1664,device='cuda',dtype=torch.bfloat16)
    ref=ragged_weight_grad(x,dy,ends,8)
    got=ragged_weight_grad(x,dy,ends,8,transposed_output=True)
    assert torch.equal(ref,got)
    assert got.transpose(1,2).is_contiguous()


@pytest.mark.gpu
@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('setting', ['1', '2'])
def test_inner_checkpoint_full_moe_gradients(monkeypatch, skew, setting):
    from model.model_instinct import InstinctConfig, MOEFeedForward
    from model.checkpointing import checkpoint_ffn
    from model.moe_dispatch import inner_checkpoint_eligible
    torch.manual_seed(731)
    config=InstinctConfig(hidden_size=512,num_experts=8,num_experts_per_tok=2,
                          norm_topk_prob=True,moe_intermediate_size=1664,use_grad_checkpoint=1)
    moe=MOEFeedForward(config).cuda().train()
    if skew:
        with torch.no_grad():
            moe.gate.weight.zero_()
    x=torch.randn(1,513,512,device='cuda',requires_grad=True)
    dy=torch.randn_like(x)
    original_dy=dy.clone()
    def run(setting):
        monkeypatch.setenv('INSTINCT_MOE_INNER_CHECKPOINT',setting)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y,aux=checkpoint_ffn(moe,x)
        grads=torch.autograd.grad((y,aux),(x,*moe.parameters()),(dy,torch.ones_like(aux)))
        return y,aux,grads
    ref,ra,rg=run('0')
    got,ga,gg=run(setting)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        assert inner_checkpoint_eligible(x,moe.experts,moe.act_fn,1)
    assert torch.equal(ref,got)
    assert torch.equal(ra,ga)
    assert all(torch.equal(a,b) for a,b in zip(rg,gg))
    assert torch.equal(dy,original_dy)  # Only the operator-owned temporary may be reused.
    monkeypatch.setenv('INSTINCT_GROUPED_MM_BACKEND','cached')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        assert not inner_checkpoint_eligible(x,moe.experts,moe.act_fn,1)
