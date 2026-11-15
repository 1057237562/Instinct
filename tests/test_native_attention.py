"""Fused attention oracle checks, including the exact replayed dropout mask."""
import numpy as np
import pytest
import datasets  # Windows pyarrow/torch DLL order.
import torch
import triton
import triton.language as tl
from model.dense_triton.runtime import Scalar
from model.dense_cublas.autograd import Engine
from tests.test_cublas_training import HostBackend
from tests.test_dense_triton import kernel_test, close, random


@kernel_test
@pytest.mark.parametrize('dim,probability', [(96,.2),(160,0.),(192,.25),(256,.5)])
def test_fused_dropout_large_heads_forward_backward(dim,probability):
    e=Engine(HostBackend(),seed=911)
    q=random((2,35,4,dim),scale=.2).requires_grad_()
    k=random((2,35,2,dim),seed=4,scale=.2).requires_grad_()
    v=random((2,35,2,dim),seed=5,scale=.2).requires_grad_()
    mask=torch.ones(2,35,dtype=torch.long);mask[0,:3]=0;mask[1,:]=0
    # Use the public elementwise RNG operation to independently materialize the
    # reference mask. The actual attention never materializes S x S on device.
    e.counter=1234
    keep=e.numpy(e.dropout(e.full((2,4,35,35),1.),probability))
    e.counter=1234
    a,b,c=[e.array(x.detach().numpy(),requires_grad=True) for x in (q,k,v)]
    out=e.attention(a,b,c,e.array(mask.numpy()),probability)
    assert e.counter==1234+(2*4*35*35 if probability else 0)
    kk=k.repeat_interleave(2,2).transpose(1,2);vv=v.repeat_interleave(2,2).transpose(1,2)
    scores=q.transpose(1,2)@kk.transpose(-1,-2)/dim**.5
    allowed=torch.ones(35,35,dtype=torch.bool).tril()[None,None]&mask[:,None,None,:].bool()
    probs=scores.masked_fill(~allowed,-float('inf')).softmax(-1).nan_to_num()
    expected=((probs*torch.from_numpy(keep))@vv).transpose(1,2)
    # Fully masked rows must have zero outputs and gradients.
    expected.backward(torch.ones_like(expected));grads=e.vjp(out.sum())
    close(out.data,expected,atol=3e-5)
    for tensor,ref in zip((a,b,c),(q,k,v)):close(grads[tensor],ref.grad,atol=6e-5)


@kernel_test
def test_fused_attention_checkpoint_dropout_rng():
    def run(recompute):
        e=Engine(HostBackend(),seed=29)
        q=e.array(np.random.default_rng(2).normal(size=(1,9,2,160)).astype('float32')*.1,requires_grad=True)
        k=e.array(np.random.default_rng(3).normal(size=(1,9,1,160)).astype('float32')*.1,requires_grad=True)
        v=e.array(np.random.default_rng(4).normal(size=(1,9,1,160)).astype('float32')*.1,requires_grad=True)
        fn=lambda z:e.attention(z,k,v,dropout=.3)
        out=e.checkpoint(fn,q,[k,v]) if recompute else fn(q)
        g=e.vjp(out.sum())
        return e.numpy(out),[e.numpy(g[x]) for x in (q,k,v)],e.counter
    a,ga,ca=run(False);b,gb,cb=run(True)
    np.testing.assert_array_equal(a,b)
    for x,y in zip(ga,gb):np.testing.assert_allclose(x,y,atol=1e-6)
    assert ca==cb


@triton.jit
def _reference_mask(OUT, seed, offset, N:tl.constexpr, P:tl.constexpr, BLOCK:tl.constexpr=256):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    keep=tl.rand(seed,i.to(tl.int64)+offset)>=P
    tl.store(OUT+i,tl.where(keep,1./(1.-P),0.),i<N)


@kernel_test
def test_dropout_counter_crosses_uint32_boundary():
    e=Engine(HostBackend(),seed=19);shape=(1,17,2,32)
    q,k,v=[random(shape,seed=i,scale=.1).requires_grad_() for i in (1,2,3)]
    a,b,c=[e.array(x.detach().numpy(),requires_grad=True) for x in (q,k,v)]
    counter=2**32-7;e.counter=counter
    out=e.attention(a,b,c,dropout=.3)
    keep=e.full((1,2,17,17),0.)
    e.backend.launch(_reference_mask,(triton.cdiv(keep.size,256),),(keep.data,Scalar(19,'i32'),Scalar(counter,'i64')),N=keep.size,P=.3)
    scores=q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2)/32**.5
    probs=scores.masked_fill(~torch.ones(17,17,dtype=torch.bool).tril(),-float('inf')).softmax(-1)
    expected=((probs*torch.from_numpy(e.numpy(keep)))@v.transpose(1,2)).transpose(1,2)
    expected.sum().backward();grads=e.vjp(out.sum())
    close(out.data,expected,atol=2e-5)
    for x,y in zip((a,b,c),(q,k,v)):close(grads[x],y.grad,atol=2e-5)
    e.counter=counter+2**32
    distinct=e.attention(a,b,c,dropout=.3)
    assert not np.allclose(e.numpy(out),e.numpy(distinct))
