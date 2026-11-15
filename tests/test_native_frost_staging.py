"""Packed pointer offsets and padded scratch boundaries without a CUDA launch."""
import os
import numpy as np
import pytest


def test_frost_staging_preserves_gaps_lengths_and_gqa_offsets():
    if os.environ.get('TRITON_INTERPRET')!='1':pytest.skip('CPU interpreter only')
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.frost_attention import gather,scatter,gather_dy
    b=ReusingBackend();e=Engine(b);rng=np.random.default_rng(872)
    total,seq,h,hk,d=80,32,4,1,16;starts=(3,43);lengths=(19,27)
    offsets=e.array(np.asarray(starts,np.int64)*(h*d)).data
    lens=e.ops.cast(e.array(np.asarray(lengths,np.int64)).data,'i32')
    src=[e.array(rng.normal(size=(total,heads,d)).astype(np.float32),'bf16').data for heads in (h,hk,hk,h,h)]
    padded=[b.empty((2,seq,heads,d),'bf16') for heads in (h,hk,hk,h,h)]
    b.launch(gather,(8,2),(*src,*padded,offsets,lens),SEQ=seq,H=h,HK=hk,D=d,B=256)
    for original,actual in zip(src,padded):
        original=e.numpy(original);actual=e.numpy(actual)
        for i,(s,n) in enumerate(zip(starts,lengths)):
            np.testing.assert_array_equal(actual[i,:n],original[s:s+n])
            np.testing.assert_array_equal(actual[i,n:],0)
    staged_dy=b.empty(padded[-1].shape,'bf16');unchanged=e.numpy(src[-1]).copy()
    b.launch(gather_dy,(8,2),(src[-1],staged_dy,offsets,lens),SEQ=seq,H=h,D=d,B=256)
    np.testing.assert_array_equal(e.numpy(staged_dy),e.numpy(padded[-1]))
    np.testing.assert_array_equal(e.numpy(src[-1]),unchanged)
    out=[e.array(np.full((total,heads,d),-77,np.float32),'bf16').data for heads in (h,hk,hk)]
    b.launch(scatter,(8,2),(*padded[:3],*out,offsets,lens),SEQ=seq,H=h,HK=hk,D=d,B=256)
    for original,actual in zip(src,out):
        expected=np.full(actual.shape,-77,np.float32);original=e.numpy(original)
        for s,n in zip(starts,lengths):expected[s:s+n]=original[s:s+n]
        np.testing.assert_array_equal(e.numpy(actual),expected)
