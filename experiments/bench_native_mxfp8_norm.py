"""MX scale boundary validation and a resource-bounded Norm tile shortlist."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
import triton.language as tl
from experiments.smoke_dense_triton import RejectTorch
from model.dense_cublas.mxfp8 import scale_exponent


@triton.jit
def check_scale(X,E,I,ER,IR,N:tl.constexpr,MAXIMUM:tl.constexpr,B:tl.constexpr=256):
    j=tl.program_id(0)*B+tl.arange(0,B);x=tl.load(X+j,j<N,0.)
    exponent,inverse=scale_exponent(x,MAXIMUM)
    raw=tl.div_rn(x,MAXIMUM).to(tl.uint32,bitcast=True)
    e=(raw>>23)&255;m=raw&0x7fffff
    e+=(((e>0)&(e<254)&(m!=0))|((e==0)&(m>0x400000))).to(tl.uint32)
    inv=tl.exp2(127.-e.to(tl.float32))
    tl.store(E+j,exponent,j<N);tl.store(I+j,inverse,j<N)
    tl.store(ER+j,e,j<N);tl.store(IR+j,inv,j<N)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--backward-only',action='store_true')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.mxfp8 import norm
    from experiments.validate_native_mxfp8 import reference
    report=dict(status='running',scales=[],cases=[])
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b);e=Engine(b,fp8=True);rng=np.random.default_rng(153)
        # Every finite nonnegative BF16 value, every FP32 power-of-two / FP8
        # threshold neighborhood, subnormals, and a reproducible FP32 sample.
        boundaries=np.concatenate([((np.arange(1,255,dtype=np.uint32)<<23)|m)[:,None]+np.arange(-4,5,dtype=np.int64)
                                   for m in (0,0x600000)])
        bits=np.unique(np.concatenate([np.arange(0x7f80,dtype=np.uint32)<<16,boundaries.ravel(),
            np.arange(20,dtype=np.uint32),rng.integers(0,0x7f800000,size=1_000_000,dtype=np.uint32)]).astype(np.uint32))
        values=bits.view(np.float32);values=values[np.isfinite(values)&(values>=0)]
        mark=b.mark();x=e.array(values).data
        outputs=[b.empty(x.shape,dtype) for dtype in ('i32','fp32','i32','fp32')]
        for maximum in (448.,57344.):
            b.launch(check_scale,(triton.cdiv(x.numel,256),),(x,*outputs),N=x.numel,MAXIMUM=maximum)
            exponent,inverse,expected_e,expected_i=[b.to_numpy(v) for v in outputs]
            raw=(values/np.float32(maximum)).view(np.uint32);cpu_e=(raw>>23)&255;m=raw&0x7fffff
            cpu_e+=(((cpu_e>0)&(cpu_e<254)&(m!=0))|((cpu_e==0)&(m>0x400000))).astype(np.uint32)
            cpu_i=np.exp2(np.float32(127)-cpu_e.astype(np.float32))
            np.testing.assert_array_equal(exponent,cpu_e)
            np.testing.assert_array_equal(inverse,cpu_i)
            # The legacy CUDA division flushes subnormal quotients to zero.
            # Above that range it must agree bitwise with the integer method.
            normal=values/np.float64(maximum)>=np.finfo(np.float32).tiny
            np.testing.assert_array_equal(exponent[normal],expected_e[normal])
            np.testing.assert_array_equal(inverse[normal],expected_i[normal])
            report['scales'].append(dict(fp8_max=maximum,values=x.numel,independent_reference_bitwise_equal=True,
                legacy_subnormal_differences=int(np.count_nonzero(exponent!=expected_e))))
        b.release_since(mark)
        shape=(12288,1536);x=e.array(rng.standard_normal(shape,dtype=np.float32));w=e.array(np.ones(1536,np.float32))
        addition=e.array(rng.standard_normal(shape,dtype=np.float32),'bf16')
        for residual in (() if a.backward_only else (False,True)):
            baseline=None
            # One warp per row minimizes synchronization; two/four rows amortize
            # scales' swizzled writes. This is a four-entry shortlist, not a sweep.
            for rows,warps in ((1,4),(1,1),(2,2),(4,4)):
                mark=b.mark();b.mxfp8_norm_forward_rows=rows
                b.kernel_options['model.dense_cublas.mxfp8.mxfp8_norm_forward']={'num_warps':warps}
                b._commands=[]
                try:z,y,inv=norm(e.ops,x.data,w.data,1e-6,addition.data if residual else None)
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                run();q,s=y.mx32_quantized
                actual=[b.to_numpy(v) for v in (z,y,inv,q,s)]
                if baseline is None:baseline=actual
                errors=[float(np.linalg.norm(v.astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-20))
                        for v,ref in zip(actual[:3],baseline[:3])]
                # Different warp reductions can cross a BF16 rounding tie.
                assert errors[0]==0 and errors[1]<2e-5 and errors[2]<1e-6,errors
                # Validate quantization against the actual rounded Norm output.
                expected_q,expected_s,_=reference(actual[1][:128], 'fp8e4nv')
                np.testing.assert_array_equal(actual[3][:128],expected_q)
                np.testing.assert_array_equal(actual[4].view(np.uint8)[:expected_s.size],expected_s)
                for _ in range(3):run()
                b.time_call(run,b.main_stream,repeats=100)
                times=[b.time_call(run,b.main_stream,repeats=10) for _ in range(5)]
                row=dict(residual=residual,rows=rows,warps=warps,median_ms=float(np.median(times)),samples_ms=times,
                         relative_l2=errors,bitwise_equal=all(np.array_equal(v,ref) for v,ref in zip(actual,baseline)))
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2));b.release_since(mark)
        if a.backward_only:
            b.mxfp8_training=True;b.mxfp8_norm_backward_fusion=True
            b.mxfp8_norm_fusion=True
            parts=[e.array(rng.standard_normal(shape,dtype=np.float32),'bf16').data for _ in range(2)]
            _,saved=e.ops.norm(x.data,w.data,out_dtype='bf16');baseline=None
            for groups,warps in ((1024,4),(2048,4),(1024,2),(2048,2)):
                mark=b.mark();b.mxfp8_norm_backward_groups=groups
                b.kernel_options['model.dense_cublas.mxfp8.mxfp8_norm_backward']={'num_warps':warps}
                b._commands=[]
                try:dx,dw=e.ops.norm_backward_sum(saved,parts,x.data)
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                run();actual=[b.to_numpy(v) for v in (dx,dw)]
                if baseline is None:baseline=actual
                errors=[float(np.linalg.norm(v.astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-20)) for v,ref in zip(actual,baseline)]
                assert max(errors)<1e-5,errors
                b.time_call(run,b.main_stream,repeats=100)
                times=[b.time_call(run,b.main_stream,repeats=10) for _ in range(5)]
                row=dict(backward=True,groups=groups,warps=warps,median_ms=float(np.median(times)),samples_ms=times,
                         relative_l2=errors,dx_equal=bool(np.array_equal(actual[0],baseline[0])))
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2));b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
