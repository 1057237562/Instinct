"""Independent NumPy reference for MX scales, FP8 bytes and cuBLAS GEMMs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def fp8_table(dtype):
    bits=np.arange(256,dtype=np.uint8);e5=dtype=='fp8e5';mant_bits=2 if e5 else 3;bias=15 if e5 else 7
    exponent=(bits.astype(np.int32)&127)>>mant_bits;mantissa=bits&((1<<mant_bits)-1)
    result=np.where(exponent==0,np.exp2(1-bias)*mantissa/(1<<mant_bits),
                    np.exp2(exponent-bias)*(1+mantissa/(1<<mant_bits)))
    result=np.where((exponent==31) if e5 else ((exponent==15)&(mantissa==7)),np.nan,result)
    return (result*np.where(bits&128,-1.,1.)).astype(np.float32)


def reference(x,dtype):
    rows,cols=x.shape;pr=(rows+127)//128*128;pc=(cols+127)//128*128
    padded=np.zeros((pr,pc),np.float32);padded[:rows,:cols]=x
    blocks=padded.reshape(pr,pc//32,32);maximum=np.max(np.abs(blocks),axis=2)
    maximum_value=448. if dtype=='fp8e4nv' else 57344.
    raw=(maximum/np.float32(maximum_value)).view(np.uint32)
    exponent=(raw>>23)&255;mantissa=raw&0x7fffff
    exponent+=(((exponent>0)&(exponent<254)&(mantissa!=0))|((exponent==0)&(mantissa>0x400000))).astype(np.uint32)
    inverse=np.exp2(np.float32(127)-exponent.astype(np.float32))
    values=blocks*inverse[:,:,None]
    table=fp8_table(dtype);limit=124 if dtype=='fp8e5' else 127
    positive=table[:limit];v=np.minimum(np.abs(values),maximum_value)
    hi=np.minimum(np.searchsorted(positive,v),limit-1);lo=np.maximum(hi-1,0)
    dlo=v-positive[lo];dhi=positive[hi]-v
    selected=np.where((dhi<dlo)|((dhi==dlo)&((hi&1)==0)),hi,lo).astype(np.uint8)
    selected|=np.signbit(values).astype(np.uint8)*128
    dequant=table[selected]*np.exp2(exponent.astype(np.float32)-np.float32(127))[:,:,None]
    r=np.arange(pr)[:,None];c=np.arange(pc//32)[None,:]
    offsets=(r//128)*(pc//32)*128+(c//4)*512+(r%32)*16+((r%128)//32)*4+c%4
    scales=np.empty(pr*pc//32,np.uint8);scales[offsets]=exponent.astype(np.uint8)
    return selected.reshape(pr,pc)[:rows,:cols],scales,dequant.reshape(pr,pc)[:rows,:cols]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--compressed-cache',action='store_true')
    a=parser.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.mxfp8 import quantize
    report=dict(status='running',quantization=[],gemm=[])
    with TrainingBackend(allocator_backend='async') as b:
        ops=CublasDenseOps(b);rng=np.random.default_rng(149)
        for shape in ((33,160),(128,128),(256,384)):
            for dtype in ('fp8e4nv','fp8e5'):
                mark=b.mark();x=rng.standard_normal(shape,dtype=np.float32)
                x*=np.exp2((np.arange(shape[0])%19-9).astype(np.float32))[:,None]
                x[0]=0
                x[1,:6]=np.asarray([448.,-448.,57344.,-57344.,2**-120,2**120],np.float32)
                device=ops.cast(b.from_numpy(x),'bf16');original=b.to_numpy(device)
                q,s=quantize(ops,device,dtype,dual_layout=True);expected,scales,_=reference(original,dtype)
                np.testing.assert_array_equal(b.to_numpy(q),expected)
                np.testing.assert_array_equal(b.to_numpy(s).view(np.uint8),scales)
                qt=q.transpose_data;expected_t,scales_t,_=reference(original.T,dtype)
                np.testing.assert_array_equal(b.to_numpy(qt),expected_t)
                np.testing.assert_array_equal(b.to_numpy(qt.mx32_scale).view(np.uint8),scales_t)
                row=dict(shape=list(shape),dtype=dtype,fp8_bytes_equal=True,scale_bytes_equal=True,transpose_equal=True)
                report['quantization'].append(row);print(json.dumps(row),flush=True);b.release_since(mark)
        from model.dense_cublas.ffn import cached_statistics
        from model.dense_cublas.mxfp8 import quantize_swiglu
        import triton
        report['swiglu']=[]
        for shape in ((128,256),(137,192)):
            for backward in (False,True):
                mark=b.mark();inputs=[ops.cast(b.from_numpy(rng.standard_normal(shape,dtype=np.float32)),'bf16') for _ in range(3)]
                fused_inputs=inputs
                if a.compressed_cache:
                    packed=[quantize(ops,x,'fp8e4nv')[0] for x in inputs[:2]]
                    decoded=[reference(b.to_numpy(x),'fp8e4nv')[2] for x in inputs[:2]]
                    inputs=[ops.cast(b.from_numpy(x),'bf16') for x in decoded]+inputs[2:]
                    fused_inputs=packed+inputs[2:]
                count=triton.cdiv(inputs[0].numel,1024)
                partials=[b.empty((count,),'fp32') for _ in range(3)];values=[b.empty(shape,'bf16') for _ in range(3)]
                b.launch(cached_statistics,(count,),(*inputs,*partials,*values),N=inputs[0].numel,BACKWARD=backward)
                y,sg,up=[b.to_numpy(v) for v in values] if backward else (b.to_numpy(values[0]),None,None)
                actual=quantize_swiglu(ops,*fused_inputs[:2],fused_inputs[2] if backward else None)
                pairs=[(actual[0],y.T if backward else y,'fp8e4nv')]
                if backward:pairs.extend([(actual[2],sg,'fp8e5'),(actual[2].transpose_data,sg.T,'fp8e5'),
                                          (actual[4],up,'fp8e5'),(actual[4].transpose_data,up.T,'fp8e5')])
                for tensor,value,dtype in pairs:
                    expected,scales,_=reference(value,dtype)
                    np.testing.assert_array_equal(b.to_numpy(tensor),expected)
                    np.testing.assert_array_equal(b.to_numpy(tensor.mx32_scale).view(np.uint8),scales)
                if not backward:
                    packed=quantize_swiglu(ops,*inputs[:2],save_inputs=True)
                    for tensor,original in zip((packed[2],packed[4]),inputs[:2]):
                        expected,scales,_=reference(b.to_numpy(original),'fp8e4nv')
                        np.testing.assert_array_equal(b.to_numpy(tensor),expected)
                        np.testing.assert_array_equal(b.to_numpy(tensor.mx32_scale).view(np.uint8),scales)
                    expected,scales,_=reference(y,'fp8e4nv')
                    np.testing.assert_array_equal(b.to_numpy(packed[0]),expected)
                    np.testing.assert_array_equal(b.to_numpy(packed[1]).view(np.uint8),scales)
                report['swiglu'].append(dict(shape=list(shape),backward=backward,compressed_cache=a.compressed_cache,bytes_and_scales_equal=True))
                b.release_since(mark)
        for dtype in ('fp8e4nv','fp8e5'):
            mark=b.mark();m,n,k=128,256,384;values=[];gpu=[]
            for shape,kind in (((m,k),'fp8e4nv'),((n,k),dtype)):
                x=ops.cast(b.from_numpy(rng.standard_normal(shape,dtype=np.float32)),'bf16')
                original=b.to_numpy(x);q,s=quantize(ops,x,kind);gpu.extend((q,s))
                values.append(reference(original,kind)[2])
            left,sa,right,sb=gpu;out=b.empty((n,m),'fp32');size=b.config.workspace_mib*2**20;work=b.workspaces[b.main_stream]
            plan,candidates=b.native.create_plan(m,n,k,'fp8e4nv',dtype,'fp32',size,8,0,mxfp8=True,block_scales=(sa.storage,sb.storage))
            b.native.run(plan,0,left.storage,right.storage,out.storage,sa.storage,sb.storage,work.storage,size,b.main_stream)
            expected=values[1]@values[0].T;actual=b.to_numpy(out)
            error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-5)
            row=dict(shape_mnk=[m,n,k],types=['fp8e4nv',dtype],relative_l2_to_dequantized_reference=error,candidates=len(candidates))
            report['gemm'].append(row);print(json.dumps(row),flush=True);b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
