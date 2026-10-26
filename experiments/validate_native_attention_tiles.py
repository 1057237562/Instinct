"""Torch-free real-mask output/VJP/isolation checks and forward tile screening."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/results/cublas/real_attention_tiles.json')
    args=parser.parse_args()
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.data import Tokenizer
    from model.dense_cublas.parquet_data import ParquetPretrain,PackedChunk
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_triton import attention
    tokenizer=Tokenizer('D:/AI/Instinct/model')
    data=ParquetPretrain('D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet',tokenizer,
                        'cache/native_tokens',lengths=[3072,4096],token_budget=12288,threads=4)
    manifest=data.build(0)
    chunk=PackedChunk(data.cache_path(0),manifest,tokenizer.pad)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',cases=[])
    try:
        for bucket,batch in ((0,4),(1,3)):
            arrays,_,_=chunk.batch(bucket,np.arange(batch))
            ids=arrays[2];seq=ids.shape[1]
            boundary=np.ones_like(ids,dtype=bool);boundary[:,1:]=ids[:,1:]!=ids[:,:-1]
            canonical=np.maximum.accumulate(np.where(boundary,np.arange(seq),0),axis=1)
            canonical[ids<0]=-1
            rng=np.random.default_rng(131)
            with TrainingBackend(allocator_backend='async') as backend:
                ops=CublasDenseOps(backend)
                q,k,v,dy=[ops.cast(backend.from_numpy(rng.standard_normal((batch,seq,h,96),dtype=np.float32)),'bf16')
                          for h in (16,4,4,16)]
                mask=backend.from_numpy(canonical.astype(np.int64))
                reference=None
                for bm,bn,warps,stages in ((32,32,4,2),(128,64,8,2),(64,64,4,3),(128,32,8,3),(64,32,4,3)):
                    mark=backend.mark()
                    backend.attention_tile=32
                    backend.attention_forward_tile,backend.attention_forward_cols=bm,bn
                    backend.kernel_options['model.dense_triton.attention.forward']=dict(num_warps=warps,num_stages=stages)
                    try:
                        out,saved=ops.attention(q,k,v,mask,packed=True)
                        gradients=ops.attention_backward(saved,dy)
                        values=[backend.to_numpy(t) for t in (out,*gradients)]
                        if reference is None:reference=values
                        errors=[]
                        for actual,expected in zip(values,reference):
                            assert np.isfinite(actual).all()
                            errors.append(float(np.sqrt(np.square(actual-expected).sum(dtype=np.float64)/
                                                        max(np.square(expected).sum(dtype=np.float64),1e-30))))
                        assert max(errors)<.005, errors
                        _,_,_,_,_,_,options,seed,counter=saved
                        options=options|dict(BM=bm,BN=bn)
                        from model.dense_triton.runtime import Scalar
                        call=lambda:backend.launch(attention.forward,((seq+bm-1)//bm,batch*16),
                            (q,k,v,mask,out,saved[5],Scalar(seed,'i32'),Scalar(counter,'i64')),**options)
                        for _ in range(3):call()
                        timing=[backend.time_call(call,backend.main_stream,repeats=10) for _ in range(3)]
                        entry=next(value for key,value in reversed(backend._cache.items())
                                   if key[0]=='forward' and dict(key[2])['BM']==bm and dict(key[2])['BN']==bn
                                   and dict(key[3]).get('num_warps')==warps and dict(key[3]).get('num_stages')==stages)
                        query=backend._bind('cuFuncGetAttribute',[ct.POINTER(ct.c_int),ct.c_int,ct.c_void_p])
                        registers,local=ct.c_int(),ct.c_int()
                        query(ct.byref(registers),4,entry[0]);query(ct.byref(local),3,entry[0])
                        row=dict(batch=batch,seq=seq,bm=bm,bn=bn,warps=warps,stages=stages,
                                 relative_l2=errors,median_forward_ms=float(np.median(timing)),
                                 registers=registers.value,local_bytes=local.value,shared_bytes=entry[1].shared)
                    except Exception as error:
                        row=dict(batch=batch,seq=seq,bm=bm,bn=bn,warps=warps,stages=stages,error=repr(error))
                    report['cases'].append(row);print(json.dumps(row),flush=True)
                    target.write_text(json.dumps(report,indent=2))
                    backend.release_since(mark)
        report.update(status='complete',torch_imported='torch' in sys.modules)
        target.write_text(json.dumps(report,indent=2))
    finally:chunk.close()


if __name__=='__main__':main()
