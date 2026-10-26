"""Native cuDNN C++ SDPA correctness/performance probe; no weight updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='experiments/results/cublas/cpp_sdpa.json')
    p.add_argument('--packed',action='store_true')
    a=p.parse_args();sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from experiments.native_sdpa.bridge import Sdpa
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',cases=[])
    chunk=None
    if a.packed:
        from model.dense_cublas.data import Tokenizer
        from model.dense_cublas.parquet_data import ParquetPretrain,PackedChunk
        tok=Tokenizer('D:/AI/Instinct/model')
        data=ParquetPretrain('D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet',tok,
            'cache/native_tokens',lengths=[3072,4096],token_budget=12288,threads=4)
        manifest=data.build(0);chunk=PackedChunk(data.cache_path(0),manifest,tok.pad)
    with TrainingBackend(allocator_backend='async') as b:
        ops=CublasDenseOps(b);lib=Sdpa()
        try:
            for batch,seq in ((2,128),(4,3072),(3,4096)):
                mark=b.mark();rng=np.random.default_rng(13)
                q,k,v,dy=[ops.cast(b.from_numpy(rng.standard_normal((batch,seq,h,96),dtype=np.float32)),'bf16') for h in (16,4,4,16)]
                mask=None;metadata=[];docs=batch;maximum=seq
                if a.packed:
                    if seq==128:
                        ids=np.tile(np.repeat([5,8,5,-1],[19,22,76,11]),(batch,1))
                    else:ids=chunk.batch(0 if seq==3072 else 1,np.arange(batch))[0][2]
                    boundary=np.ones_like(ids,dtype=bool);boundary[:,1:]=ids[:,1:]!=ids[:,:-1]
                    canonical=np.maximum.accumulate(np.where(boundary,np.arange(seq),0),axis=1);canonical[ids<0]=-1
                    mask=b.from_numpy(canonical.astype(np.int64))
                    starts=[];lengths=[]
                    for row in range(batch):
                        positions=np.flatnonzero(boundary[row]);ends=np.r_[positions[1:],seq]
                        for start,end in zip(positions,ends):
                            if ids[row,start]>=0:starts.append(row*seq+int(start));lengths.append(int(end-start))
                    offsets=np.asarray(starts+[batch*seq],np.int64)
                    metadata=[b.from_numpy(offsets*width) for width in (16*96,4*96,16)]
                    metadata.append(b.from_numpy(np.asarray(lengths,np.int32)))
                    docs=len(lengths);maximum=max(lengths)
                ref,saved=ops.attention(q,k,v,mask,packed=a.packed);refs=[b.to_numpy(x) for x in (ref,*ops.attention_backward(saved,dy))]
                out=b.empty(q.shape,'bf16',zero=True);stats=b.empty((docs,16,maximum),'fp32',zero=True)
                dq,dk,dv=[b.empty(x.shape,x.dtype,zero=True) for x in (q,k,v)]
                timings=[]
                for backward in (False,True):
                    plan,n=lib.plan(docs,maximum,16,4,96,backward,b.main_stream,total=batch*seq if a.packed else 0)
                    work=b.empty((max(1,(n+3)//4),),'fp32')
                    buffers=[q,k,v,out,stats]+([dy,dq,dk,dv] if backward else [])
                    if a.packed:buffers=buffers+([None]*(9-len(buffers)))+metadata
                    call=lambda:lib.run(plan,buffers,work)
                    for _ in range(3):call()
                    ms=b.time_call(call,b.main_stream,repeats=20)
                    timings.append(dict(backward=backward,ms=ms,workspace_bytes=n))
                errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip((b.to_numpy(t) for t in (out,dq,dk,dv)),refs)]
                assert max(errors)<.01,errors
                row=dict(batch=batch,seq=seq,documents=docs,max_document_length=maximum,packed=a.packed,relative_l2=errors,timings=timings)
                report['cases'].append(row);print(json.dumps(row),flush=True)
                target.write_text(json.dumps(report,indent=2));b.release_since(mark)
        finally:lib.close()
    if chunk is not None:chunk.close()
    report.update(status='complete',torch_imported='torch' in sys.modules)
    target.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
