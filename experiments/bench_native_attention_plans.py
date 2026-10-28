"""CPU geometry shortlist then bounded SDPA timing on production metadata."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def lengths_of(ids):
    result=[]
    for row in ids:
        edges=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        result.extend(int(e-s) for s,e in zip(edges[:-1],edges[1:]) if row[s]>=0)
    return tuple(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--neighbors',action='store_true')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.data import Tokenizer
    from model.dense_cublas.parquet_data import ParquetPretrain,PackedChunk
    from model.dense_cublas.vendor_attention import NativeAttention,document_groups
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.policy import enable_fused_ops
    tok=Tokenizer('D:/AI/Instinct/model')
    data=ParquetPretrain('D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet',tok,
        'cache/native_tokens',lengths=[3072,4096],token_budget=12288,threads=4)
    batches=[]
    for group in (0,1):
        manifest=data.build(group);chunk=PackedChunk(data.cache_path(group),manifest,tok.pad)
        try:
            order=chunk.order(data.token_budget,data.seed,0,group)
            for bucket,indices in (order[-64:] if group==0 else order[:224]):
                batch,_,_=chunk.batch(bucket,indices);batches.append(batch[2])
        finally:chunk.close()
    geometries=[(tuple(ids.shape),lengths_of(ids)) for ids in batches]
    def score(q,size,penalty):
        padded=calls=0;plans=set();max_pad=0
        for shape,lengths in geometries:
            for start,n in document_groups(lengths,size,q,shape[1],penalty):
                maximum=min(shape[1],((max(lengths[start:start+n])+q-1)//q)*q)
                padded+=n*maximum;calls+=1;max_pad=max(max_pad,n*maximum)
                plans.add((n,maximum,shape[0]*shape[1]))
        # dQ/stats bytes plus a launch-cost prior. No GPU parameter sweep.
        return dict(attention_plan_quantum=q,attention_docs_per_plan=size,attention_group_penalty=penalty,
            padded_tokens=padded,calls=calls,plans=2*len(plans),max_workspace_estimate=max_pad*16*96*4,
            predicted_cost=padded+calls*4096)
    baseline=score(1024,8,4096)
    ranked=sorted([score(q,s,p) for q,s,p in itertools.product((256,512,1024),(4,8,16),(2048,4096,8192))],
                  key=lambda c:c['predicted_cost'])
    shortlisted=[baseline]
    for row in ranked:
        if row==baseline or row['plans']>128 or row['max_workspace_estimate']>512*2**20:continue
        # Retain different length granularities instead of near-identical policies.
        if any(row['attention_plan_quantum']==c['attention_plan_quantum'] for c in shortlisted):continue
        shortlisted.append(row)
        if len(shortlisted)==3:break
    # Two metadata quantiles per bucket cover typical and fragmented samples.
    indices=[]
    for seq in (3072,4096):
        selected=sorted((i for i,(s,l) in enumerate(geometries) if s[1]==seq),key=lambda i:len(geometries[i][1]))
        indices.extend([selected[len(selected)//2],selected[9*len(selected)//10]])
    report=dict(status='running',cpu_ranking=ranked,shortlist=shortlisted,sample_indices=indices,cases=[])
    trials=[(False,baseline)]+[(2 if a.neighbors else True,c) for c in shortlisted]
    if a.neighbors:trials=[(2,baseline),(2,score(512,4,4096))]
    for tune,config in trials:
        with TrainingBackend(allocator_backend='async') as b:
            enable_fused_ops(b);ops=CublasDenseOps(b);b.attention_tune=tune;b.attention_plan_limit=128
            for k in ('attention_plan_quantum','attention_docs_per_plan','attention_group_penalty'):setattr(b,k,config[k])
            native=NativeAttention(b);b.native_attention=native;rng=np.random.default_rng(372)
            arrays=[]
            for h in (16,4,4,16):
                mark=b.mark();x=ops.cast(b.from_numpy(rng.standard_normal((12288,h,96),dtype=np.float32)*.2),'bf16')
                b.release_since(mark,keep=(x,));arrays.append(x)
            for index in indices:
                mark=b.mark();ids=batches[index];shape=ids.shape
                q,k,v,dy=[x.view(*shape,h,96) for x,h in zip(arrays,(16,4,4,16))]
                native.prepare(shape,ids,16,4,96)
                mask=b.empty((1,),'i32');mask.native_attention_context=native.context
                out,saved=native.attention(q,k,v,mask,0.,0,0,None,True,None)
                native.backward(saved,dy,None)
                # Time normal calls: scratch addresses are recycled between
                # calls, so replaying captured Python closures would be invalid.
                def run_live():
                    local=b.mark();out,saved=native.attention(q,k,v,mask,0.,0,0,None,True,None)
                    native.backward(saved,dy,None);b.release_since(local)
                for _ in range(2):run_live()
                times=[b.time_call(run_live,b.main_stream,repeats=3) for _ in range(3)]
                row=dict(tune=tune,config=config,sample_index=index,shape=list(shape),documents=len(lengths_of(ids)),
                         median_ms=float(np.median(times)))
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2));b.release_since(mark)
            report.setdefault('plan_tuning',[]).append(dict(config=config,tune=tune,metrics=native.metrics))
    report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
