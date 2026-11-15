"""Fixed-weight CUDA cache-budget resume audit; never updates training weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    from experiments.smoke_dense_triton import RejectTorch
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas import checkpoint,policy,training_profile
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas.selective import recipe_plan
    cfg=dict(hidden_size=1536,intermediate_size=5504,num_hidden_layers=2,vocab_size=6400,
             num_attention_heads=16,num_key_value_heads=4,head_dim=96,max_position_embeddings=4096,
             use_grad_checkpoint=4,dropout=0.,selective_cache_mib=6)
    report=dict(scope='Two-layer fixed-weight CUDA audit; no parameter updates or production cursor changes',cases=[])
    state=args.output.with_suffix('.resume.npz')
    with TrainingBackend(allocator_backend='async',async_release_mib=2048) as backend:
        policy.enable_fused_ops(backend)
        training_profile.apply(backend,'v2_mxfp8',training_profile.settings('v2_mxfp8'))
        e=Engine(backend,fp8=True,fp8_filter='auto',seed=47)
        model=Model(e,cfg,param_dtype='bf16',activation_dtype='bf16',muon_mantissa=True)
        opt=Optimizer(e,model.parameters(),'muon',muon_mantissa=True,muon_momentum_dtype='bf16')
        rng=np.random.default_rng(21)
        progress=dict(steps=136,processed_tokens=1665280,parquet_cursor=dict(group=6,batch=91))
        checkpoint.save(state,model,opt,progress,rng)
        digest=hashlib.sha256(state.read_bytes()).hexdigest()
        before=checkpoint.snapshot(model,opt,progress,rng)
        model.config['selective_cache_mib']=64
        restored=checkpoint.load(state,model,opt,rng)
        after=checkpoint.snapshot(model,opt,restored,rng)
        for key,value in before.items():
            if key!='__metadata__':np.testing.assert_array_equal(after[key],value,err_msg=key)
        a=json.loads(str(before['__metadata__']));b=json.loads(str(after['__metadata__']))
        assert b.pop('cache_precision_reference')=={'selective_cache_mib':6}
        a.pop('config');b.pop('config');assert a==b
        del before,after
        report['resume_state_exact']=True
        seed_counter=e.seed,e.counter
        from model.dense_cublas import mxfp8
        original_swiglu=mxfp8.quantize_swiglu
        counts={}
        def counted_swiglu(ops,gate,up,*values,**keywords):
            if keywords.get('save_inputs'):counts['saved']=counts.get('saved',0)+1
            if gate.dtype=='fp8e4nv':counts['reused']=counts.get('reused',0)+1
            return original_swiglu(ops,gate,up,*values,**keywords)
        mxfp8.quantize_swiglu=counted_swiglu
        for seq in (128,256):
            ids=np.random.default_rng(seq).integers(3,6400,size=(2,seq),dtype=np.int64)
            results=[]
            for budget in (6,64):
                counts.clear()
                model.config['selective_cache_mib']=budget;e.seed,e.counter=seed_counter
                mark=backend.mark();out=model(ids,ids,return_logits=False)
                loss=e.item(out.loss)
                gradients={p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}
                assert np.isfinite(loss) and all(np.isfinite(g).all() for g in gradients.values())
                backend.synchronize();backend.release_since(mark)
                plans=[recipe_plan(model.config,(2,seq,1536),'bf16',i,
                                   attention_min_seq=1,auxiliary_kv=True,compressed_ffn=True,
                                   compressed_reference=model.cache_precision_reference) for i in range(2)]
                compressed=sum(p.compress_ffn and any('gate_proj' in n for n in p.linear_names) for p in plans)
                assert counts.get('saved',0)==compressed and counts.get('reused',0)==compressed,counts
                results.append((loss,gradients,sum(any('gate_proj' in n for n in p.linear_names) for p in plans),dict(counts)))
            expected,actual=results
            error=sum(np.square(actual[1][k].astype(np.float64)-g).sum() for k,g in expected[1].items())
            norm=sum(np.square(g.astype(np.float64)).sum() for g in expected[1].values())
            relative=float(np.sqrt(error/norm))
            assert abs(actual[0]-expected[0])<1e-5
            assert relative<1e-6,relative
            assert expected[2]<actual[2] and actual[2]==2
            report['cases'].append(dict(shape=list(ids.shape),loss=actual[0],loss_difference=actual[0]-expected[0],
                gradient_relative_l2=relative,saved_gate_layers_before=expected[2],saved_gate_layers_after=actual[2],
                measured_compressed_cache_calls=actual[3]))
        mxfp8.quantize_swiglu=original_swiglu
        assert opt.accumulated==0 and opt.updates==0 and not opt.present
        assert hashlib.sha256(state.read_bytes()).hexdigest()==digest
        report.update(status='complete',model_config=model.config,parameters=sum(p.size for p in model.parameters()),
                      no_optimizer_updates=True,checkpoint_file_unchanged=True,torch_imported='torch' in sys.modules)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
