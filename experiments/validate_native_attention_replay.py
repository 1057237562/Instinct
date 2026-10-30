"""Torch-free packed cuDNN isolation, multi-forward replay and ckpt parity."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--diagnose-repeat',action='store_true')
    p.add_argument('--candidate-quantum',type=int,default=128)
    p.add_argument('--plan-limit',type=int,default=64)
    p.add_argument('--shared-handles',action='store_true')
    p.add_argument('--async-metadata',action='store_true')
    p.add_argument('--docs-per-plan',type=int,default=0)
    p.add_argument('--documents',type=int,default=3)
    p.add_argument('--local-maxima',action='store_true')
    p.add_argument('--group-penalty',type=int,default=0)
    p.add_argument('--tune',action='store_true')
    p.add_argument('--cost-model',action='store_true')
    p.add_argument('--forward-cost-model',action='store_true')
    p.add_argument('--skewed-documents',action='store_true',help='One long document followed by short documents, to exercise local plan maxima')
    p.add_argument('--relative-l2-limit',type=float,default=1e-3,help='Explicit accepted whole-gradient tolerance; recorded with the result')
    a=p.parse_args()
    if not np.isfinite(a.relative_l2_limit) or a.relative_l2_limit<=0:p.error('relative L2 limit must be positive and finite')
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    cfg=dict(hidden_size=192,num_hidden_layers=2,intermediate_size=512,num_attention_heads=4,
        num_key_value_heads=1,head_dim=96,vocab_size=256,max_position_embeddings=4096)
    result=dict(status='running',cases=[],relative_l2_limit=a.relative_l2_limit,
                candidate_policy=dict(quantum=a.candidate_quantum,plan_limit=a.plan_limit,
                    shared_handles=a.shared_handles,
                    docs_per_plan=a.docs_per_plan,group_penalty=a.group_penalty,
                    local_maxima=a.local_maxima,tune=a.tune,
                    cost_model='forward' if a.forward_cost_model else a.cost_model))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def write():a.output.write_text(json.dumps(result,indent=2))
    def run(mode,seq,fp8,interleaved=False):
        with TrainingBackend(allocator_backend='async') as b:
            enable_fused_ops(b);b.attention_shared_handle=mode==4 and a.shared_handles
            b.native_attention=NativeAttention(b)
            # Keep the control independent of the currently shipped defaults.
            b.attention_plan_quantum=512;b.attention_plan_limit=64
            b.attention_local_maxima=False
            b.attention_group_penalty=0
            b.fused_scale_inverse=False
            if mode==4:
                b.attention_plan_quantum=a.candidate_quantum
                b.attention_plan_limit=a.plan_limit
                b.async_attention_metadata=a.async_metadata
                b.attention_docs_per_plan=a.docs_per_plan
                b.attention_local_maxima=a.local_maxima
                b.attention_group_penalty=a.group_penalty
                b.attention_tune=a.tune
                b.attention_cost_model='forward' if a.forward_cost_model else a.cost_model
                if b.attention_cost_model:b.native_attention.warm_catalog(seq,seq,4,1,96)
            e=Engine(b,fp8=fp8);m=Model(e,cfg|dict(use_grad_checkpoint=mode),param_dtype='bf16',muon_mantissa=True)
            ids=np.arange(seq,dtype=np.int64)[None]%253+3
            segments=np.minimum(np.arange(seq)*a.documents//seq,a.documents-1)[None];segments[:,-7:]=-1
            if a.skewed_documents:
                split=seq//2
                segments[:,:split]=0
                segments[:,split:]=1+np.arange(seq-split)*max(1,a.documents-1)//(seq-split)
                segments[:,-7:]=-1
            out=m(ids,ids,sequence_ids=segments)
            loss=e.item(out.loss);logits=e.numpy(out.logits)
            if interleaved:
                other=np.zeros_like(segments);other[:,seq//2:]=8
                m(ids,ids,sequence_ids=other)
            grads={param.name:e.numpy(grad) for param,grad in e.vjp(out.loss).items()}
            assert b.native_attention.forward_calls and b.native_attention.backward_calls
            assert np.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())
            return loss,grads,logits
    for seq,fp8 in ((137,False),(137,True),(4096,False),(4096,True)):
        ref=run(2,seq,fp8)
        control=run(2,seq,fp8) if a.diagnose_repeat else ref
        control_l2=max(float(np.linalg.norm(control[1][k]-v)/max(np.linalg.norm(v),1e-20)) for k,v in ref[1].items())
        control_abs=max(float(np.max(np.abs(control[1][k]-v))) for k,v in ref[1].items())
        for interleaved in (False,True):
            actual=run(4,seq,fp8,interleaved)
            np.testing.assert_allclose(actual[0],ref[0],rtol=1e-6,atol=1e-6)
            worst=0.;relative=0.;original_bound=True
            for name,expected in ref[1].items():
                within=bool(np.allclose(actual[1][name],expected,rtol=2e-3,atol=2e-5))
                original_bound=original_bound and within
                # cuDNN 9.27 explicitly disallows deterministic ragged backward
                # on SM12X. Keep the original comparison in the report, and
                # bound both local BF16 rounding and whole-gradient deviation.
                if not a.diagnose_repeat:np.testing.assert_allclose(actual[1][name],expected,rtol=1/128,atol=2e-5,err_msg=name)
                worst=max(worst,float(np.max(np.abs(actual[1][name]-expected))))
                relative=max(relative,float(np.linalg.norm(actual[1][name]-expected)/max(np.linalg.norm(expected),1e-20)))
            if not a.diagnose_repeat:assert relative<a.relative_l2_limit,(seq,fp8,relative)
            row=dict(seq=seq,fp8=fp8,interleaved=interleaved,loss=actual[0],max_gradient_difference=worst,
                relative_l2=relative,repeat_control_l2=control_l2,repeat_control_max_abs=control_abs,original_bound=original_bound)
            result['cases'].append(row);print(json.dumps(row),flush=True);write()
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b);b.attention_shared_handle=a.shared_handles;b.native_attention=NativeAttention(b)
        e=Engine(b);m=Model(e,cfg|dict(use_grad_checkpoint=4),param_dtype='bf16',muon_mantissa=True)
        ids=np.arange(137,dtype=np.int64)[None]%253+3
        segments=np.asarray([[3]*19+[8]*22+[3]*81+[-1]*15],np.int64)
        reference=e.numpy(m(ids,sequence_ids=segments).logits)
        changed=ids.copy();changed[:,19:41]=(changed[:,19:41]+97)%253+3
        actual=e.numpy(m(changed,sequence_ids=segments).logits)
        np.testing.assert_array_equal(actual[:,:19],reference[:,:19])
        np.testing.assert_array_equal(actual[:,41:122],reference[:,41:122])
        future=ids.copy();future[:,10:19]=(future[:,10:19]+41)%253+3
        actual=e.numpy(m(future,sequence_ids=segments).logits)
        np.testing.assert_array_equal(actual[:,:10],reference[:,:10])
    result.update(status='diagnostic' if a.diagnose_repeat else 'complete',document_and_causal_isolation=True,
        backward_deterministic=False,relative_l2_limit=a.relative_l2_limit,torch_imported='torch' in sys.modules);write()


if __name__=='__main__':main()
