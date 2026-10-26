"""Check terminal Dense replay against the full replay, without weight updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fused-ffn',action='store_true')
    p.add_argument('--schedule',action='store_true')
    p.add_argument('--cudnn',action='store_true')
    p.add_argument('--aligned',action='store_true')
    p.add_argument('--aligned-pointwise',action='store_true')
    p.add_argument('--async-metadata',action='store_true')
    p.add_argument('--residual-norm',action='store_true')
    p.add_argument('--plan-quantum',type=int,default=128)
    p.add_argument('--docs-per-plan',type=int,default=0)
    p.add_argument('--norm-rope-rows',type=int,default=1)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.policy import enable_fused_ops
    result=dict(status='running',cases=[])
    for fp8 in (False,True):
        for mode in (3,4):
            for seq in (137,512):
                for cache in (0,8):
                    reference=None
                    for enabled in (False,True):
                        with TrainingBackend(allocator_backend='async') as b:
                            enable_fused_ops(b);b.terminal_replay=enabled
                            b.fused_ffn=a.fused_ffn and enabled
                            b.norm_groups=1024 if a.schedule and enabled else 128
                            b.elementwise_block=1024 if a.schedule and enabled else 256
                            b.aligned_pointers=(a.aligned or a.aligned_pointwise) and enabled
                            b.aligned_pointwise_only=a.aligned_pointwise
                            b.async_attention_metadata=a.async_metadata and enabled
                            b.fused_residual_norm=a.residual_norm and enabled
                            b.attention_plan_quantum=a.plan_quantum if enabled else 128
                            b.attention_docs_per_plan=a.docs_per_plan if enabled else 0
                            b.norm_rope_rows=a.norm_rope_rows if enabled else 1
                            if a.cudnn:
                                from model.dense_cublas.vendor_attention import NativeAttention
                                b.native_attention=NativeAttention(b)
                            b.selective_attention_min_seq=1 if cache else 4096
                            e=Engine(b,fp8=fp8,fp8_filter='eligible')
                            cfg=dict(hidden_size=192,num_hidden_layers=2,intermediate_size=512,
                                num_attention_heads=4,num_key_value_heads=1,head_dim=96,
                                vocab_size=256,max_position_embeddings=1024,use_grad_checkpoint=mode,
                                selective_cache_mib=cache)
                            m=Model(e,cfg,param_dtype='bf16',muon_mantissa=True)
                            ids=np.arange(seq*2,dtype=np.int64).reshape(2,seq)%253+3
                            segments=np.tile(np.arange(seq)*3//seq,(2,1));segments[:,-7:]=-1
                            out=m(ids,ids,sequence_ids=segments)
                            loss=e.item(out.loss)
                            grads={x.name:e.numpy(g) for x,g in e.vjp(out.loss).items()}
                            if not enabled:reference=loss,grads;continue
                            if a.aligned:np.testing.assert_allclose(loss,reference[0],rtol=1e-6,atol=1e-6)
                            else:assert loss==reference[0],(fp8,mode,seq,cache,loss,reference[0])
                            assert grads.keys()==reference[1].keys()
                            for name,g in grads.items():
                                if a.cudnn:
                                    np.testing.assert_allclose(g,reference[1][name],atol=2e-5,rtol=1/128)
                                    relative=np.linalg.norm(g-reference[1][name])/max(np.linalg.norm(reference[1][name]),1e-20)
                                    assert relative<.001,(name,relative)
                                elif a.schedule and 'norm.weight' in name:
                                    np.testing.assert_allclose(g,reference[1][name],atol=1e-7,rtol=1e-5)
                                elif name=='model.embed_tokens.weight':np.testing.assert_allclose(g,reference[1][name],atol=1e-7,rtol=1e-6)
                                else:np.testing.assert_array_equal(g,reference[1][name],err_msg=name)
                            row=dict(fp8=fp8,mode=mode,seq=seq,cache_mib=cache,loss=loss,loss_difference=loss-reference[0],
                                worst_relative_l2=max(float(np.linalg.norm(g-reference[1][n])/max(np.linalg.norm(reference[1][n]),1e-20)) for n,g in grads.items()),passed=True)
                            result['cases'].append(row);print(json.dumps(row),flush=True)
                            a.output.write_text(json.dumps(result,indent=2))
    result.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
