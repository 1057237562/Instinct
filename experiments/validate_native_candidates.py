"""Compare native kernel candidates against the existing full-model loss/VJP."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/results/cublas/candidate_parity.json')
    parser.add_argument('--exact-only',action='store_true')
    a=parser.parse_args()
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    cfg=dict(hidden_size=192,num_hidden_layers=2,intermediate_size=512,
             num_attention_heads=4,num_key_value_heads=1,head_dim=96,
             vocab_size=256,max_position_embeddings=4096,use_grad_checkpoint=4)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',cases=[])
    variants=[('baseline',{}),('pack',dict(pack_tile=32)),('compact',dict(compact_gradients=True)),
              ('tiles',dict(pack_tile=32,attention_forward_tile=128,attention_forward_cols=64)),
              ('candidate',dict(pack_tile=32,attention_forward_tile=128,attention_forward_cols=64,compact_gradients=True)),
              ('fast_accum',dict(pack_tile=32,attention_forward_tile=128,attention_forward_cols=64,compact_gradients=True)),
              ('native_sdpa',dict(pack_tile=32,compact_gradients=True))]
    exact={'baseline','pack','compact','norm_cast','swiglu_amax','quant_pack','fusions'}
    variants.extend([('norm_cast',dict(fused_norm_cast=True)),
                     ('quant_pack',dict(fused_quant_pack=True)),
                     ('swiglu_amax',dict(compact_gradients=True,fused_swiglu_amax=True)),
                     ('fusions',dict(pack_tile=32,compact_gradients=True,fused_swiglu_amax=True,fused_norm_cast=True,fused_quant_pack=True))])
    if a.exact_only:variants=[v for v in variants if v[0] in exact]
    bf16_reference={}
    for fp8 in (False,True):
        for seq in (137,1024):
            reference=None
            for name,options in variants:
                with TrainingBackend(allocator_backend='async',config=ScheduleConfig(fp8_fast_accum_output=name=='fast_accum')) as b:
                    for key,value in options.items():setattr(b,key,value)
                    if name=='native_sdpa':
                        from experiments.native_sdpa.adapter import NativeAttention
                        b.native_attention=NativeAttention(b)
                    if options.get('attention_forward_tile'):
                        b.kernel_options['model.dense_triton.attention.forward']=dict(num_warps=8,num_stages=2)
                    e=Engine(b,fp8=fp8)
                    m=Model(e,cfg,param_dtype='bf16',muon_mantissa=True)
                    ids=np.arange(seq*2,dtype=np.int64).reshape(2,seq)%253+3
                    segments=np.tile(np.arange(seq,dtype=np.int64)*3//seq,(2,1));segments[:,-7:]=-1
                    out=m(ids,ids,sequence_ids=segments)
                    loss=e.item(out.loss)
                    grads={p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}
                    assert np.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())
                    if reference is None:reference=(loss,grads)
                    if not fp8 and name=='baseline':bf16_reference[seq]=grads
                    diffs={key:float(np.linalg.norm(value-reference[1][key])/max(np.linalg.norm(reference[1][key]),1e-20))
                           for key,value in grads.items()}
                    row=dict(name=name,fp8=fp8,seq=seq,loss=loss,loss_difference=loss-reference[0],
                             worst_relative_l2=max(diffs.values()),gradient_relative_l2=diffs,
                             bitwise_gradients=all(np.array_equal(v,reference[1][k]) for k,v in grads.items()))
                    row['gradient_l2_to_bf16']={key:float(np.linalg.norm(value-bf16_reference[seq][key])/max(np.linalg.norm(bf16_reference[seq][key]),1e-20)) for key,value in grads.items()}
                    row['within_original_comparison_bound']=row['worst_relative_l2']<.035
                    report['cases'].append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('gradient_relative_l2','gradient_l2_to_bf16')}),flush=True)
                    target.write_text(json.dumps(report,indent=2))
                    if name in exact:
                        assert loss==reference[0]
                        # Embedding scatter uses atomics: scheduling may change
                        # its FP32 reduction order even with identical operands.
                        for key,value in grads.items():
                            if key=='model.embed_tokens.weight':
                                np.testing.assert_allclose(value,reference[1][key],atol=1e-7,rtol=1e-6)
                            else:np.testing.assert_array_equal(value,reference[1][key],err_msg=key)
                    # Attention algorithms have different reduction orders. Keep
                    # the original bound/result visible and also compare both
                    # FP8 paths to the common BF16 oracle; do not hide rejections.
                    elif not fp8:assert row['within_original_comparison_bound'],row
    report.update(status='complete',torch_imported='torch' in sys.modules,
                  original_comparison_passed=all(x['within_original_comparison_bound'] for x in report['cases']))
    target.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
