"""Synthetic gradient accumulation parity; no optimizer updates or training data."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas.policy import enable_fused_ops
    report=dict(status='running',cases=[])
    for fp8 in (False,True):
        for seq in (137,512):
            reference=None
            for enabled in (False,True):
                with TrainingBackend(allocator_backend='async') as b:
                    enable_fused_ops(b);b.release_checkpoint_cache=enabled;b.fused_boundary_norm=True;b.fused_activation_layouts=True
                    e=Engine(b,fp8=fp8);cfg=dict(hidden_size=192,num_hidden_layers=2,intermediate_size=512,num_attention_heads=4,
                        num_key_value_heads=1,head_dim=96,vocab_size=256,max_position_embeddings=1024,use_grad_checkpoint=4,selective_cache_mib=8)
                    m=Model(e,cfg,param_dtype='bf16',muon_mantissa=True)
                    opt=Optimizer(e,m.parameters(),'muon',muon_mantissa=True,muon_momentum_dtype='bf16')
                    losses=[]
                    for micro in range(2):
                        ids=(np.arange(seq*2,dtype=np.int64).reshape(2,seq)+micro)%253+3
                        segments=np.tile(np.arange(seq)*3//seq,(2,1));segments[:,-7:]=-1
                        mark=b.mark();out=m(ids,ids,sequence_ids=segments);losses.append(e.item(out.loss));opt.backward(out.loss)
                        values={p.name:e.numpy(g) for p,g in opt.gradients.items()}
                        after_backward=b.memory()['live_bytes'];b.release_since(mark)
                    if not enabled:reference=(losses,values,after_backward)
                    else:
                        assert losses==reference[0]
                        for name,value in values.items():
                            if name=='model.embed_tokens.weight':
                                # Same atomic-embedding bound as validate_native_replay.
                                np.testing.assert_allclose(value,reference[1][name],atol=1e-7,rtol=1e-6,err_msg=name)
                            else:np.testing.assert_array_equal(value,reference[1][name],err_msg=name)
                        row=dict(fp8=fp8,sequence=seq,accumulated=opt.accumulated,
                            hidden_gradients_bitwise_equal=True,
                            worst_relative_l2=max(float(np.linalg.norm(v-reference[1][n])/max(np.linalg.norm(reference[1][n]),1e-20)) for n,v in values.items()),
                            baseline_live_after_backward=reference[2],candidate_live_after_backward=after_backward)
                        report['cases'].append(row);print(json.dumps(row),flush=True)
                        a.output.write_text(json.dumps(report,indent=2))
    report.update(status='complete',torch_imported='torch' in sys.modules);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
