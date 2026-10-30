"""Bounded norm/quantization candidates on the V2 activation geometry."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--forward',action='store_true')
    p.add_argument('--activation-layouts',action='store_true')
    p.add_argument('--backward-groups',action='store_true',help='Four RMSNorm backward partitions; Q/K Norm is unchanged')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.policy import enable_fused_ops
    report=dict(status='running',cases=[])
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b);e=Engine(b,fp8=True);rng=np.random.default_rng(713)
        shape=(12288,1536)
        x=e.array(rng.normal(size=shape).astype(np.float32));w=e.array(rng.normal(size=(1536,)).astype(np.float32))
        parts=[e.array(rng.normal(size=shape).astype(np.float32),'bf16').data for _ in range(3)]
        residual=e.array(rng.normal(size=shape).astype(np.float32))
        _,saved=e.ops.norm(x.data,w.data,out_dtype='bf16')
        kinds=('activation_layouts',) if a.activation_layouts else (('input_forward','residual_forward') if a.forward else ('norm','quant'))
        if a.backward_groups:kinds=('norm_two_parts','norm_three_parts')
        for kind in kinds:
            reference=None
            configs=[dict(aligned=False,warps=4,tile=32),dict(aligned=True,warps=4,tile=64),
                     dict(aligned=True,warps=8,tile=64)] if kind=='norm' else [dict(aligned=False,warps=4,tile=t) for t in (32,64,128)]
            if a.forward:configs=[dict(aligned=False,warps=4,tile=32,rows=r) for r in (0,1,4)]
            if a.activation_layouts:configs=[dict(aligned=False,warps=4,tile=64,dual=v) for v in (False,True)]
            if a.backward_groups:
                configs=[dict(aligned=False,warps=4,tile=64,groups=g) for g in (1024,512,2048,4096)]
            for config in configs:
                mark=b.mark();b.aligned_norm_backward=config['aligned'];b.quant_tile=config['tile']
                if a.backward_groups:b.norm_groups=config['groups']
                b.kernel_options['model.dense_cublas.fused.rms_backward_sum']={'num_warps':config['warps']}
                b.norm_forward_rows=max(1,config.get('rows',1));b.fused_input_norm_amax=config.get('rows',0)>0
                b._commands=[]
                try:
                    if a.backward_groups:
                        selected=parts[:2] if kind=='norm_two_parts' else parts
                        out=e.ops.norm_backward_sum(saved,selected,residual.data)
                    elif kind=='activation_layouts':
                        q,s=e.ops.quantize(parts[0],dual_layout=config['dual'])
                        qt=e.ops._pack(q,shape[1],shape[0],shape[1],shape[0],(1,shape[1]),True)
                        out=(q,s,qt)
                    elif kind=='input_forward':
                        y,_=e.ops.norm(x.data,w.data,out_dtype='bf16');q,s=e.ops.quantize(y);out=(y,q,s)
                    elif kind=='residual_forward':
                        z,y=e.residual_norm(x,e.wrap(parts[0]),w,out_dtype='bf16');out=(z.data,y.data)
                    else:out=e.ops.norm_backward_sum(saved,parts,residual.data) if kind=='norm' else e.ops.quantize(parts[0],'fp8e5')
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                run();values=[b.to_numpy(x) for x in out]
                if kind=='quant':values.append(b.to_numpy(out[0].transpose_data))
                if reference is None:reference=values
                errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(values,reference)]
                valid=max(errors)<1e-6
                if a.backward_groups:
                    np.testing.assert_array_equal(values[0],reference[0])
                    assert valid,errors
                if kind in ('quant','activation_layouts'):
                    for x,y in zip(values,reference):np.testing.assert_array_equal(x,y)
                for _ in range(2):run()
                if a.backward_groups:b.time_call(run,b.main_stream,repeats=100)
                times=[b.time_call(run,b.main_stream,repeats=5) for _ in range(5)]
                row=dict(kind=kind,config=config,median_ms=float(np.median(times)),relative_l2=errors,valid=valid)
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2));b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
