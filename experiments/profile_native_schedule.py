"""Shape-only native allocation/dispatch audit; no GPU timings or numeric results.

Runs the real Model/Engine/Optimizer/StepGraph with a metadata CUDA backend.
Use --repo to audit an isolated old checkout with this same measurement code.
GEMM/library workspaces, driver modules and allocator fragmentation are excluded.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import ctypes as ct
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--config',type=Path)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--seq',type=int,default=256)
    p.add_argument('--checkpoint',type=int,choices=[0,1,2],default=1)
    p.add_argument('--mode',choices=['eager','graph'],default='eager')
    p.add_argument('--fp8',action='store_true');p.add_argument('--loss-chunk',type=int,default=0)
    p.add_argument('--dense',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    import numpy as np
    from model.dense_triton.runtime import ITEMSIZE
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas.graph import StepGraph

    class MetadataBackend(TrainingBackend):
        def __init__(self):
            super().__init__(pin_memory=False)
            self.arch=120;self.main_stream=self.weight_stream=self.stream=1
            self.address=256;self.raw={};self.peak=0;self.syncs=0;self.mallocs=0
            self.operations=Counter();self.event=0
            self.ctas=Counter();self.copy_bytes=Counter();self.casts=Counter();self.gemm_flops=0
            self.peak_live=0;self.last_operation='parameters';self.peak_allocation=None;self.shapes={};self.peak_groups={}
            def malloc(out,size):
                self.address+=(size+255)//256*256
                ct.cast(out,ct.POINTER(ct.c_uint64))[0]=self.address
                self.raw[self.address]=size;self.mallocs+=1
                self.peak=max(self.peak,sum(self.raw.values()))
            self._malloc=malloc;self._free=lambda pointer:self.raw.pop(pointer)
        def _check(self):pass
        def empty(self,shape,dtype='fp32',zero=False):
            result=super().empty(shape,dtype,zero)
            self.shapes[result.storage]=(tuple(shape),dtype)
            live=sum(size for serial,size in self._live.values())
            if live>self.peak_live:
                self.peak_live=live;self.peak_allocation=dict(after=self.last_operation,shape=shape,dtype=dtype)
                groups=Counter()
                for ptr,(_,size) in self._live.items():groups[str(self.shapes.get(ptr))]+=size
                self.peak_groups=dict(groups.most_common(12))
            return result
        def synchronize(self):self.syncs+=1
        def from_numpy(self,array):
            array=np.asarray(array)
            with self.host_constants() if hasattr(self,'host_constants') else nullcontext():
                return self.empty(array.shape or (1,),{'f':'fp32','i':'i64','u':'i64'}[array.dtype.kind])
        def to_numpy(self,tensor):
            self.synchronize()
            return np.ones(tensor.shape,dtype=np.float32)
        def launch(self,fn,grid,values,**constants):
            self.last_operation='triton:'+fn.__name__
            self.operations['triton:'+fn.__name__]+=1
            self.ctas[fn.__name__]+=math.prod(grid)
            if fn.__name__=='pack_matrix':
                self.copy_bytes['gemm_pack']+=(constants['ROWS']*constants['COLS']+constants['PAD_ROWS']*constants['PAD_COLS'])*ITEMSIZE[values[0].dtype]
            elif fn.__name__=='unpack_matrix':
                self.copy_bytes['gemm_crop']+=constants['ROWS']*constants['COLS']*(ITEMSIZE[values[0].dtype]+ITEMSIZE[values[1].dtype])
            elif fn.__name__=='strided_copy':
                self.copy_bytes['strided_copy']+=constants['N']*(ITEMSIZE[values[0].dtype]+ITEMSIZE[values[1].dtype])
            elif fn.__name__=='cast':
                self.copy_bytes['cast']+=constants['SIZE']*(ITEMSIZE[values[0].dtype]+ITEMSIZE[values[1].dtype])
                self.casts[str(values[0].shape)+' '+values[0].dtype+'->'+values[1].dtype]+=1
            elif fn.__name__=='projection_cast':
                for i in range(3):
                    self.copy_bytes['projection_cast']+=constants['N'+str(i)]*(ITEMSIZE[values[i].dtype]+ITEMSIZE[values[i+3].dtype])
            if fn.__name__ in ('grouped_mm','grouped_dw'):
                self.gemm_flops+=2*values[0].shape[0]*constants['N']*constants['K']
            self._submit('triton:'+fn.__name__,lambda:None)
        def matmul(self,*values,role='gemm',**layout):
            self.last_operation='cublasLt:'+role
            self.operations['cublasLt:'+role]+=1
            self.gemm_flops+=2*math.prod(values[-3:])
            self._submit('cublasLt:'+role,lambda:None)
        def create_event(self,timing=False):
            self.event+=1;return self.event
        def record_event(self,event):pass
        def wait_event(self,event):pass

    class MetadataModel(Model):
        def _parameter(self,name,shape,**kwargs):
            value=self.e.wrap(self.e.backend.empty(shape,self.param_dtype),True,name)
            self.params[name]=value;return value

    config=dict(hidden_size=768,num_hidden_layers=8,num_attention_heads=8,num_key_value_heads=4,
        vocab_size=6400,use_moe=True,num_experts=4,num_experts_per_tok=1,
        intermediate_size=2432,moe_intermediate_size=2432,max_position_embeddings=32768,
        dropout=0.,flash_attn=True,tie_word_embeddings=True)
    if args.config:config.update(json.loads(args.config.read_text()))
    config['use_grad_checkpoint']=args.checkpoint
    if args.dense:config['use_moe']=False
    b=MetadataBackend();e=Engine(b,fp8=args.fp8);m=MetadataModel(e,config)
    m.loss_chunk_size=args.loss_chunk
    opt=Optimizer(e,m.parameters());persistent=sum(b.raw.values())
    b.operations.clear();b.ctas.clear();b.copy_bytes.clear();b.casts.clear();b.gemm_flops=0
    b.syncs=b.mallocs=0;b.peak=persistent
    batch=np.ones((args.batch,args.seq),np.int64)
    if args.mode=='graph':
        from model.dense_cublas.backend import ExecutionPlan
        ExecutionPlan.capture=lambda self:None  # Only audit preparation and fixed addresses.
        def loss(values):
            output=m(*values,return_logits=False)
            return output.loss+output.aux_loss
        graph=StepGraph(m,opt,(batch,batch),loss,2)
        tape_operations=Counter(command.label for command in graph.plan.commands)
    else:
        out=m(batch,batch,return_logits=False)
        opt.backward(out.loss+out.aux_loss);opt.step()
        tape_operations=None
    result=dict(mode='metadata_'+args.mode,batch=args.batch,seq=args.seq,checkpoint=args.checkpoint,
        fp8=args.fp8,config=config,persistent_bytes=persistent,peak_arena_bytes=b.peak,
        live_bytes=sum(size for serial,size in b._live.values()),cached_bytes=b._pool_bytes,
        peak_live_bytes=b.peak_live,peak_allocation=b.peak_allocation,
        largest_live_groups_at_peak=b.peak_groups,
        host_synchronizations=b.syncs,driver_allocations=b.mallocs,
        dispatches=sum((tape_operations if tape_operations is not None else b.operations).values()),
        preparation_dispatches=sum(b.operations.values()),operations=dict(b.operations),casts_by_shape=dict(b.casts),
        triton_ctas=dict(b.ctas),explicit_copy_bytes=dict(b.copy_bytes),gemm_flops=b.gemm_flops,
        tape_operations=None if tape_operations is None else dict(tape_operations),
        limitations='Shape/schedule audit only; no GPU execution, measured speed, library/context memory or numerical validation')
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
