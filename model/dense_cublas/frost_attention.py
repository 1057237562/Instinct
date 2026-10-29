"""Opt-in, Torch-free adapter to NVIDIA's SM120 CuTe backward kernels.

The separately installed frontend source and DSL remain optional. Only kernel
modules are loaded: the frontend's Python graph/Torch bindings are not needed.
Packed documents are staged into bounded dense groups, with explicit lengths;
the staging and scatter are part of every performance comparison.
"""
import math
import os
from pathlib import Path
import site
import sys
import types
import numpy as np
import triton
import triton.language as tl


def load_template():
    root=Path(__file__).resolve().parents[2]
    source=Path(os.environ.get('INSTINCT_CUDNN_FRONTEND_SOURCE',root/'.cache/cudnn-frontend'))/'python/cudnn'
    dependencies=Path(os.environ.get('INSTINCT_CUTE_DSL_PATH',root/'.cache/native-dsl'))
    if not source.is_dir():raise RuntimeError('SM120 DSL requires a cudnn-frontend source checkout')
    if dependencies.is_dir():site.addsitedir(str(dependencies))
    # Load the upstream kernel subpackages without importing its compiled graph
    # bindings. Never replace an already imported frontend package.
    for suffix in ('','frost','frost.tile_dsl','sdpa','sdpa.bwd','sdpa.bwd.kernels','sdpa.bwd.kernels.sm120'):
        name='cudnn'+('.'+suffix if suffix else '')
        if name in sys.modules:continue
        path=source.joinpath(*suffix.split('.')) if suffix else source
        module=types.ModuleType(name);module.__path__=[str(path)];module.__file__=str(path/'__init__.py')
        sys.modules[name]=module
    from cudnn.frost.template_loader import load_template as load
    from cudnn.sdpa.bwd.config_sm120 import TemplateParams
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    return load(str(source/'sdpa/bwd/kernels/sm120/bprop_f16.py'),TemplateParams(
        dtype_qkv=DTYPE_BF16,is_causal=True,causal_top_left=True,
        seq_q_lens_present=True,seq_kv_lens_present=True),tag='instinct_sm120_bf16')


@triton.jit
def gather(Q,K,V,O,DY,PQ,PK,PV,PO,PDY,OFFSETS,LENGTHS,
           SEQ:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,B:tl.constexpr=1024):
    doc=tl.program_id(1);i=tl.program_id(0)*B+tl.arange(0,B)
    start=tl.load(OFFSETS+doc);length=tl.load(LENGTHS+doc)
    qvalid=(i<SEQ*H*D)&(i//(H*D)<length);qo=doc*SEQ*H*D+i
    tl.store(PQ+qo,tl.load(Q+start+i,qvalid,0),i<SEQ*H*D)
    tl.store(PO+qo,tl.load(O+start+i,qvalid,0),i<SEQ*H*D)
    tl.store(PDY+qo,tl.load(DY+start+i,qvalid,0),i<SEQ*H*D)
    kvstart=start//(H*D)*(HK*D);kvvalid=(i<SEQ*HK*D)&(i//(HK*D)<length);ko=doc*SEQ*HK*D+i
    tl.store(PK+ko,tl.load(K+kvstart+i,kvvalid,0),i<SEQ*HK*D)
    tl.store(PV+ko,tl.load(V+kvstart+i,kvvalid,0),i<SEQ*HK*D)


@triton.jit
def scatter(PQ,PK,PV,Q,K,V,OFFSETS,LENGTHS,
            SEQ:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,B:tl.constexpr=1024):
    doc=tl.program_id(1);i=tl.program_id(0)*B+tl.arange(0,B)
    start=tl.load(OFFSETS+doc);length=tl.load(LENGTHS+doc)
    qvalid=(i<SEQ*H*D)&(i//(H*D)<length)
    tl.store(Q+start+i,tl.load(PQ+doc*SEQ*H*D+i,qvalid,0),qvalid)
    kvstart=start//(H*D)*(HK*D);valid=(i<SEQ*HK*D)&(i//(HK*D)<length)
    tl.store(K+kvstart+i,tl.load(PK+doc*SEQ*HK*D+i,valid,0),valid)
    tl.store(V+kvstart+i,tl.load(PV+doc*SEQ*HK*D+i,valid,0),valid)


class FrostBackward:
    def __init__(self,backend):
        if backend.arch!=120:raise ValueError('SM120 backward requires an SM120 device')
        self.b=backend;self.template=load_template();self.plans={}
        self.metrics={'calls':0,'plans':0,'max_workspace_bytes':0}

    def plan(self,docs,seq,h,hk,d):
        key=(docs,seq,h,hk,d)
        if key not in self.plans:
            if len(self.plans)>=64:raise RuntimeError('SM120 backward plan budget exceeded')
            owner=self.template.compile((12,0),b=docs,qh=h,kvh=hk,sq=seq,skv=seq,d_qk=d,d_v=d)
            from cudnn.frost.compiled_cache import positional_entry
            entry=positional_entry(owner.entry)
            if entry is None:raise RuntimeError('SM120 backward has no pointer entry')
            if owner.workspace_bytes>512*2**20:raise RuntimeError('SM120 backward workspace exceeds 512 MiB')
            self.plans[key]=(owner,entry)
            self.metrics['plans']=len(self.plans)
            self.metrics['max_workspace_bytes']=max(self.metrics['max_workspace_bytes'],owner.workspace_bytes)
        return self.plans[key]

    def run(self,q,k,v,out,dy,stats,metadata,dq,dk,dv,docs,seq,h,hk,d):
        owner,entry=self.plan(docs,seq,h,hk,d);b=self.b
        pq,pk,pv,po,pdy,pdq,pdk,pdv=[b.empty((docs,seq,heads,d),'bf16') for heads in (h,hk,hk,h,h,h,hk,hk)]
        grid=(triton.cdiv(seq*h*d,1024),docs)
        b.launch(gather,grid,(q,k,v,out,dy,pq,pk,pv,po,pdy,metadata[0],metadata[3]),SEQ=seq,H=h,HK=hk,D=d)
        workspace=b.empty((triton.cdiv(owner.workspace_bytes,4),),'fp32')
        # Match attention.cpp's float reciprocal of float sqrt exactly; the
        # forward LSE was computed using this rounded scale.
        scale=float(np.float32(1)/np.sqrt(np.float32(d)))
        frame=[x.storage for x in (pq,pk,pv,po,pdy,stats,pdq,pdk,pdv,metadata[3],metadata[3])]
        frame += [None]*4+[workspace.storage,scale*math.log2(math.e),scale,b.main_stream]
        b._submit('frost:attention_backward',lambda:entry(*frame))
        b.launch(scatter,grid,(pdq,pdk,pdv,dq,dk,dv,metadata[0],metadata[3]),SEQ=seq,H=h,HK=hk,D=d)
        self.metrics['calls']+=1
