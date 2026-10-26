"""Optional native packed SDPA with bounded plans and compact saved LSE."""
from collections import OrderedDict
import ctypes as ct
import time
import numpy as np
import triton
import triton.language as tl
from .vendor_bridge import Sdpa


@triton.jit
def stats_copy(PADDED,COMPACT,OFFSETS,LENGTHS,SEQ:tl.constexpr,HEADS:tl.constexpr,EXPAND:tl.constexpr,BLOCK:tl.constexpr=256):
    doc,head=tl.program_id(0),tl.program_id(1)
    i=tl.program_id(2)*BLOCK+tl.arange(0,BLOCK)
    start=tl.load(OFFSETS+doc)//HEADS
    length=tl.load(LENGTHS+doc)
    padded=(doc*HEADS+head)*SEQ+i
    compact=(start+i)*HEADS+head
    if EXPAND:
        value=tl.load(COMPACT+compact,i<length,0.)
        tl.store(PADDED+padded,value,i<SEQ)
    else:
        value=tl.load(PADDED+padded,i<length,0.)
        tl.store(COMPACT+compact,value,i<length)


class NativeAttention:
    def __init__(self,backend):
        self.b=backend;self.lib=Sdpa();self.plans=OrderedDict();self.context=None
        self.forward_calls=self.backward_calls=0
        self.upload_slots=[];self.upload_index=0
        self.metrics=dict(plan_builds=0,plan_evictions=0,plan_build_seconds=0.,prepare_seconds=0.,max_workspace_bytes=0,workspace_fallbacks=0)

    def upload_metadata(self,arrays):
        if not getattr(self.b,'async_attention_metadata',False) or self.b._commands is not None:
            return [self.b.from_numpy(x) for x in arrays]
        from model.dense_triton.runtime import Tensor
        offsets=[];size=0
        for array in arrays:offsets.append(size);size+=triton.cdiv(array.nbytes,256)*256
        capacity=1024*1024
        if size>capacity:return [self.b.from_numpy(x) for x in arrays]
        if not self.upload_slots:
            ptr=ct.c_void_p
            alloc=self.b._bind('cuMemHostAlloc',[ct.POINTER(ptr),ct.c_size_t,ct.c_uint])
            self.free_host=self.b._bind('cuMemFreeHost',[ptr])
            self.upload_async=self.b._bind('cuMemcpyHtoDAsync_v2',[ct.c_uint64,ptr,ct.c_size_t,ptr])
            for _ in range(2):
                host=ptr();alloc(ct.byref(host),capacity,0)
                ready=self.b.create_event()
                self.b._pinned_events.add(ready)
                self.upload_slots.append(dict(host=host,ready=ready,used=False))
        slot=self.upload_slots[self.upload_index%2];self.upload_index+=1
        if slot['used']:self.b._event_sync(slot['ready'])
        for array,offset in zip(arrays,offsets):ct.memmove(slot['host'].value+offset,array.ctypes.data,array.nbytes)
        storage=self.b.empty((size//8,),'i64')
        self.upload_async(storage.storage,slot['host'],size,self.b.stream)
        self.b.record_event(slot['ready']);slot['used']=True
        return [Tensor(x.shape,'i64' if x.dtype==np.int64 else 'i32',storage.storage+offset,self.b)
                for x,offset in zip(arrays,offsets)]

    def prepare(self,shape,segments,heads,kv_heads,dim):
        begin=time.perf_counter()
        self.context=None
        if shape[1]<getattr(self,'minimum_seq',0):return
        if segments is None:return
        ids=np.asarray(segments,np.int64)
        if ids.shape!=shape:raise ValueError('native attention host metadata shape mismatch')
        batch,seq=shape;starts=[];lengths=[]
        for row in range(batch):
            boundaries=np.r_[0,np.flatnonzero(ids[row,1:]!=ids[row,:-1])+1,seq]
            for start,end in zip(boundaries[:-1],boundaries[1:]):
                if ids[row,start]>=0:starts.append(row*seq+int(start));lengths.append(int(end-start))
        if not lengths:return
        # Keep descriptor geometry stable across similar document mixes. Padded
        # zero-length documents are deliberately not used by this first probe.
        quantum=getattr(self.b,'attention_plan_quantum',128)
        if quantum<128 or quantum%128:raise ValueError('attention plan quantum must be a positive multiple of 128')
        maximum=min(triton.cdiv(seq,128)*128,triton.cdiv(max(lengths),quantum)*quantum)
        if len(lengths)*maximum*heads*4>64*2**20:return
        geometry=(len(lengths),maximum,heads,kv_heads,dim,batch*seq,[])
        limit=getattr(self.b,'attention_workspace_limit_mib',1792)*2**20
        workspaces=[self.plan(chunk,backward)[1] for chunk in self.chunks(geometry) for backward in (False,True)]
        if max(workspaces)>limit:
            self.metrics['workspace_fallbacks']+=1
            self.metrics['prepare_seconds']+=time.perf_counter()-begin
            return
        offsets=np.asarray(starts+[batch*seq],np.int64)
        metadata=self.upload_metadata([offsets*width for width in (heads*dim,kv_heads*dim,heads)]
            +[np.asarray(lengths,np.int32)])
        self.context=(len(lengths),maximum,heads,kv_heads,dim,batch*seq,metadata)
        self.metrics['prepare_seconds']+=time.perf_counter()-begin

    def chunks(self,context):
        from model.dense_triton.runtime import Tensor,ITEMSIZE
        docs,maximum,h,hk,d,total,metadata=context
        size=getattr(self.b,'attention_docs_per_plan',0) or docs
        for start in range(0,docs,size):
            count=min(size,docs-start)
            views=[Tensor((count+(index<3),),x.dtype,x.storage+start*ITEMSIZE[x.dtype],self.b)
                   for index,x in enumerate(metadata)]
            yield count,maximum,h,hk,d,total,views

    def plan(self,context,backward):
        docs,maximum,h,hk,d,total,_=context
        key=(docs,maximum,h,hk,d,total,backward)
        if key not in self.plans:
            # Descriptors own runtime resources, but never retain GPU workspaces.
            if len(self.plans)>=getattr(self.b,'attention_plan_limit',32):
                self.b.synchronize()
                _,(old,_) = self.plans.popitem(last=False)
                self.lib.lib.sdpa_destroy(old);self.lib.plans.remove(old)
                self.metrics['plan_evictions']+=1
            begin=time.perf_counter()
            self.plans[key]=self.lib.plan(docs,maximum,h,hk,d,backward,self.b.main_stream,total)
            self.metrics['plan_builds']+=1
            self.metrics['plan_build_seconds']+=time.perf_counter()-begin
            self.metrics['max_workspace_bytes']=max(self.metrics['max_workspace_bytes'],self.plans[key][1])
        self.plans.move_to_end(key)
        return self.plans[key]

    def attention(self,q,k,v,mask,dropout,seed,counter,cached,packed,fallback):
        # Bind metadata to the mask carried by this particular forward/replay,
        # rather than whichever model invocation most recently prepared a batch.
        context=getattr(mask,'native_attention_context',self.context)
        if not packed or context is None or q.dtype!='bf16' or dropout:
            return fallback(q,k,v,mask,dropout,seed,counter,cached,packed)
        docs,maximum,h,hk,d,total,metadata=context
        out,lse=cached if cached is not None else (self.b.empty(q.shape,q.dtype,zero=True),self.b.empty((total,h),'fp32',zero=True))
        if cached is None:
            self.forward_calls+=1
            for chunk in self.chunks(context):
                docs,maximum,h,hk,d,total,metadata=chunk
                plan,n=self.plan(chunk,False)
                mark=self.b.mark()
                stats=self.b.empty((docs,h,maximum),'fp32')
                work=self.b.empty((max(1,(n+3)//4),),'fp32')
                buffers=[q,k,v,out,stats,None,None,None,None,*metadata]
                self.b._submit('cudnn:attention_forward',lambda plan=plan,buffers=buffers,work=work:self.lib.run(plan,buffers,work))
                self.b.launch(stats_copy,(docs,h,triton.cdiv(maximum,256)),(stats,lse,metadata[2],metadata[3]),SEQ=maximum,HEADS=h,EXPAND=False)
                self.b.release_since(mark)
        return out,(q,k,v,mask,out,lse,('native_sdpa',context),seed,counter)

    def backward(self,saved,dy,fallback):
        if not isinstance(saved[6],tuple) or saved[6][0]!='native_sdpa':return fallback(saved,dy)
        q,k,v,_,out,lse,(_,context),_,_=saved
        self.backward_calls+=1
        docs,maximum,h,hk,d,total,metadata=context
        dq,dk,dv=[self.b.empty(x.shape,x.dtype,zero=True) for x in (q,k,v)]
        for chunk in self.chunks(context):
            docs,maximum,h,hk,d,total,metadata=chunk
            plan,n=self.plan(chunk,True)
            mark=self.b.mark()
            stats=self.b.empty((docs,h,maximum),'fp32')
            self.b.launch(stats_copy,(docs,h,triton.cdiv(maximum,256)),(stats,lse,metadata[2],metadata[3]),SEQ=maximum,HEADS=h,EXPAND=True)
            work=self.b.empty((max(1,(n+3)//4),),'fp32')
            buffers=[q,k,v,out,stats,dy,dq,dk,dv,*metadata]
            self.b._submit('cudnn:attention_backward',lambda plan=plan,buffers=buffers,work=work:self.lib.run(plan,buffers,work))
            self.b.release_since(mark)
        return dq,dk,dv

    def close(self):
        if self.upload_slots:
            self.b.synchronize()
            for slot in self.upload_slots:self.free_host(slot['host'])
            self.upload_slots=[]
        self.lib.close();self.plans.clear()
