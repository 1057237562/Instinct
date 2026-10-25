"""Two pinned host/device slots on a dedicated CUDA HtoD stream."""
import ctypes as ct
import numpy as np


class DevicePrefetch:
    def __init__(self,engine,batch_size,sequence_length,enabled=False):
        self.e,self.b,self.enabled=engine,engine.backend,enabled
        self.slots=[]
        if not enabled:return
        b=self.b;p=ct.c_void_p;u64=ct.c_uint64;size=ct.c_size_t
        self.alloc=b._bind('cuMemHostAlloc',[ct.POINTER(p),size,ct.c_uint])
        self.free=b._bind('cuMemFreeHost',[p])
        self.upload=b._bind('cuMemcpyHtoDAsync_v2',[u64,p,size,p])
        self.stream=b._make_stream(0)
        capacity=batch_size*sequence_length
        try:
            for _ in range(2):
                slot={'device':[],'host':[],'ready':b.create_event()}
                self.slots.append(slot)
                for _ in range(2):
                    ptr=p();self.alloc(ct.byref(ptr),capacity*8,0);slot['host'].append(ptr)
                    slot['device'].append(b.empty((capacity,),'i64'))
        except BaseException:self.close();raise
    def __enter__(self):return self
    def _stage(self,slot,batch):
        if not self.enabled:return batch
        # The training loop finishes a batch before reusing its slot. This event
        # also protects pinned bytes if a consumer stops before compute begins.
        if slot.get('used'):self.b._event_sync(slot['ready'])
        values=[]
        with self.b.on_stream(self.stream):
            for array,host,device in zip(batch,slot['host'],slot['device']):
                array=np.ascontiguousarray(array,dtype=np.int64)
                if array.size>device.numel:raise ValueError('prefetch batch exceeds slot capacity')
                ct.memmove(host,array.ctypes.data,array.nbytes)
                self.upload(device.storage,host,array.nbytes,self.stream)
                from model.dense_triton.runtime import Tensor
                values.append(self.e.wrap(Tensor(array.shape,'i64',device.storage,self.b)))
            self.b.record_event(slot['ready'])
        slot['used']=True
        return tuple(values)
    def batches(self,source):
        if not self.enabled:
            for begin,batch in source:yield begin,batch,batch
            return
        source=iter(source)
        def stage(index):
            item=next(source,None)
            if item is None:return None
            begin,batch=item
            return begin,batch,self._stage(self.slots[index],batch),index
        current=stage(0)
        if current is None:return
        following=stage(1)
        while current is not None:
            begin,batch,device,index=current
            self.b.wait_event(self.slots[index]['ready'])
            yield begin,batch,device
            # Ensure all readers are done before this slot can be reused.
            self.b.synchronize()
            current=following
            following=stage(index) if current is not None else None
    def close(self):
        if self.enabled:
            self.b.synchronize()
            for slot in self.slots:
                for ptr in slot['host']:self.free(ptr)
            self.slots=[]
    def __exit__(self,*exc):self.close()
