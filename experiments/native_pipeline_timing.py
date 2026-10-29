"""Diagnostic timing at existing update fences, without new synchronizations."""
import ctypes as ct
import time


class PipelineTiming:
    def __init__(self,trainer,backend,optimizer,data):
        self.trainer,self.b,self.opt,self.data=trainer,backend,optimizer,data
        self.saved=(trainer.supervised_step,trainer.DevicePrefetch,optimizer.backward,data.batches)
        self.events=[];self.pending=[];self.cpu={}
        def add(name,seconds):self.cpu[name]=self.cpu.get(name,0.)+seconds
        def forward(*args,**kwargs):
            index=len(self.pending)
            if index==len(self.events):self.events.append((backend.create_event(timing=True),backend.create_event(timing=True)))
            pair=self.events[index];backend.record_event(pair[0]);begin=time.perf_counter()
            result=self.saved[0](*args,**kwargs)
            add('forward_submit_seconds',time.perf_counter()-begin);self.pending.append(pair)
            return result
        def backward(*args,**kwargs):
            begin=time.perf_counter();result=self.saved[2](*args,**kwargs)
            add('backward_submit_seconds',time.perf_counter()-begin);backend.record_event(self.pending[-1][1])
            return result
        def batches(*args,**kwargs):
            source=iter(self.saved[3](*args,**kwargs))
            while True:
                begin=time.perf_counter()
                try:item=next(source)
                except StopIteration:return
                add('loader_next_seconds',time.perf_counter()-begin)
                yield item
        def prefetch(*args,**kwargs):
            instance=self.saved[1](*args,**kwargs);original=instance._stage
            def stage(*values,**options):
                begin=time.perf_counter();result=original(*values,**options)
                add('h2d_stage_host_seconds',time.perf_counter()-begin)
                return result
            instance._stage=stage
            return instance
        trainer.supervised_step=forward;trainer.DevicePrefetch=prefetch
        optimizer.backward=backward;data.batches=batches

    def drain(self):
        """Caller must have completed its existing accumulated-update fence."""
        durations=[]
        for start,end in self.pending:
            elapsed=ct.c_float();self.b._elapsed(ct.byref(elapsed),start,end);durations.append(elapsed.value)
        result=dict(self.cpu,forward_backward_stream_ms=sum(durations),microsteps=len(durations),
            scope='CUDA event spans include GPU launch gaps; CPU durations overlap GPU work and must not be added to stream spans')
        self.pending=[];self.cpu={}
        return result

    def close(self):
        self.trainer.supervised_step,self.trainer.DevicePrefetch,self.opt.backward,self.data.batches=self.saved
