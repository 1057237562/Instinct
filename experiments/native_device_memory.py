"""Optional device-wide NVML sampling, without a CUDA context or GPU barriers.

Samples are lower bounds on the peak, not process allocation accounting. WDDM
memory ownership belongs to Windows; do not subtract this counter from a CUDA
pool high-water mark taken at a different instant.
"""
import ctypes as ct
import os
import threading
import time


class DeviceMemorySampler:
    def __init__(self, device=0, interval=.02):
        if interval<=0:raise ValueError('positive sample interval required')
        self.device,self.interval=device,interval
        self.phase='setup';self.peaks={};self.error=None
        self._stop=threading.Event();self._thread=None;self._lib=None

    def __enter__(self):
        try:
            lib=ct.CDLL('nvml.dll' if os.name=='nt' else 'libnvidia-ml.so.1')
            def check(status):
                if status:raise RuntimeError(f'NVML status {status}')
            check(lib.nvmlInit_v2());self._lib=lib
            handle=ct.c_void_p()
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ct.c_uint,ct.POINTER(ct.c_void_p)]
            check(lib.nvmlDeviceGetHandleByIndex_v2(self.device,ct.byref(handle)))
            class Memory(ct.Structure):
                _fields_=[(name,ct.c_ulonglong) for name in ('total','free','used')]
            lib.nvmlDeviceGetMemoryInfo.argtypes=[ct.c_void_p,ct.POINTER(Memory)]
            def read():
                value=Memory();check(lib.nvmlDeviceGetMemoryInfo(handle,ct.byref(value)))
                return {name:int(getattr(value,name)) for name in ('total','free','used')}
            self._read=read;self._start=time.perf_counter()
            self._thread=threading.Thread(target=self._run,name='native-memory-sampler',daemon=True)
            self._thread.start()
        except Exception as exc:self.error=repr(exc)
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                phase=self.phase;value=self._read();now=time.perf_counter()-self._start
                row=self.peaks.setdefault(phase,dict(samples=0,first_seconds=now,peak_used_bytes=0))
                row['samples']+=1;row['last_seconds']=now
                if value['used']>=row['peak_used_bytes']:
                    row.update(peak_used_bytes=value['used'],free_bytes_at_peak=value['free'],
                               total_bytes=value['total'],peak_seconds=now)
            except Exception as exc:
                self.error=repr(exc);break
            self._stop.wait(self.interval)

    def __exit__(self,*exc):
        self._stop.set()
        if self._thread:self._thread.join(timeout=5)
        if self._thread and self._thread.is_alive():self.error='NVML sampling thread did not finish'
        elif self._lib:self._lib.nvmlShutdown()

    def report(self):
        return dict(scope='NVML device-wide used memory, including reserved memory; sampled lower bound, WDDM-managed, not per-process allocated memory',
                    device=self.device,interval_seconds=self.interval,phases=self.peaks,error=self.error)
