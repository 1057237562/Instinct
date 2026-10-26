"""CUPTI driver-activity collector used only by standalone profiling probes."""
import ctypes as ct
import json
import os
from pathlib import Path


class Collector:
    def __init__(self):
        root=Path(__file__).resolve().parents[2]
        self.directory=os.add_dll_directory(str(Path(os.environ['CUDA_PATH'])/'extras/CUPTI/lib64'))
        self.lib=ct.CDLL(str(root/'.cache/cupti-trace-build/Release/instinct_cupti_trace.dll'))
        self.lib.trace_error.restype=ct.c_char_p
        self.lib.trace_push.argtypes=[ct.c_uint64];self.lib.trace_stop.argtypes=[ct.c_char_p]
        self.active=False;self.phase='';self.labels={};self.host={}
    def check(self,status):
        if status:raise RuntimeError(f'CUPTI status {status}: '+self.lib.trace_error().decode())
    def start(self):
        self.labels={};self.host={};self.check(self.lib.trace_start());self.active=True
    def push(self,name):
        name=self.phase+'/'+name
        label=self.labels.setdefault(name,len(self.labels)+1)
        self.check(self.lib.trace_push(label));return name
    def pop(self):self.check(self.lib.trace_pop())
    def stop(self,path):
        self.active=False;path=Path(path)
        self.check(self.lib.trace_stop(str(path).encode()))
        path.with_suffix('.labels.json').write_text(json.dumps(dict(labels=self.labels,host=self.host),indent=2))
        return str(path)
