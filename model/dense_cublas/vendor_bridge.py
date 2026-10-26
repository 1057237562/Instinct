"""Optional cuDNN C ABI; no cuDNN Python frontend or Torch dependency."""
import ctypes as ct
import os
from pathlib import Path


class Sdpa:
    def __init__(self,library=None):
        root=Path(__file__).resolve().parents[2]
        runtime=Path(os.environ.get('CUDNN_PATH',root/'.cache/native-libs/nvidia/cudnn'))
        self.dirs=[]
        if os.name=='nt':
            directories=[runtime/'bin']
            if os.environ.get('CUDA_PATH'):
                directories.extend([Path(os.environ['CUDA_PATH'])/'bin',Path(os.environ['CUDA_PATH'])/'bin/x64'])
            self.dirs=[os.add_dll_directory(str(p)) for p in directories if p.is_dir()]
        # cuDNN dispatches further DLL loads itself using the process search path.
        if (runtime/'bin').is_dir():os.environ['PATH']=str(runtime/'bin')+os.pathsep+os.environ['PATH']
        selected=library or os.environ.get('INSTINCT_CUDNN_BRIDGE')
        candidates=[Path(selected)] if selected else [root/'model/dense_cublas/native/build/Release/native_sdpa.dll',
            root/'model/dense_cublas/native/build/libnative_sdpa.so',root/'.cache/native-sdpa-build/Release/native_sdpa.dll']
        target=next((p for p in candidates if p.is_file()),None)
        if target is None:
            for directory in self.dirs:directory.close()
            raise RuntimeError('Build the optional native SDPA bridge or set INSTINCT_CUDNN_BRIDGE; CUDNN_PATH selects its runtime')
        self.lib=ct.CDLL(str(target.resolve()))
        self.lib.sdpa_error.restype=ct.c_char_p
        self.lib.sdpa_create.argtypes=[ct.c_int64]*5+[ct.c_int,ct.c_int64,ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_uint64)]
        self.lib.sdpa_run.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p]
        self.lib.sdpa_destroy.argtypes=[ct.c_void_p]
        self.plans=[]

    def check(self,status):
        if status:raise RuntimeError(self.lib.sdpa_error().decode())

    def plan(self,b,s,h,hk,d,backward,stream,total=0):
        p=ct.c_void_p();workspace=ct.c_uint64()
        self.check(self.lib.sdpa_create(b,s,h,hk,d,backward,total,stream,ct.byref(p),ct.byref(workspace)))
        self.plans.append(p)
        return p,workspace.value

    def run(self,plan,buffers,workspace):
        pointers=(ct.c_void_p*len(buffers))(*(x.storage if x is not None else 0 for x in buffers))
        self.check(self.lib.sdpa_run(plan,pointers,workspace.storage))

    def close(self):
        for p in self.plans:self.lib.sdpa_destroy(p)
        self.plans=[]
        for directory in self.dirs:directory.close()
        self.dirs=[]
