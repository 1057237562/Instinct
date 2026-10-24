"""Load the small C ABI shim built against public CUDA/cuBLASLt headers."""
import ctypes as ct
import os
from pathlib import Path


DTYPES = {'fp32': 0, 'bf16': 1, 'fp16': 2, 'fp8e4nv': 3, 'fp8e5': 4}


class LtError(RuntimeError):
    pass


class NativeLt:
    def __init__(self, path=None):
        self._dll_dirs = []
        if os.name == 'nt' and os.environ.get('CUDA_PATH'):
            self._dll_dirs.append(os.add_dll_directory(str(Path(os.environ['CUDA_PATH']) / 'bin')))
        root = Path(__file__).parent / 'native' / 'build'
        selected = path or os.environ.get('INSTINCT_CUBLASLT_BRIDGE')
        candidates = [Path(selected)] if selected else [root / 'libinstinct_cublaslt.so',
                        root / 'Release' / 'instinct_cublaslt.dll', root / 'instinct_cublaslt.dll']
        target = next((p for p in candidates if p.is_file()), None)
        if target is None:
            raise RuntimeError('Build model/dense_cublas/native with CMake and CUDA >=12.8; '
                               'or set INSTINCT_CUBLASLT_BRIDGE to the built library')
        self.lib = ct.CDLL(str(target.resolve()))
        ptr, integer, u64 = ct.c_void_p, ct.c_int, ct.c_uint64
        self._error = self._bind('error', [], ct.c_char_p)
        self._abi = self._bind('abi', [], integer)
        if self._abi() != 1:
            raise RuntimeError('cuBLASLt bridge ABI mismatch; rebuild the native library')
        self.version = int(self._bind('version', [], u64)())
        self._create = self._bind('create', [ct.POINTER(ptr)])
        self._destroy = self._bind('destroy', [ptr])
        self._plan = self._bind('plan', [ptr] + [ct.c_int64] * 3 + [integer] * 3
                                + [u64, integer, integer, ct.POINTER(ptr)])
        self._plan_destroy = self._bind('plan_destroy', [ptr])
        self._count = self._bind('candidate_count', [ptr])
        self._info = self._bind('candidate_info', [ptr, integer, ct.POINTER(u64), ct.POINTER(integer), ct.POINTER(ct.c_float)])
        self._run = self._bind('run', [ptr, integer] + [ptr] * 6 + [u64, ptr])
        self.handle = ptr()
        self.check(self._create(ct.byref(self.handle)))
        self.plans = []

    def _bind(self, suffix, types, result=ct.c_int):
        fn = getattr(self.lib, 'instinct_lt_' + suffix)
        fn.argtypes, fn.restype = types, result
        return fn

    def check(self, result):
        if result != 0:
            raise LtError(self._error().decode('utf-8', errors='replace'))

    def create_plan(self, m, n, k, a_dtype, b_dtype, out_dtype, workspace_bytes, candidates, sm_target):
        plan = ct.c_void_p()
        self.check(self._plan(self.handle, m, n, k, DTYPES[a_dtype], DTYPES[b_dtype], DTYPES[out_dtype],
                              workspace_bytes, candidates, sm_target, ct.byref(plan)))
        self.plans.append(plan)
        info = []
        for i in range(self._count(plan)):
            size, algo, waves = ct.c_uint64(), ct.c_int(), ct.c_float()
            self.check(self._info(plan, i, ct.byref(size), ct.byref(algo), ct.byref(waves)))
            info.append({'index': i, 'workspace_bytes': size.value, 'algo_id': algo.value, 'waves': waves.value})
        return plan, info

    def run(self, plan, index, a, b, out, sa, sb, workspace, workspace_bytes, stream):
        self.check(self._run(plan, index, a, b, out, sa, sb, workspace, workspace_bytes, stream))

    def close(self):
        first_error = None
        for p in reversed(self.plans):
            try:
                self.check(self._plan_destroy(p))
            except Exception as exc:
                first_error = first_error or exc
        self.plans.clear()
        if self.handle:
            try:
                self.check(self._destroy(self.handle))
            except Exception as exc:
                first_error = first_error or exc
            self.handle = None
        for directory in self._dll_dirs:
            directory.close()
        self._dll_dirs.clear()
        if first_error:
            raise first_error
