"""Small CUDA Driver API launcher: Triton compilation without importing torch.

Single device, single stream, explicit context lifetime. Allocations are owned
until context exit (an arena); callers should reuse buffers for training loops.
This experimental launcher pins Triton 3.4's ABI and rejects scratch/cluster
kernels instead of silently launching an incompatible binary.
"""
import ctypes as ct
from dataclasses import dataclass
import inspect
import math
import os

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


ITEMSIZE = {'fp32': 4, 'bf16': 2, 'fp16': 2, 'fp8e4nv': 1, 'fp8e5': 1,
            'i32': 4, 'i64': 8}


@dataclass
class Tensor:
    shape: tuple
    dtype: str
    storage: object
    owner: object

    @property
    def numel(self):
        return math.prod(self.shape)

    def view(self, *shape):
        if math.prod(shape) != self.numel:
            raise ValueError('view must preserve the element count')
        return Tensor(tuple(shape), self.dtype, self.storage, self.owner)


def constants_for(fn, values):
    """Fill constexpr defaults without initializing Triton's default driver."""
    result = {name: p.default for name, p in inspect.signature(fn.fn).parameters.items()
              if p.default is not inspect.Parameter.empty}
    result.update(values)
    return result


def compile_kernel(fn, dtypes, constants, arch=89):
    if not triton.__version__.startswith('3.4.'):
        raise RuntimeError('This experimental direct launcher requires Triton 3.4.x')
    constants = constants_for(fn, constants)
    names = [name for name in fn.arg_names if name not in constants]
    if len(names) != len(dtypes):
        raise ValueError(f'{fn.__name__}: runtime argument count does not match signature')
    source = ASTSource(fn, signature={n: '*' + t for n, t in zip(names, dtypes)},
                       constexprs=constants)
    # Explicit target avoids GPUDriver.__init__, which otherwise imports torch.
    return triton.compile(source, target=GPUTarget('cuda', arch, 32),
                           options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})


class CudaBackend:
    """Use as a context manager. Not thread-safe; no implicit allocator cache."""
    def __init__(self, device=0):
        self.device = device
        self._allocations = []
        self._modules = []
        self._cache = {}
        self._context = None

    def _bind(self, name, types):
        func = getattr(self.lib, name)
        func.argtypes, func.restype = types, ct.c_int
        def call(*args):
            status = func(*args)
            if status:
                raise RuntimeError(f'{name}: CUDA driver error {status}')
        return call

    def __enter__(self):
        if self._context is not None:
            raise RuntimeError('backend is already active')
        self.lib = ct.WinDLL('nvcuda.dll') if os.name == 'nt' else ct.CDLL('libcuda.so.1')
        ptr, u64, size, integer = ct.c_void_p, ct.c_uint64, ct.c_size_t, ct.c_int
        self._bind('cuInit', [ct.c_uint])(0)
        dev = integer()
        self._bind('cuDeviceGet', [ct.POINTER(integer), integer])(ct.byref(dev), self.device)
        attribute = self._bind('cuDeviceGetAttribute', [ct.POINTER(integer), integer, integer])
        major, minor = integer(), integer()
        attribute(ct.byref(major), 75, dev)
        attribute(ct.byref(minor), 76, dev)
        self.arch = major.value * 10 + minor.value
        if self.arch < 80:
            raise RuntimeError('Dense kernels require NVIDIA compute capability >= 8.0; FP8 requires >= 8.9')
        self._prior = ptr()
        self._bind('cuCtxGetCurrent', [ct.POINTER(ptr)])(ct.byref(self._prior))
        self._set_context = self._bind('cuCtxSetCurrent', [ptr])
        self._release = self._bind('cuDevicePrimaryCtxRelease_v2', [integer])
        self._dev = dev
        self._context = ptr()
        self._bind('cuDevicePrimaryCtxRetain', [ct.POINTER(ptr), integer])(ct.byref(self._context), dev)
        try:
            self._set_context(self._context)
            self._malloc = self._bind('cuMemAlloc_v2', [ct.POINTER(u64), size])
            self._free = self._bind('cuMemFree_v2', [u64])
            self._htod = self._bind('cuMemcpyHtoD_v2', [u64, ptr, size])
            self._dtoh = self._bind('cuMemcpyDtoH_v2', [ptr, u64, size])
            self._zero = self._bind('cuMemsetD8_v2', [u64, ct.c_ubyte, size])
            self._meminfo = self._bind('cuMemGetInfo_v2', [ct.POINTER(size)] * 2)
            self._sync = self._bind('cuCtxSynchronize', [])
            self._load = self._bind('cuModuleLoadData', [ct.POINTER(ptr), ptr])
            self._unload = self._bind('cuModuleUnload', [ptr])
            self._getfn = self._bind('cuModuleGetFunction', [ct.POINTER(ptr), ptr, ct.c_char_p])
            self._setattr = self._bind('cuFuncSetAttribute', [ptr, integer, integer])
            self._launch = self._bind('cuLaunchKernel', [ptr] + [ct.c_uint] * 7 + [ptr, ct.POINTER(ptr), ct.POINTER(ptr)])
        except Exception:
            self._set_context(self._prior)
            self._release(self._dev)
            self._context = None
            raise
        return self

    def _check(self):
        if self._context is None:
            raise RuntimeError('CUDA backend is not active')
        self._set_context(self._context)

    def empty(self, shape, dtype='fp32', zero=False):
        self._check()
        shape = tuple(shape)
        if not shape or min(shape) <= 0 or dtype not in ITEMSIZE:
            raise ValueError('expected a positive shape and supported dtype')
        if dtype.startswith('fp8') and self.arch < 89:
            raise RuntimeError('Native FP8 GEMM requires compute capability >= 8.9')
        pointer = ct.c_uint64()
        nbytes = math.prod(shape) * ITEMSIZE[dtype]
        self._malloc(ct.byref(pointer), nbytes)
        self._allocations.append((pointer, nbytes))
        if zero:
            self._zero(pointer, 0, nbytes)
        return Tensor(shape, dtype, pointer.value, self)

    def from_numpy(self, values):
        import numpy as np
        values = np.ascontiguousarray(values)
        mapping = {np.dtype('float32'): 'fp32', np.dtype('float16'): 'fp16',
                   np.dtype('int32'): 'i32', np.dtype('int64'): 'i64'}
        if values.dtype not in mapping:
            raise ValueError('upload accepts float32/float16/int32/int64; use a cast kernel for BF16')
        result = self.empty(values.shape, mapping[values.dtype])
        self._htod(result.storage, values.ctypes.data, values.nbytes)
        return result

    def to_numpy(self, tensor):
        import numpy as np
        self._check_tensor(tensor)
        dtype = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'uint16',
                 'fp8e4nv': 'uint8', 'fp8e5': 'uint8', 'i32': 'int32', 'i64': 'int64'}[tensor.dtype]
        result = np.empty(tensor.shape, dtype=dtype)
        self.synchronize()
        self._dtoh(result.ctypes.data, tensor.storage, result.nbytes)
        if tensor.dtype == 'bf16':
            return (result.astype(np.uint32) << 16).view(np.float32)
        # FP8 is returned as raw bytes, avoiding a dependency on a framework dtype.
        return result

    def _check_tensor(self, tensor):
        self._check()
        if tensor.owner is not self:
            raise ValueError('tensor belongs to a different backend/context')

    def launch(self, fn, grid, args, **constants):
        if not 1 <= len(grid) <= 3 or any(g <= 0 for g in grid):
            raise ValueError('launch grid must have 1..3 positive dimensions')
        for arg in args:
            self._check_tensor(arg)
        constants = constants_for(fn, constants)
        key = (fn.__name__, tuple(a.dtype for a in args), tuple(sorted(constants.items())))
        if key not in self._cache:
            kernel = compile_kernel(fn, [a.dtype for a in args], constants, self.arch)
            meta = kernel.metadata
            if (meta.global_scratch_size or meta.num_ctas != 1 or meta.launch_cooperative_grid
                    or meta.tensordesc_meta or meta.launch_pdl):
                raise RuntimeError('Unsupported Triton launch ABI: scratch/cluster/TMA/PDL')
            module, function = ct.c_void_p(), ct.c_void_p()
            binary = ct.create_string_buffer(kernel.asm['cubin'])
            self._load(ct.byref(module), binary)
            self._modules.append(module)
            self._getfn(ct.byref(function), module, meta.name.encode())
            if meta.shared > 48 * 1024:
                self._setattr(function, 8, meta.shared)  # MAX_DYNAMIC_SHARED_SIZE_BYTES
            self._cache[key] = (function, meta)
        function, meta = self._cache[key]
        # Triton 3.4 adds a global-scratch pointer even when its size is zero.
        values = [ct.c_uint64(arg.storage) for arg in args] + [ct.c_uint64(0)]
        params = (ct.c_void_p * len(values))(*[ct.addressof(v) for v in values])
        grid = tuple(grid) + (1,) * (3 - len(grid))
        self._launch(function, *grid, meta.num_warps * 32, 1, 1, meta.shared, None, params, None)

    def synchronize(self):
        self._check()
        self._sync()

    def memory(self):
        self.synchronize()
        free, total = ct.c_size_t(), ct.c_size_t()
        self._meminfo(ct.byref(free), ct.byref(total))
        return {'arena_bytes': sum(n for _, n in self._allocations),
                'device_free_bytes': free.value, 'device_total_bytes': total.value}

    def __exit__(self, exc_type, exc, traceback):
        # Always release the context even if synchronization/cleanup reports an error.
        error = None
        try:
            self.synchronize()
        except Exception as failure:
            error = failure
        for callback, values in [(self._free, [p for p, _ in self._allocations]),
                                 (self._unload, self._modules)]:
            for value in reversed(values):
                try:
                    callback(value)
                except Exception as failure:
                    error = error or failure
        try:
            self._set_context(self._prior)
        finally:
            self._release(self._dev)
            self._context = None
            self._allocations.clear()
            self._modules.clear()
            self._cache.clear()
        if error is not None and exc is None:
            raise error
