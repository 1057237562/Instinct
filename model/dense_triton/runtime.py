"""Small CUDA Driver API launcher: Triton compilation without importing torch.

Single device, single stream, explicit context lifetime. Allocations are owned
until context exit (an arena); callers should reuse buffers for training loops.
This experimental launcher pins Triton 3.4's ABI and rejects scratch/cluster
kernels instead of silently launching an incompatible binary.
"""
import ctypes as ct
from dataclasses import dataclass
import inspect
from functools import lru_cache
import math
import os

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


ITEMSIZE = {'fp32': 4, 'bf16': 2, 'fp16': 2, 'fp8e4nv': 1, 'fp8e5': 1,
            'i32': 4, 'i64': 8, 'u16': 2}


@dataclass(frozen=True)
class Scalar:
    """A runtime scalar kernel argument; changing its value does not recompile."""
    value: object
    kind: str = 'fp32'

    @property
    def dtype(self):
        return 'scalar:' + self.kind


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
        result=Tensor(tuple(shape), self.dtype, self.storage, self.owner)
        if hasattr(self,'amax_partial'):result.amax_partial=self.amax_partial
        return result


@lru_cache(maxsize=None)
def _constant_defaults(fn):
    return {name: p.default for name, p in inspect.signature(fn.fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


def constants_for(fn, values):
    """Fill constexpr defaults without initializing Triton's default driver."""
    result = dict(_constant_defaults(fn))
    result.update(values)
    return result


def compile_kernel(fn, dtypes, constants, arch=89, compiler_options=None, alignments=None):
    if not triton.__version__.startswith('3.4.'):
        raise RuntimeError('This experimental direct launcher requires Triton 3.4.x')
    constants = constants_for(fn, constants)
    names = [name for name in fn.arg_names if name not in constants]
    if len(names) != len(dtypes):
        raise ValueError(f'{fn.__name__}: runtime argument count does not match signature')
    attrs = {} if alignments is None else {
        (fn.arg_names.index(name),): [['tt.divisibility', alignment]]
        for name, alignment in zip(names, alignments) if alignment > 1}
    source = ASTSource(fn, signature={n: t.removeprefix('scalar:') if t.startswith('scalar:') else '*' + t for n, t in zip(names, dtypes)},
                       constexprs=constants, attrs=attrs)
    # Explicit target avoids GPUDriver.__init__, which otherwise imports torch.
    return triton.compile(source, target=GPUTarget('cuda', arch, 32),
                           options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False,
                                    **(compiler_options or {})})


class CudaBackend:
    """Use as a context manager. Not thread-safe; no implicit allocator cache."""
    def __init__(self, device=0):
        self.device = device
        self._allocations = []
        self._modules = []
        self._cache = {}
        self._argument_buffers = {}
        self.kernel_options = {}
        self.aligned_pointers = False
        self._context = None
        self.pin_memory = False
        self._host_pointer, self._host_capacity = None, 0
        self.stream = None  # Optional explicit stream; default behavior stays unchanged.

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
                   np.dtype('int32'): 'i32', np.dtype('int64'): 'i64', np.dtype('uint16'): 'u16'}
        if values.dtype not in mapping:
            raise ValueError('upload accepts float32/float16/int32/int64; use a cast kernel for BF16')
        result = self.empty(values.shape, mapping[values.dtype])
        if self.pin_memory and values.nbytes <= 16 * 1024 * 1024:
            # Bound host-locked memory; unusually large weight uploads stay pageable.
            # Pinned staging is owned by this CUDA context and reused only after
            # the synchronous HtoD completes. CPU workers can prefetch meanwhile.
            if self._host_pointer is None or self._host_capacity < values.nbytes:
                allocate = self._bind('cuMemHostAlloc', [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_uint])
                self._host_free = self._bind('cuMemFreeHost', [ct.c_void_p])
                pointer = ct.c_void_p()
                allocate(ct.byref(pointer), values.nbytes, 0)
                if self._host_pointer is not None: self._host_free(self._host_pointer)
                self._host_pointer, self._host_capacity = pointer, values.nbytes
            ct.memmove(self._host_pointer, values.ctypes.data, values.nbytes)
            self._htod(result.storage, self._host_pointer, values.nbytes)
        else:
            self._htod(result.storage, values.ctypes.data, values.nbytes)
        return result

    def to_numpy(self, tensor):
        import numpy as np
        self._check_tensor(tensor)
        dtype = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'uint16',
                 'fp8e4nv': 'uint8', 'fp8e5': 'uint8', 'i32': 'int32', 'i64': 'int64', 'u16': 'uint16'}[tensor.dtype]
        result = np.empty(tensor.shape, dtype=dtype)
        self.synchronize()
        self._dtoh(result.ctypes.data, tensor.storage, result.nbytes)
        if tensor.dtype == 'bf16':
            return (result.astype(np.uint32) << 16).view(np.float32)
        # FP8 is returned as raw bytes, avoiding a dependency on a framework dtype.
        return result

    def _check_tensor(self, tensor, *, check_context=True):
        if check_context:self._check()
        if tensor.owner is not self:
            raise ValueError('tensor belongs to a different backend/context')

    def launch(self, fn, grid, args, **constants):
        if not 1 <= len(grid) <= 3 or any(g <= 0 for g in grid):
            raise ValueError('launch grid must have 1..3 positive dimensions')
        self._check()
        for arg in args:
            if not isinstance(arg, Scalar):
                self._check_tensor(arg,check_context=False)
        constants = constants_for(fn, constants)
        compiler_options = self.kernel_options.get(fn.fn.__module__ + '.' + fn.__name__, {})
        # Native tensors may be manually constructed as offset views. Specialize
        # on the actual address, rather than assuming every pointer is aligned.
        align=self.aligned_pointers
        if align and getattr(self,'aligned_pointwise_only',False):
            # Sum/mean reduction layouts can change BF16 rounding downstream.
            # Restrict vectorized specialization to elementwise/max/layout ops.
            align=fn.__name__ in {'cast','add','binary','binary_scalar','gradient_sum','scaled_copy',
                'fill','pack_matrix','pack_matrix_tiled','unpack_matrix','inverse_scale',
                'amax_partial','scale_from_amax','quantize','quantize_dual',
                'rounded_swiglu','rounded_swiglu_amax','statistics','quantized_values'}
        alignments = tuple(1 if isinstance(arg,Scalar) else min(16,arg.storage & -arg.storage)
                           for arg in args) if align else None
        key = (fn.__name__, tuple(a.dtype for a in args), tuple(sorted(constants.items())),
               tuple(sorted(compiler_options.items())), alignments)
        if key not in self._cache:
            kernel = compile_kernel(fn, [a.dtype for a in args], constants, self.arch, compiler_options, alignments)
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
        # cuLaunchKernel copies its parameter values before returning. This
        # backend is single-threaded, so reuse host ABI slots across launches.
        signature=tuple(a.dtype if isinstance(a,Scalar) else 'pointer' for a in args)
        buffers=self._argument_buffers.get(signature)
        if buffers is None:
            types={'scalar:fp32':ct.c_float,'scalar:i32':ct.c_int32,'scalar:i64':ct.c_int64,'pointer':ct.c_uint64}
            values=[types[k]() for k in signature]+[ct.c_uint64(0)]
            params=(ct.c_void_p*len(values))(*[ct.addressof(v) for v in values])
            buffers=self._argument_buffers[signature]=(values,params)
        values,params=buffers
        for target,arg in zip(values,args):target.value=arg.value if isinstance(arg,Scalar) else arg.storage
        grid = tuple(grid) + (1,) * (3 - len(grid))
        self._launch(function, *grid, meta.num_warps * 32, 1, 1, meta.shared, self.stream, params, None)

    def synchronize(self):
        self._check()
        self._sync()

    def memory(self):
        self.synchronize()
        free, total = ct.c_size_t(), ct.c_size_t()
        self._meminfo(ct.byref(free), ct.byref(total))
        return {'arena_bytes': sum(n for _, n in self._allocations),
                'host_pinned_bytes': self._host_capacity,
                'device_free_bytes': free.value, 'device_total_bytes': total.value}

    def __exit__(self, exc_type, exc, traceback):
        # Always release the context even if synchronization/cleanup reports an error.
        error = None
        try:
            self.synchronize()
        except Exception as failure:
            error = failure
        if self._host_pointer is not None:
            try: self._host_free(self._host_pointer)
            except Exception as failure: error = error or failure
            self._host_pointer, self._host_capacity = None, 0
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
            self._argument_buffers.clear()
        if error is not None and exc is None:
            raise error
