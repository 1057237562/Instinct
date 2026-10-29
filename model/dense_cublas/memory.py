"""Reusable temporary allocations for dynamic training and recomputation."""
from contextlib import contextmanager
from .backend import CublasBackend
from model.dense_triton.runtime import Tensor, ITEMSIZE
from . import tensor_kernels as tk
from model.dense_triton.runtime import Scalar
import math
import ctypes as ct
import triton


class AllocationRegistry:
    """Ordered allocation ownership with O(1) removal of released addresses."""
    def __init__(self):self.entries={}
    def append(self,item):
        pointer,size=item
        if pointer.value in self.entries:raise RuntimeError('duplicate owned CUDA allocation')
        self.entries[pointer.value]=item
    def discard(self,pointer):self.entries.pop(pointer)
    def __iter__(self):return iter(self.entries.values())
    def __len__(self):return len(self.entries)
    def clear(self):self.entries.clear()


class TrainingBackend(CublasBackend):
    recycle_vjp = True
    def __init__(self, *args, pool_limit_mib=256, pin_memory=True, allocator_backend='driver', async_release_mib=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._allocations=AllocationRegistry()
        self.pin_memory = pin_memory
        if allocator_backend not in ('driver','async'):raise ValueError('unknown native allocator backend')
        self.allocator_backend=allocator_backend
        self._async_pointers=set();self._async_pool=None
        self._pool, self._live, self._serial = {}, {}, 0
        self._pool_bytes = 0
        self.pool_limit = int(pool_limit_mib * 1024 * 1024)
        if self.pool_limit < 0: raise ValueError('pool_limit_mib must be nonnegative')
        self._event_pool, self._live_events = [], {}
        self._pinned=set();self._pinned_events=set()
        self._plan_storage=set()
        self._uploading_constant=False
        self._live_bytes=self._peak_live_bytes=0
        self.allocator_stats=dict(requests=0,pool_hits=0,releases=0,empty_releases=0,release_syncs=0)
        self.async_release_mib = async_release_mib
        if async_release_mib is not None and async_release_mib < 0:
            raise ValueError('async_release_mib must be nonnegative')

    def __enter__(self):
        super().__enter__()
        if self.allocator_backend=='driver':return self
        try:
            supported=ct.c_int()
            self._bind('cuDeviceGetAttribute',[ct.POINTER(ct.c_int),ct.c_int,ct.c_int])(ct.byref(supported),115,self._dev)
            if not supported.value:raise RuntimeError('device does not support stream-ordered CUDA allocation')
            self._sync_malloc,self._sync_free=self._malloc,self._free
            allocate=self._bind('cuMemAllocAsync',[ct.POINTER(ct.c_uint64),ct.c_size_t,ct.c_void_p])
            release=self._bind('cuMemFreeAsync',[ct.c_uint64,ct.c_void_p])
            pool=ct.c_void_p()
            self._bind('cuDeviceGetDefaultMemPool',[ct.POINTER(ct.c_void_p),ct.c_int])(ct.byref(pool),self._dev)
            self._async_pool=pool
            self._pool_attribute=self._bind('cuMemPoolGetAttribute',[ct.c_void_p,ct.c_int,ct.c_void_p])
            if self.async_release_mib is not None:
                self._set_pool_attribute=self._bind('cuMemPoolSetAttribute',[ct.c_void_p,ct.c_int,ct.c_void_p])
                self._prior_release_threshold=ct.c_uint64()
                self._pool_attribute(pool,4,ct.byref(self._prior_release_threshold))
                threshold=ct.c_uint64(int(self.async_release_mib*1024*1024))
                self._set_pool_attribute(pool,4,ct.byref(threshold))
            def malloc(out,size):
                # Synchronous host uploads are outside the compute stream.
                # Keep those allocations synchronous; all device temporaries
                # use the current stream and its existing event dependencies.
                if self._uploading_constant:return self._sync_malloc(out,size)
                allocate(out,size,self.stream)
                self._async_pointers.add(ct.cast(out,ct.POINTER(ct.c_uint64))[0])
            def free(pointer):
                address=pointer.value if hasattr(pointer,'value') else pointer
                if address in self._async_pointers:
                    release(address,self.stream);self._async_pointers.remove(address)
                else:self._sync_free(pointer)
            self._malloc,self._free=malloc,free
        except Exception as exc:
            self.__exit__(type(exc),exc,exc.__traceback__)
            raise
        return self

    def __exit__(self,exc_type,exc,traceback):
        native=getattr(self,'native_attention',None)
        if native is not None:
            self.synchronize();native.close();self.native_attention=None
        if hasattr(self,'_prior_release_threshold'):
            self._set_pool_attribute(self._async_pool,4,ct.byref(self._prior_release_threshold))
        if hasattr(self,'_sync_free'):
            # cuMemFree is valid for async allocations after synchronization.
            # Restore it before CublasBackend destroys the compute streams.
            self.synchronize()
            self._malloc,self._free=self._sync_malloc,self._sync_free
        return super().__exit__(exc_type,exc,traceback)

    def empty(self, shape, dtype='fp32', zero=False):
        if self._context is None:self._check()
        shape = tuple(shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape) or dtype not in ITEMSIZE:
            raise ValueError('expected a positive shape and supported dtype')
        if dtype.startswith('fp8') and self.arch < 89:
            raise RuntimeError('Native FP8 requires compute capability >= 8.9')
        size = math.prod(shape) * ITEMSIZE[dtype]
        constant=self._uploading_constant and self._commands is not None
        # Synchronous HtoD does not order itself after non-blocking compute
        # streams. A host upload must use a fresh address, otherwise it could
        # overwrite pooled storage while an earlier main-stream reader runs.
        # Fixed replay inputs use copy_many_from_numpy with explicit fences.
        pool = None if self._uploading_constant else self._pool.get(size)
        if pool:
            # Reusing a host-side ownership record performs no CUDA operation.
            # Actual allocation, initialization and consumers bind their context.
            # Avoid a redundant cuCtxSetCurrent for every temporary tensor.
            tensor = Tensor(shape, dtype, pool.pop(), self)
            self._pool_bytes -= size
            self.allocator_stats['pool_hits']+=1
        else:
            tensor = super().empty(shape, dtype, zero=False)
        self._serial += 1
        self._live[tensor.storage] = (self._serial, size)
        self._live_bytes+=size
        self._peak_live_bytes=max(self._peak_live_bytes,self._live_bytes)
        self.allocator_stats['requests']+=1
        if self._commands is not None:self._plan_storage.add(tensor.storage)
        # Host constants are uploaded only during preparation, never replayed.
        # They cannot alias earlier OR later tape scratch: earlier commands
        # would overwrite the supposedly immutable value on the next replay.
        if constant:self._pinned.add(tensor.storage)
        if zero:
            self.launch(tk.fill, (triton.cdiv(tensor.numel, 256),), (tensor, Scalar(0.)), N=tensor.numel)
        return tensor

    @contextmanager
    def host_constants(self):
        previous,self._uploading_constant=self._uploading_constant,True
        try:yield
        finally:self._uploading_constant=previous

    def from_numpy(self,values):
        with self.host_constants():return super().from_numpy(values)

    def mark(self):
        return self._serial

    def create_event(self, timing=False):
        if timing: return super().create_event(timing=True)
        event = self._event_pool.pop() if self._event_pool else super().create_event()
        self._serial += 1
        self._live_events[event] = self._serial
        return event

    def pin_since(self,mark):
        self._pinned.update(p for p,(serial,size) in self._live.items() if serial>mark)
        self._pinned_events.update(event for event,serial in self._live_events.items() if serial>mark)

    def seal_plan_allocations(self):
        """Protect every address in the tape, including scratch recycled in build.

        Within a single-stream plan the same address can serve disjoint tensor
        lifetimes. After build it must not be lent to an unrelated eager step.
        """
        self._pinned.update(self._plan_storage)
        for size,pool in self._pool.items():
            retained=[]
            for pointer in pool:
                if pointer in self._plan_storage:
                    self._serial+=1;self._live[pointer]=(self._serial,size)
                    self._pool_bytes-=size;self._live_bytes+=size
                else:retained.append(pointer)
            pool[:]=retained
        self._peak_live_bytes=max(self._peak_live_bytes,self._live_bytes)
        self._plan_storage.clear()

    def release_since(self, mark, keep=()):
        self.allocator_stats['releases']+=1
        single_stream=self.stream==self.main_stream==self.weight_stream
        # Multi-stream tapes keep their original conservative arena. Eager dW
        # overlap still fences all consumers before any buffer/event is reused.
        if self._commands is not None and not single_stream:return
        retained = {t.storage for t in keep if t is not None}|self._pinned
        # Insertion order is allocation serial order, including addresses reused
        # from the pool. A local VJP must not scan all model/optimizer buffers.
        allocations=[]
        for pointer,(serial,size) in reversed(self._live.items()):
            if serial<=mark:break
            if pointer not in retained:allocations.append((pointer,size))
        allocations.reverse()
        events=[]
        for event,serial in reversed(self._live_events.items()):
            if serial<=mark:break
            if event not in self._pinned_events:events.append(event)
        events.reverse()
        if not allocations and not events:
            self.allocator_stats['empty_releases']+=1
            return
        # Main-stream reuse is ordered after earlier readers by CUDA itself;
        # a host/context barrier at every checkpoint/parameter is unnecessary.
        if not single_stream:
            self.synchronize();self.allocator_stats['release_syncs']+=1
        for pointer,size in allocations:
            self._pool.setdefault(size, []).append(pointer)
            self._pool_bytes += size;self._live_bytes-=size
            del self._live[pointer]
        for event in events:
            self._event_pool.append(event)
            del self._live_events[event]
        self._trim_pool()

    def release_owned(self,tensors):
        """Consume private checkpoint outputs after their last queued reader.

        Callers must own complete base allocations, not arbitrary views. This
        is deliberately separate from mark-based temporary reclamation.
        """
        tensors=tuple(tensors);pointers={t.storage for t in tensors}
        if any(t.owner is not self for t in tensors):raise ValueError('foreign checkpoint allocation')
        if any(p not in self._live or p in self._pinned for p in pointers):
            raise ValueError('checkpoint release requires unpinned live base allocations')
        if self.stream!=self.main_stream or self.weight_stream!=self.main_stream:
            self.synchronize();self.allocator_stats['release_syncs']+=1
        for p in pointers:
            _,size=self._live.pop(p)
            self._live_bytes-=size;self._pool_bytes+=size
            self._pool.setdefault(size,[]).append(p)
        self._trim_pool()

    def _trim_pool(self):
        # Recorded addresses must survive capture. Their lifetimes may alias,
        # but no cuMemFree is allowed until the owning plan/context is closed.
        if self._commands is not None:return
        # Dynamic MoE token counts and decode lengths must not grow the cache
        # without a bound. Release unused buffers, never active tensors.
        for size in sorted(self._pool, reverse=True):
            pool = self._pool[size]
            while pool and self._pool_bytes > self.pool_limit:
                pointer = pool.pop()
                self._free(pointer)
                self._allocations.discard(pointer)
                self._pool_bytes -= size

    def memory(self):
        result=super().memory() | {'live_bytes': self._live_bytes,
                                  'peak_live_bytes':self._peak_live_bytes,
                                  'cached_bytes': self._pool_bytes,
                                  'cached_limit_bytes': self.pool_limit,
                                  'allocator_stats':dict(self.allocator_stats)}
        if self._async_pool is not None:
            threshold=ct.c_uint64();self._pool_attribute(self._async_pool,4,ct.byref(threshold))
            result['async_release_threshold_bytes']=threshold.value
            for name,attribute in (('async_reserved_bytes',5),('async_peak_reserved_bytes',6),
                                   ('async_used_bytes',7),('async_peak_used_bytes',8)):
                value=ct.c_uint64();self._pool_attribute(self._async_pool,attribute,ct.byref(value))
                result[name]=value.value
        return result

    @contextmanager
    def temporary(self):
        mark = self.mark()
        try:
            yield
        finally:
            self.release_since(mark)
