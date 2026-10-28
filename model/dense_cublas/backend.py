"""cuBLASLt plans, per-stream workspaces, event dependencies and graph replay."""
from contextlib import contextmanager
import ctypes as ct
from dataclasses import dataclass
import statistics

from model.dense_triton.runtime import CudaBackend
from .bridge import NativeLt, LtError


@dataclass(frozen=True)
class ScheduleConfig:
    workspace_mib: int = 32  # Per stream, not shared by concurrent GEMMs.
    candidates: int = 8
    tune: bool = False
    overlap_dw: bool = False
    dw_sm_target: int = 0  # Heuristic hint, not an SM reservation.
    tune_repeats: int = 5
    fp8_fast_accum_output: bool = False

    def __post_init__(self):
        if self.workspace_mib < 0 or not 1 <= self.candidates <= 64 or self.dw_sm_target < 0 or self.tune_repeats < 1:
            raise ValueError('invalid cuBLASLt scheduling limits')


@dataclass
class Command:
    label: str
    lane: str
    execute: object


class ExecutionPlan:
    def __init__(self, backend, commands, outputs):
        self.backend, self.commands, self.outputs = backend, tuple(commands), outputs
        self.graph, self.graph_exec = None, None
        self.capture_count,self.replay_count=0,0

    def replay(self):
        self.backend._check()
        self.replay_count+=1
        if self.graph_exec:
            self.backend._graph_launch(self.graph_exec, self.backend.main_stream)
        else:
            for command in self.commands:
                command.execute()
        return self.outputs

    def capture(self):
        """After preparation/tuning, capture the already allocated complete DAG."""
        b = self.backend
        b._check()
        if self.graph_exec:
            raise RuntimeError('plan is already captured')
        b.synchronize()
        graph, executable = ct.c_void_p(), ct.c_void_p()
        b._capture_begin(b.main_stream, 0)  # CU_STREAM_CAPTURE_MODE_GLOBAL
        try:
            for command in self.commands:
                command.execute()
        except Exception:
            try:
                b._capture_end(b.main_stream, ct.byref(graph))
            except Exception:
                pass  # Invalidated captures can fail at end; preserve original error.
            if graph:
                b._graph_destroy(graph)
            raise
        b._capture_end(b.main_stream, ct.byref(graph))
        try:
            b._graph_instantiate(ct.byref(executable), graph, 0)
        except Exception:
            b._graph_destroy(graph)
            raise
        self.graph, self.graph_exec = graph, executable
        self.capture_count+=1

    def describe(self):
        return [{'index': i, 'lane': c.lane, 'operation': c.label} for i, c in enumerate(self.commands)]


class CublasBackend(CudaBackend):
    """One prepared fixed-shape block step per context; buffers remain stable.

    Main stream carries forward and dX. Optional low-priority stream carries dW.
    An end-of-step event joins dW back into main before any next replay/update.
    """
    def __init__(self, device=0, config=ScheduleConfig(), library=None):
        super().__init__(device)
        self.config, self.library = config, library
        self.bounded_tuning=True
        self.native = None
        self.direct_layout = True
        self.main_stream = self.weight_stream = None
        self._events, self._streams, self._plans = [], [], {}
        self._commands = None
        self.execution_plan = None
        self.workspaces = {}

    def __enter__(self):
        if getattr(self, '_used', False):
            raise RuntimeError('create a fresh CublasBackend for each CUDA context lifetime')
        self._used = True
        super().__enter__()
        p, i = ct.c_void_p, ct.c_int
        try:
            self._stream_create = self._bind('cuStreamCreateWithPriority', [ct.POINTER(p), ct.c_uint, i])
            self._stream_destroy = self._bind('cuStreamDestroy_v2', [p])
            self._event_create = self._bind('cuEventCreate', [ct.POINTER(p), ct.c_uint])
            self._event_record = self._bind('cuEventRecord', [p, p])
            self._event_destroy = self._bind('cuEventDestroy_v2', [p])
            self._event_sync = self._bind('cuEventSynchronize', [p])
            self._elapsed = self._bind('cuEventElapsedTime', [ct.POINTER(ct.c_float), p, p])
            self._stream_wait = self._bind('cuStreamWaitEvent', [p, p, ct.c_uint])
            self._capture_begin = self._bind('cuStreamBeginCapture_v2', [p, i])
            self._capture_end = self._bind('cuStreamEndCapture', [p, ct.POINTER(p)])
            self._graph_instantiate = self._bind('cuGraphInstantiateWithFlags', [ct.POINTER(p), p, ct.c_uint64])
            self._graph_launch = self._bind('cuGraphLaunch', [p, p])
            self._graph_destroy = self._bind('cuGraphDestroy', [p])
            self._graph_exec_destroy = self._bind('cuGraphExecDestroy', [p])
            self._memset_async = self._bind('cuMemsetD8Async', [ct.c_uint64, ct.c_ubyte, ct.c_size_t, p])
            least, greatest = i(), i()
            self._bind('cuCtxGetStreamPriorityRange', [ct.POINTER(i), ct.POINTER(i)])(ct.byref(least), ct.byref(greatest))
            self.main_stream = self._make_stream(greatest.value)
            self.weight_stream = self._make_stream(least.value) if self.config.overlap_dw else self.main_stream
            self.stream = self.main_stream
            sm_count = i()
            self._bind('cuDeviceGetAttribute', [ct.POINTER(i), i, i])(ct.byref(sm_count), 16, self._dev)
            if self.config.dw_sm_target > sm_count.value:
                raise ValueError('dw_sm_target exceeds the physical GPU SM count')
            self.native = NativeLt(self.library)
            self.direct_layout = self.native.abi >= 2
            size = self.config.workspace_mib * 1024 * 1024
            for stream in set((self.main_stream, self.weight_stream)):
                self.workspaces[stream] = super().empty((size // 4,), 'fp32') if size else None
        except Exception as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def _make_stream(self, priority):
        stream = ct.c_void_p()
        self._stream_create(ct.byref(stream), 1, priority)  # NON_BLOCKING
        self._streams.append(stream.value)
        return stream.value

    @contextmanager
    def on_stream(self, stream):
        prior, self.stream = self.stream, stream
        try:
            yield
        finally:
            self.stream = prior

    def _submit(self, label, fn, stream=None):
        stream = self.stream if stream is None else stream
        fn()
        if self._commands is not None:
            lane = 'main' if stream == self.main_stream else 'dW'
            self._commands.append(Command(label, lane, fn))

    def launch(self, fn, grid, args, **constants):
        block=getattr(self,'elementwise_block',None)
        if block and len(grid)==1:
            module,name=fn.fn.__module__,fn.__name__
            size_key=block_key=None
            if module=='model.dense_cublas.tensor_kernels' and name in (
                    'fill','masked_fill','binary','binary_scalar','scalar_extreme_backward',
                    'unary','unary_backward','strided_copy','dropout','scaled_copy'):
                size_key,block_key='N','BLOCK'
            elif module=='model.dense_triton.elementwise' and name in ('cast','add','_swiglu_forward','_swiglu_backward'):
                size_key,block_key='SIZE','BLOCK'
            elif module=='model.dense_triton.linear' and name=='quantize':
                size_key,block_key='SIZE','BLOCK'
            elif module=='model.dense_cublas.fused' and name=='rounded_swiglu':
                size_key,block_key='N','B'
            if size_key and constants[size_key]>=65536:
                constants[block_key]=block
                grid=((constants[size_key]+block-1)//block,)
        stream = self.stream
        def execute():
            with self.on_stream(stream):
                super(CublasBackend, self).launch(fn, grid, args, **constants)
        self._submit('triton:' + fn.__name__, execute)

    def empty(self, shape, dtype='fp32', zero=False):
        out = super().empty(shape, dtype, zero=False)
        if zero:
            from model.dense_triton.runtime import ITEMSIZE
            stream, size = self.stream, out.numel * ITEMSIZE[dtype]
            self._submit('zero', lambda: self._memset_async(out.storage, 0, size, stream))
        return out

    def from_numpy(self, values):
        # Constants uploaded during preparation retain fixed storage; dynamic
        # batch inputs must be supplied as Values by the full-step graph runner.
        result = super().from_numpy(values)
        # Uploads use the synchronous Driver API, whereas compute streams are
        # NON_BLOCKING. Explicitly finish host transfers before compute reads.
        self.synchronize()
        return result

    def copy_from_numpy(self, tensor, values):
        """Update a fixed-address input between replays, never during capture.

        BF16/FP8 inputs use raw uint16/uint8 bits; this method does not quantize.
        """
        self.copy_many_from_numpy((tensor,),(values,))

    def copy_many_from_numpy(self, tensors, arrays):
        """One pair of transfer fences for the whole replay input/control batch."""
        import numpy as np
        self._check()
        if self._commands is not None:
            raise RuntimeError('host updates cannot be recorded in a prepared plan')
        tensors,arrays=tuple(tensors),tuple(arrays)
        if len(tensors)!=len(arrays):raise ValueError('upload batch lengths differ')
        prepared=[]
        for tensor,values in zip(tensors,arrays):
            self._check_tensor(tensor,check_context=False)
            values=np.ascontiguousarray(values)
            dtype={'fp32':'float32','fp16':'float16','bf16':'uint16',
                   'fp8e4nv':'uint8','fp8e5':'uint8','i32':'int32','i64':'int64'}[tensor.dtype]
            if values.shape!=tensor.shape or values.dtype!=np.dtype(dtype):
                raise ValueError(f'upload requires shape {tensor.shape} and dtype {dtype}')
            prepared.append((tensor,values))
        if not prepared:return
        self.synchronize()
        for tensor,values in prepared:
            if hasattr(tensor,'quant_scale_valid'):tensor.quant_scale_valid=False
            self._htod(tensor.storage,values.ctypes.data,values.nbytes)
        self.synchronize()

    def to_numpy_many(self,tensors):
        """Download replay statistics/loss after one fence, preserving dtypes."""
        import numpy as np
        tensors=tuple(tensors)
        self.synchronize();results=[]
        for tensor in tensors:
            self._check_tensor(tensor,check_context=False)
            dtype={'fp32':'float32','fp16':'float16','bf16':'uint16',
                   'fp8e4nv':'uint8','fp8e5':'uint8','i32':'int32','i64':'int64'}[tensor.dtype]
            array=np.empty(tensor.shape,dtype=dtype)
            self._dtoh(array.ctypes.data,tensor.storage,array.nbytes)
            if tensor.dtype=='bf16':array=(array.astype(np.uint32)<<16).view(np.float32)
            results.append(array)
        return tuple(results)

    def create_event(self, timing=False):
        event = ct.c_void_p()
        self._event_create(ct.byref(event), 0 if timing else 2)
        self._events.append(event.value)
        return event.value

    def record_event(self, event):
        stream = self.stream
        self._submit('event:record', lambda: self._event_record(event, stream))

    def wait_event(self, event):
        stream = self.stream
        self._submit('event:wait', lambda: self._stream_wait(stream, event, 0))

    def finish_backward(self):
        if self.weight_stream != self.main_stream:
            event = self.create_event()
            with self.on_stream(self.weight_stream):
                self.record_event(event)
            with self.on_stream(self.main_stream):
                self.wait_event(event)

    def build(self, callback):
        """Prepare once, returning a replayable step with no replay allocations."""
        if self.execution_plan is not None or self._commands is not None:
            raise RuntimeError('one execution plan per backend context; create a new context for a new shape')
        self.synchronize()
        self._commands = []
        try:
            with self.on_stream(self.main_stream):
                outputs = callback()
                self.finish_backward()
            self.synchronize()
            plan = ExecutionPlan(self, self._commands, outputs)
        finally:
            self._commands = None
        self.execution_plan = plan
        if hasattr(self,'seal_plan_allocations'):self.seal_plan_allocations()
        return plan

    def time_call(self, callback, stream, repeats=1):
        # Direct timing calls are deliberately excluded from the replay trace.
        start, end = ct.c_void_p(), ct.c_void_p()
        self._event_create(ct.byref(start), 0)
        try:
            self._event_create(ct.byref(end), 0)
            self._event_record(start, stream)
            for _ in range(repeats):
                callback()
            self._event_record(end, stream)
            self._event_sync(end)
            milliseconds = ct.c_float()
            self._elapsed(ct.byref(milliseconds), start, end)
            return milliseconds.value / repeats
        finally:
            if end:
                self._event_destroy(end)
            self._event_destroy(start)

    def matmul(self, a, b, out, scale_a, scale_b, m, n, k, role='gemm',
               transpose_a=True, transpose_b=False):
        self._check()
        ashape = (m, k) if transpose_a else (k, m)
        bshape = (k, n) if transpose_b else (n, k)
        if a.shape != ashape or b.shape != bshape or out.shape != (n, m):
            raise ValueError('GEMM buffers do not match the planned dimensions/transposes')
        if (a.dtype.startswith('fp8') or b.dtype.startswith('fp8')) and (not transpose_a or transpose_b):
            raise ValueError('FP8 requires column-major TN')
        for tensor in (a, b, out):
            self._check_tensor(tensor,check_context=False)
            if tensor.storage % 256:
                raise ValueError('cuBLASLt plans require base allocations aligned to 256 bytes')
        for tensor in (scale_a, scale_b):
            if tensor is not None:
                self._check_tensor(tensor,check_context=False)
                if tensor.dtype != 'fp32' or tensor.numel != 1:
                    raise ValueError('cuBLASLt tensorwise scale must be a device FP32 scalar')
        stream = self.stream
        is_dw = stream == self.weight_stream and stream != self.main_stream
        hint = self.config.dw_sm_target if is_dw else 0
        size = self.config.workspace_mib * 1024 * 1024
        validated_large=(getattr(self,'fp8_fast_accum_large',False) and self.arch==120 and (m,n,k) in (
            (1536,12288,1536),(5504,12288,1536),(1536,12288,5504),(6400,12288,1536)))
        fast_accum=bool((self.config.fp8_fast_accum_output or validated_large) and role=='forward' and a.dtype.startswith('fp8'))
        key = (m, n, k, a.dtype, b.dtype, out.dtype, size, hint, stream, transpose_a, transpose_b, fast_accum)
        workspace = self.workspaces[stream]
        pointers = (a.storage, b.storage, out.storage,
                    None if scale_a is None else scale_a.storage,
                    None if scale_b is None else scale_b.storage,
                    None if workspace is None else workspace.storage, size, stream)
        if key not in self._plans:
            extra={'fast_accum':True} if fast_accum else {}
            plan, candidates = self.native.create_plan(m, n, k, a.dtype, b.dtype, out.dtype,
                                                       size, self.config.candidates, hint, transpose_a, transpose_b,**extra)
            # Drain competing work so one-off tuning isn't biased by outstanding dW.
            # All this happens during build, never inside capture/replay.
            self.synchronize()
            winners = []
            bounded=self.config.tune and getattr(self,'bounded_tuning',False)
            if bounded:
                from .tuning import gemm_shortlist,race
                def measure(candidate,repeats):
                    try:
                        self.native.run(plan,candidate['index'],*pointers)
                        return self.time_call(lambda:self.native.run(plan,candidate['index'],*pointers),stream,repeats)
                    except LtError as error:
                        candidate['rejected']=str(error);return float('inf')
                best=race(gemm_shortlist(candidates),measure,baseline=candidates[0]['index'])
                winners=[best]
            for candidate in (() if bounded else candidates):
                index = candidate['index']
                try:
                    self.native.run(plan, index, *pointers)
                    if self.config.tune:
                        timings = [self.time_call(lambda i=index: self.native.run(plan, i, *pointers),
                                                  stream, self.config.tune_repeats) for _ in range(3)]
                        candidate['median_ms'] = statistics.median(timings)
                    else:
                        self.synchronize()
                    winners.append(candidate)
                    if not self.config.tune:
                        break
                except LtError as exc:
                    candidate['rejected'] = str(exc)
            if not winners:
                raise LtError(f'all cuBLASLt candidates failed for {key}')
            best = min(winners, key=lambda c: c.get('median_ms', c['index']))
            self._plans[key] = (plan, best['index'], {'shape_mnk': [m, n, k], 'types': [a.dtype, b.dtype, out.dtype],
                    'lane': 'dW' if is_dw else 'main', 'sm_target': hint,
                    'transpose_ab': [transpose_a, transpose_b],
                    'fp8_fast_accum':fast_accum,
                    'selected_index': best['index'], 'candidates': candidates})
        plan, index, _ = self._plans[key]
        self._submit('cublasLt:' + role, lambda: self.native.run(plan, index, *pointers))

    def tuning_report(self):
        return {'cublas_version': self.native.version, 'bridge_abi': self.native.abi,
                'runtime_override':getattr(self.native,'runtime_path',None),
                'direct_layout': self.direct_layout, 'workspace_bytes_per_stream': self.config.workspace_mib * 1024 * 1024,
                'streams': len(self._streams), 'plans': [record for _, _, record in self._plans.values()]}

    def batched_matmul(self,a,b,c,out,m,n,k,astrides,bstrides,alpha=1.,beta=0.):
        self._check()
        batch=a.shape[0]
        if a.dtype!=b.dtype or a.dtype!=out.dtype or (beta and c.dtype!=out.dtype):
            raise ValueError('batched GEMM requires matching high precision dtypes')
        if a.dtype not in ('bf16','fp16','fp32') or b.shape[0]!=batch or out.shape!=(batch,m,n):
            raise ValueError('invalid batched GEMM geometry')
        for tensor in (a,b,c,out):
            self._check_tensor(tensor,check_context=False)
            if tensor.storage%256:raise ValueError('batched GEMM requires aligned base allocations')
        if beta and c.shape!=(batch,m,n):raise ValueError('batched GEMM C shape mismatch')
        def transpose(strides,rows,cols):
            if strides==(rows*cols,cols,1):return False
            if strides==(rows*cols,1,rows):return True
            raise ValueError('batched GEMM only supports contiguous matrices or their transposes')
        ta,tb=transpose(bstrides,k,n),transpose(astrides,m,k)
        size=self.config.workspace_mib*2**20;stream=self.stream
        workspace=self.workspaces[stream]
        key=('batched',m,n,k,batch,a.dtype,astrides,bstrides,size,stream,bool(beta))
        pointers=(b.storage,a.storage,c.storage if beta else out.storage,out.storage,
                  float(alpha),float(beta),None if workspace is None else workspace.storage,size,stream)
        if key not in self._plans:
            plan,candidates=self.native.create_batched_plan(n,m,k,a.dtype,size,self.config.candidates,
                ta,tb,batch,(bstrides[0],astrides[0],m*n))
            self.synchronize();winners=[]
            bounded=self.config.tune and getattr(self,'bounded_tuning',False)
            if bounded:
                from .tuning import gemm_shortlist,race
                def measure(candidate,repeats):
                    try:
                        self.native.run_batched(plan,candidate['index'],*pointers)
                        return self.time_call(lambda:self.native.run_batched(plan,candidate['index'],*pointers),stream,repeats)
                    except LtError as error:
                        candidate['rejected']=str(error);return float('inf')
                best=race(gemm_shortlist(candidates),measure,baseline=candidates[0]['index'])
                winners=[best]
            for candidate in (() if bounded else candidates):
                index=candidate['index']
                try:
                    self.native.run_batched(plan,index,*pointers)
                    if self.config.tune:
                        candidate['median_ms']=statistics.median(self.time_call(
                            lambda i=index:self.native.run_batched(plan,i,*pointers),stream,self.config.tune_repeats) for _ in range(3))
                    else:self.synchronize()
                    winners.append(candidate)
                    if not self.config.tune:break
                except LtError as exc:candidate['rejected']=str(exc)
            if not winners:raise LtError('no usable batched cuBLASLt algorithm')
            best=min(winners,key=lambda x:x.get('median_ms',x['index']))
            self._plans[key]=(plan,best['index'],dict(shape_mnk=[m,n,k],batch=batch,
                types=[a.dtype]*3,lane='Muon',selected_index=best['index'],candidates=candidates))
        plan,index,_=self._plans[key]
        self._submit('cublasLt:muon',lambda:self.native.run_batched(plan,index,*pointers))

    def __exit__(self, exc_type, exc, traceback):
        error = None
        try:
            self.synchronize()
        except Exception as failure:
            error = failure
        actions = []
        if self.execution_plan:
            if self.execution_plan.graph_exec:
                actions.append(lambda: self._graph_exec_destroy(self.execution_plan.graph_exec))
            if self.execution_plan.graph:
                actions.append(lambda: self._graph_destroy(self.execution_plan.graph))
        if self.native:
            actions.append(self.native.close)
        actions.extend(lambda e=e: self._event_destroy(e) for e in reversed(self._events))
        actions.extend(lambda s=s: self._stream_destroy(s) for s in reversed(self._streams))
        for action in actions:
            try:
                action()
            except Exception as failure:
                error = error or failure
        self._events.clear()
        self._streams.clear()
        self._plans.clear()
        self.workspaces.clear()
        self.native = None
        try:
            super().__exit__(exc_type, exc or error, traceback)
        finally:
            self.execution_plan = None
        if error and exc is None:
            raise error
