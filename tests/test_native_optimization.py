"""Numerical checks with the production allocator reusing real CPU byte storage.

Triton's interpreter executes kernels; only cuBLAS is replaced by Torch matmul.
This catches lifetime/aliasing bugs that metadata-only pool accounting cannot.
"""
import ctypes as ct
import math
import os

import datasets  # Windows DLL ordering.
import numpy as np
import pytest
import torch

from model.dense_cublas.autograd import Engine
from model.dense_cublas.memory import TrainingBackend
from model.dense_cublas.model import Model
from model.dense_cublas.optim import Optimizer
from model.dense_triton.runtime import Tensor, Scalar, ITEMSIZE
from tests.test_dense_cublas import ScheduleOracle
from tests.test_dense_triton import ReferenceBackend, DTYPES
from tests.test_cublas_training import HostBackend
from tests.test_legacy_resume import CONFIG

pytestmark = pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='CPU interpreter only')


class ReusingBackend(TrainingBackend):
    def __init__(self):
        super().__init__(pool_limit_mib=256, pin_memory=False)
        self.arch = 120
        self.main_stream = self.weight_stream = self.stream = 1
        self.buffers = {}
        self.syncs = 0
        def malloc(out, size):
            storage = torch.empty(size, dtype=torch.uint8)
            pointer = storage.data_ptr()
            self.buffers[pointer] = storage
            ct.cast(out, ct.POINTER(ct.c_uint64))[0] = pointer
        self._malloc = malloc
        self._free = lambda pointer: self.buffers.pop(pointer)

    def _check(self): pass
    def synchronize(self): self.syncs += 1

    def cpu(self, tensor):
        if isinstance(tensor, Scalar): return tensor
        size=tensor.numel*ITEMSIZE[tensor.dtype]
        if tensor.storage in self.buffers:base=tensor.storage
        else:base=next(p for p,buf in self.buffers.items() if p<=tensor.storage and tensor.storage+size<=p+buf.numel())
        offset=tensor.storage-base
        storage = self.buffers[base][offset:offset+size].view(DTYPES[tensor.dtype]).view(tensor.shape)
        return Tensor(tensor.shape, tensor.dtype, storage, self)

    def launch(self, fn, grid, args, **constants):
        self._submit('triton:' + fn.__name__,
            lambda: ReferenceBackend.launch(self, fn, grid, tuple(self.cpu(x) for x in args), **constants))

    def matmul(self, a, b, out, sa, sb, m, n, k, **options):
        ScheduleOracle.matmul(self, self.cpu(a), self.cpu(b), self.cpu(out),
            None if sa is None else self.cpu(sa), None if sb is None else self.cpu(sb), m, n, k, **options)

    def from_numpy(self, array):
        array = np.ascontiguousarray(array)
        dtype = {'float32':'fp32', 'int64':'i64', 'int32':'i32', 'float16':'fp16', 'uint16':'u16'}[str(array.dtype)]
        with self.host_constants(): result = self.empty(array.shape, dtype)
        self.cpu(result).storage.copy_(torch.from_numpy(array.copy()))
        return result

    def to_numpy(self, tensor):
        storage = self.cpu(tensor).storage
        if tensor.dtype == 'bf16' or tensor.dtype.startswith('fp8'): storage = storage.float()
        return storage.numpy().copy()

    def copy_many_from_numpy(self, tensors, arrays):
        for tensor, array in zip(tensors, arrays):
            self.cpu(tensor).storage.copy_(torch.from_numpy(np.asarray(array).copy()))

    def to_numpy_many(self, tensors): return tuple(self.to_numpy(t) for t in tensors)


@pytest.mark.parametrize('maximum',[448.,57344.])
@pytest.mark.parametrize('amax',[0.,1e-13,.12345,1.,3.1415926,10000.])
def test_scale_pair_matches_rounded_reciprocal_and_survives_pool_reuse(maximum,amax):
    from model.dense_cublas.quantization import make_scale
    b=ReusingBackend();e=Engine(b)
    partial=e.array(np.asarray([amax],np.float32)).data
    mark=b.mark();scale=make_scale(e.ops,partial,maximum);inverse=e.ops._inverse(scale)
    assert scale.shape==inverse.shape==(1,)
    expected=np.float32(np.float64(maximum)/max(np.float64(np.float32(amax)),1e-12))
    np.testing.assert_array_equal(b.to_numpy(scale),[expected])
    np.testing.assert_array_equal(b.to_numpy(inverse),[np.float32(1.)/expected])
    b.release_since(mark,keep=(scale,))
    other=b.empty((8,),'fp32')
    assert other.storage!=scale.storage
    assert scale.storage%16==0 and inverse.storage%16==0
    np.testing.assert_array_equal(b.to_numpy(inverse),[np.float32(1.)/expected])


@pytest.mark.parametrize('branches,deferred',[(2,False),(3,False),(4,False),(1,True)])
@pytest.mark.parametrize('dtype',['fp32','bf16'])
def test_fused_norm_gradient_sum_preserves_recycled_storage(branches,deferred,dtype):
    from model.dense_cublas.autograd import GradientSum
    results=[]
    for enabled in (False,True):
        b=ReusingBackend();b.compact_gradients=True;b.fused_norm_backward=enabled
        e=Engine(b);rng=np.random.default_rng(931)
        x=e.array(rng.normal(size=(2,7,32)).astype(np.float32),dtype,True)
        weight=e.array(rng.normal(size=(32,)).astype(np.float32),dtype,True)
        y=e.norm(x,weight,out_dtype=dtype)
        loss=e.full((1,),0.)
        for index in range(branches):
            if deferred:
                data=e.binary(y,2.,'mul').data
                z=e._node(data,(y,),lambda g:(GradientSum((g,g)),))
            else:z=y*(index+.25)
            loss=loss+(z*z).sum()
        grads=e.vjp(loss)
        results.append((e.item(loss),e.numpy(grads[x]),e.numpy(grads[weight])))
        assert b.allocator_stats['pool_hits']>0
    for actual,expected in zip(results[1],results[0]):
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('checkpoint', [0, 1, 2])
@pytest.mark.parametrize('moe', [False, True])
def test_recycled_backward_matches_unpooled_with_shared_parameters(checkpoint, moe):
    cfg = CONFIG | dict(use_moe=moe, num_experts=3, num_experts_per_tok=2,
                        moe_intermediate_size=12, use_grad_checkpoint=checkpoint)
    e, reference = Engine(ReusingBackend()), Engine(HostBackend())
    model, other = (Model(engine, cfg, activation_dtype='fp32', seed=5) for engine in (e, reference))
    opt, oracle = (Optimizer(engine, m.parameters(), lr=.002, grad_clip=.3)
                   for engine, m in ((e, model), (reference, other)))
    for step in range(3):
        mark = e.backend.mark()
        ids = np.asarray([[1, 2, 3, 4], [3, 2, 4, 5]], np.int64) + step % 2
        for m, optimizer in ((model, opt), (other, oracle)):
            out = m(ids, ids, return_logits=False)
            optimizer.backward(out.loss + out.aux_loss)
        for p in opt.parameters:
            np.testing.assert_allclose(e.numpy(opt.gradients[p]), reference.numpy(oracle.gradients[other.params[p.name]]),
                                       atol=4e-5, rtol=2e-3, err_msg=p.name)
        if step % 2 or step == 2:
            result, expected = opt.step(), oracle.step()
            assert math.isclose(result['grad_norm'], expected['grad_norm'], rel_tol=2e-4)
        for p in opt.parameters:
            np.testing.assert_allclose(e.numpy(p), reference.numpy(other.params[p.name]), atol=3e-5, rtol=2e-3)
        e.backend.release_since(mark)
    assert e.backend.allocator_stats['pool_hits'] > 0
    assert e.backend.allocator_stats['release_syncs'] == 0


@pytest.mark.parametrize('moe', [False, True])
def test_graph_replays_reused_addresses_after_input_and_routing_changes(monkeypatch, moe):
    from model.dense_cublas.backend import ExecutionPlan
    from model.dense_cublas.graph import StepGraph
    monkeypatch.setattr(ExecutionPlan, 'capture', lambda self: None)
    cfg = CONFIG | dict(use_moe=moe, num_experts=3, num_experts_per_tok=2,
                        moe_intermediate_size=12, use_grad_checkpoint=1)
    e, r = Engine(ReusingBackend()), Engine(HostBackend())
    model, ref = (Model(engine, cfg, activation_dtype='fp32', seed=31) for engine in (e, r))
    opt, oracle = (Optimizer(engine, m.parameters(), lr=.002, grad_clip=.2, loss_scaling=True)
                   for engine, m in ((e, model), (r, ref)))
    # Changing labels and token ids changes values and MoE routing, not tape shape.
    ids = np.asarray([[1, 2, 3, 4]], np.int64)
    def loss(values):
        out = model(*values, return_logits=False)
        return out.loss + out.aux_loss
    graph = StepGraph(model, opt, (ids, ids), loss, 2)
    pinned = set(e.backend._pinned)
    assert pinned and not any(p in pinned for pool in e.backend._pool.values() for p in pool)
    for step in range(4):
        batch = (ids + step % 3, ids + (step + 1) % 3)
        out = ref(*batch, return_logits=False)
        expected = out.loss + out.aux_loss
        oracle.backward(expected)
        if step % 2: oracle.step()
        actual = graph.run(batch, .002)
        assert abs(actual - r.item(expected)) < 2e-5, (step, actual, r.item(expected))
        for p in opt.parameters:
            np.testing.assert_allclose(e.numpy(p), r.numpy(ref.params[p.name]), atol=4e-5, rtol=3e-3, err_msg=p.name)
        # Unrelated eager fallback scratch must never take a tape address.
        mark = e.backend.mark()
        scratch = [e.full((4, 8), 999.) for _ in range(4)]
        assert all(x.data.storage not in pinned for x in scratch)
        e.backend.release_since(mark)
    assert graph.plan.replay_count == 4
    assert e.backend.allocator_stats['pool_hits'] > 0


@pytest.mark.parametrize('scale', [1., 1.e-20])
def test_scaled_norm_no_full_gradient_temporaries(scale):
    e = Engine(ReusingBackend())
    arrays = [np.linspace(-2., 3., n, dtype=np.float32) / scale for n in (1, 259, 2501)]
    values = [e.array(x) for x in arrays]
    actual = e.item(e.squared_norm(values, scale))
    expected = sum(np.square(x.astype(np.float64) * scale).sum() for x in arrays)
    assert math.isclose(actual, expected, rel_tol=2e-6)


def test_ce_recomputes_gradients_handles_ignored_and_large_logits():
    e = Engine(ReusingBackend())
    logits = np.array([[1000., 999., -500.], [-20., 40., 3.], [1., 2., 3.]], np.float32)
    labels = np.array([1, -100, 0], np.int64)
    x = e.array(logits, requires_grad=True)
    loss = e.ce(x, e.array(labels))
    grad = e.vjp(loss)[x]
    ref = torch.tensor(logits, requires_grad=True)
    expected = torch.nn.functional.cross_entropy(ref, torch.tensor(labels))
    expected.backward()
    assert abs(e.item(loss) - expected.item()) < 2e-5
    np.testing.assert_allclose(e.numpy(grad), ref.grad.numpy(), atol=2e-6)
    ignored = e.ce(x, e.array(np.full(3, -100, np.int64)))
    assert e.item(ignored) == 0.
    np.testing.assert_array_equal(e.numpy(e.vjp(ignored)[x]), 0.)


@pytest.mark.parametrize('dtype', ['fp32', 'bf16', 'fp16'])
@pytest.mark.parametrize('ta,tb', [(False, False), (False, True), (True, False), (True, True)])
def test_direct_gemm_all_transposes_matches_packed_legacy_without_copies(dtype, ta, tb):
    rng = np.random.default_rng(1)
    e = Engine(HostBackend())
    a = e.array(rng.normal(size=(7, 3) if ta else (3, 7)).astype(np.float32), dtype)
    b = e.array(rng.normal(size=(5, 7) if tb else (7, 5)).astype(np.float32), dtype)
    e.backend._commands = []
    direct = e.ops.mm(a.data, b.data, ta, tb, out_dtype=dtype)
    labels = [command.label for command in e.backend._commands]
    assert labels == ['cublasLt:forward']
    e.backend.direct_layout = False
    packed = e.ops.mm(a.data, b.data, ta, tb, out_dtype=dtype)
    np.testing.assert_allclose(e.numpy(direct), e.numpy(packed), atol=3e-6, rtol=2e-6)


@pytest.mark.parametrize('weights', [False, True])
def test_parallel_expert_amax_handles_empty_experts_and_masked_tail(weights):
    from model.dense_cublas import grouped as g
    rng = np.random.default_rng(7)
    e = Engine(ReusingBackend())
    rows, width, experts = 67, 129, 3
    shape = (experts, rows, width) if weights else (rows, width)
    x = e.array(rng.normal(size=shape).astype(np.float32))
    offsets = e.array(np.array([0, 0, 13, rows], np.int64))
    tiles = math.ceil(rows * width / 1024)
    partials = e.full((experts, tiles), -123.)
    result, old = e.full((experts,), 0.), e.full((experts,), 0.)
    options = dict(WIDTH=width, ROWS=rows, WEIGHTS=weights)
    e.backend.launch(g.expert_amax_partials, (tiles, experts), (x.data, offsets.data, partials.data), TILES=tiles, **options)
    e.backend.launch(g.expert_amax_finish, (experts,), (partials.data, result.data), TILES=tiles, LIMIT=448.)
    e.backend.launch(g.expert_scale, (experts,), (x.data, offsets.data, old.data), LIMIT=448., **options)
    np.testing.assert_allclose(e.numpy(result), e.numpy(old), atol=0., rtol=0.)


@pytest.mark.parametrize('dtype', ['fp16'])
def test_wide_grouped_tiles_keep_fp32_weight_gradients(dtype):
    from model.dense_cublas.grouped import linear
    rng = np.random.default_rng(13)
    e = Engine(ReusingBackend())
    x = e.array(rng.normal(size=(67, 64)).astype(np.float32) * .1, dtype, True)
    masters = [e.array(rng.normal(size=(1, 64, 64)).astype(np.float32) * .1, requires_grad=True) for _ in range(3)]
    bank = e.cat(masters, dtype=dtype)
    offsets = e.array(np.array([0, 0, 25, 67], np.int64))
    result = linear(e, x, bank, offsets.data, 67, weight_dtype='fp32')
    gradient = e.vjp(result.sum())
    xx = torch.tensor(e.numpy(x).astype(np.float32), requires_grad=True)
    ww = torch.tensor(np.concatenate([e.numpy(w) for w in masters]), requires_grad=True)
    cast = ww.to(DTYPES[dtype]).float()
    ref = torch.cat([xx[:0] @ cast[0].T, xx[:25] @ cast[1].T, xx[25:] @ cast[2].T])
    ref.sum().backward()
    np.testing.assert_allclose(e.numpy(result), ref.detach().to(DTYPES[dtype]).float().numpy(), atol=2e-3, rtol=.02)
    np.testing.assert_allclose(e.numpy(gradient[x]), xx.grad.numpy(), atol=.01, rtol=.03)
    for i, p in enumerate(masters):
        assert gradient[p].dtype == 'fp32'
        # Unlike a BF16 master, native FP32 masters retain FP32 dW accumulation.
        begin, end = ((0, 0), (0, 25), (25, 67))[i]
        expected = torch.ones((end - begin, 64)).T @ xx.detach()[begin:end]
        np.testing.assert_allclose(e.numpy(gradient[p])[0], expected.numpy(), atol=2e-5, rtol=3e-4)


def test_host_upload_never_overwrites_pending_pooled_storage():
    b = ReusingBackend()
    mark = b.mark()
    earlier = b.empty((32,), 'fp32')
    b.release_since(mark)
    uploaded = b.from_numpy(np.ones(32, np.float32))
    assert uploaded.storage != earlier.storage
    # Same-stream device work can still reuse the allocation without a fence.
    next_device = b.empty((32,), 'fp32')
    assert next_device.storage == earlier.storage
    assert b.syncs == 0
