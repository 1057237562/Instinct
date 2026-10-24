"""Host/interpreter contract checks. The numeric GEMM oracle is NOT cuBLAS.

Native precision/concurrency/capture tests require a built shim and a real GPU.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import datasets  # Windows DLL ordering, as required by AGENTS.md.
import pytest
import torch

pytest.importorskip('triton')
from model.dense_cublas.backend import CublasBackend, ScheduleConfig
from model.dense_cublas.layout import plan_layout
from model.dense_cublas.ops import CublasDenseOps, ScheduledDenseBlock
from model.dense_triton.ops import BlockConfig
from tests.test_dense_triton import ReferenceBackend, DEVICE, close, kernel_test, random


class ScheduleOracle(CublasBackend, ReferenceBackend):
    """Execute Triton packing kernels and replace only cuBLAS with a Torch oracle."""
    def __init__(self, overlap=False):
        CublasBackend.__init__(self, config=ScheduleConfig(overlap_dw=overlap))
        self.main_stream, self.weight_stream = 1, 2 if overlap else 1
        self.stream = self.main_stream
        self.events = []
        self._recorded = set()
        self._event_id = 0

    def _check(self):
        pass

    def synchronize(self):
        pass

    def empty(self, shape, dtype='fp32', zero=False):
        tensor = ReferenceBackend.empty(self, shape, dtype, zero=False)
        self._allocations.append(tensor)
        if zero:
            self._submit('zero', lambda: tensor.storage.zero_())
        return tensor

    def launch(self, fn, grid, args, **constants):
        self._submit('triton:' + fn.__name__, lambda: ReferenceBackend.launch(self, fn, grid, args, **constants))

    def create_event(self, timing=False):
        self._event_id += 1
        return self._event_id

    def _event_record(self, event, stream):
        self._recorded.add(event)
        self.events.append(('record', event, stream))

    def _stream_wait(self, stream, event, flags):
        assert event in self._recorded
        self.events.append(('wait', event, stream))

    def matmul(self, a, b, out, sa, sb, m, n, k, role='gemm'):
        assert a.shape == (m, k) and b.shape == (n, k) and out.shape == (n, m)
        if not a.dtype.startswith('fp8'):
            assert out.dtype in (a.dtype, 'fp32')
        def execute():
            value = b.storage.view(n, k).float() @ a.storage.view(m, k).float().T
            if sa is not None:
                value *= sa.storage.float() * sb.storage.float()
            out.storage.copy_(value.to(out.storage.dtype))
        self._submit('cublasLt:' + role, execute)


def test_imports_no_torch():
    subprocess.run([sys.executable, '-c', '''
from experiments.smoke_dense_triton import RejectTorch
import sys
sys.meta_path.insert(0, RejectTorch())
from model.dense_cublas import backend, bridge, kernels, ops, layout
assert 'torch' not in sys.modules and 'torchao' not in sys.modules
'''], check=True)


def test_schedule_defaults_and_validation():
    config = ScheduleConfig()
    assert not config.overlap_dw and not config.tune and config.dw_sm_target == 0
    for kwargs in ({'workspace_mib': -1}, {'candidates': 0}, {'candidates': 65},
                   {'dw_sm_target': -1}, {'tune_repeats': 0}):
        with pytest.raises(ValueError):
            ScheduleConfig(**kwargs)


def test_layout_no_pack_for_aligned_forward_and_reject_invalid():
    layout = plan_layout((32, 48), (64, 48), transpose_b=True)
    assert not layout.a_pack and not layout.b_pack
    assert (layout.mp, layout.np, layout.kp) == (32, 64, 48)
    for a, b in (((3,), (3, 4)), ((0, 3), (3, 4)), ((2, 3), (4, 5))):
        with pytest.raises(ValueError):
            plan_layout(a, b)


def test_fixed_address_upload_checks_size_and_fences_streams():
    import numpy as np
    from model.dense_triton.runtime import Tensor
    backend = ScheduleOracle()
    trace = []
    backend.synchronize = lambda: trace.append('sync')
    backend._htod = lambda address, source, size: trace.append((address, size))
    tensor = Tensor((2, 3), 'fp32', 256, backend)
    backend.copy_from_numpy(tensor, np.ones((2, 3), dtype=np.float32))
    assert trace == ['sync', (256, 24), 'sync']
    assert not backend._allocations
    for bad in (np.ones((3, 2), dtype=np.float32), np.ones((2, 3), dtype=np.float64)):
        with pytest.raises(ValueError):
            backend.copy_from_numpy(tensor, bad)
    backend._commands = []
    with pytest.raises(RuntimeError, match='cannot be recorded'):
        backend.copy_from_numpy(tensor, np.ones((2, 3), dtype=np.float32))


@kernel_test
@pytest.mark.parametrize('ta,tb', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('m,n,k', [(3, 5, 7), (16, 32, 16)])
def test_all_matmul_layouts_and_padding(ta, tb, m, n, k):
    backend = ScheduleOracle()
    op = CublasDenseOps(backend, fp8=False)
    a = random((k, m) if ta else (m, k))
    b = random((n, k) if tb else (k, n), seed=43)
    y = op.mm(backend.wrap(a), backend.wrap(b), ta, tb)
    close(y, (a.T if ta else a) @ (b.T if tb else b))


@kernel_test
def test_scale_is_inverse_and_cache_waits_for_producer():
    backend = ScheduleOracle(overlap=True)
    op = CublasDenseOps(backend)
    scale = backend.wrap(torch.tensor([8.], device=DEVICE))
    with op.reuse_scope():
        inverse = op._inverse(scale)
        with backend.on_stream(backend.weight_stream):
            assert op._inverse(scale) is inverse
        close(inverse, torch.tensor([0.125]))
    assert backend.events == [('record', 1, 1), ('wait', 1, 2)]


@kernel_test
@pytest.mark.parametrize('rows', [3, 16])
def test_different_parameter_precision_casts_gemm_output(rows):
    backend = ScheduleOracle()
    op = CublasDenseOps(backend, fp8=False)
    a, b = random((rows, 16)), random((16, 16), seed=43)
    result = op.mm(backend.wrap(a), backend.wrap(b), out_dtype='fp16')
    assert result.dtype == 'fp16'
    close(result, (a @ b).half(), atol=0, rtol=0)


@kernel_test
@pytest.mark.parametrize('overlap', [False, True])
def test_block_gradients_schedule_replay_and_changed_input(overlap):
    from model.model_instinct import InstinctBlock, InstinctConfig, precompute_freqs_cis
    config = BlockConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, intermediate_size=48)
    ref = InstinctBlock(0, InstinctConfig(hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            intermediate_size=48, flash_attn=False, dropout=0.0)).to(DEVICE)
    backend = ScheduleOracle(overlap)
    weights = {n: backend.wrap(p.clone()) for n, p in ref.named_parameters()}
    block = ScheduledDenseBlock(CublasDenseOps(backend, fp8=False), weights, config)
    x = random((1, 5, 32)).requires_grad_()
    dy = random(x.shape, seed=44)
    cos, sin = [t.to(DEVICE) for t in precompute_freqs_cis(16, 5)]
    xb, dyb, cb, sb = [backend.wrap(t.clone()) for t in (x, dy, cos, sin)]

    def step():
        out, saved = block.forward(xb, cb, sb)
        dx, grads = block.backward(saved, dyb)
        return out, dx, grads

    plan = backend.build(step)
    count = len(backend._allocations)
    for iteration in range(2):
        # Changing values while preserving addresses must refresh every packed input.
        if iteration:
            with torch.no_grad():
                x.mul_(0.5)
                xb.storage.copy_(x)
        ref.zero_grad(set_to_none=True)
        x.grad = None
        expected = ref(x, (cos, sin))[0]
        expected.backward(dy)
        out, dx, grads = plan.replay()
        close(out, expected, atol=5e-5, rtol=3e-4)
        close(dx, x.grad, atol=5e-5, rtol=3e-4)
        for name, p in ref.named_parameters():
            close(grads[name], p.grad, atol=8e-5, rtol=5e-4)
        assert len(backend._allocations) == count
    commands = plan.describe()
    if overlap:
        assert sum(c['operation'] == 'cublasLt:dW' and c['lane'] == 'dW' for c in commands) == 7
        assert commands[-1]['operation'] == 'event:wait' and commands[-1]['lane'] == 'main'
        assert any(e[0] == 'wait' and e[2] == 2 for e in backend.events)
    else:
        assert not backend.events and all(c['lane'] == 'main' for c in commands)
    with pytest.raises(RuntimeError, match='one execution plan'):
        backend.build(step)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA GPU and built cuBLASLt shim')
@pytest.mark.parametrize('fp8,overlap,graph', [(0, False, False), (1, False, False), (1, True, False), (1, True, True)])
def test_native_smoke(fp8, overlap, graph):
    root = Path('model/dense_cublas/native/build')
    if not os.environ.get('INSTINCT_CUBLASLT_BRIDGE') and not any(p.exists() for p in (
            root / 'libinstinct_cublaslt.so', root / 'Release/instinct_cublaslt.dll', root / 'instinct_cublaslt.dll')):
        pytest.skip('build cuBLASLt shim first')
    if fp8 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    env = dict(os.environ)
    env.pop('TRITON_INTERPRET', None)
    cmd = [sys.executable, '-m', 'experiments.smoke_dense_cublas', '--fp8', str(fp8)]
    cmd += ['--overlap_dw'] if overlap else []
    cmd += ['--cuda_graph'] if graph else []
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report['status'] == 'ok'
    assert not report['torch_imported'] and not report['torchao_imported']
    assert report['prepared_memory']['arena_bytes'] == report['after_replay_memory']['arena_bytes']
