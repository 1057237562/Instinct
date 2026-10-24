"""CPU interpreter or GPU tests of the actual kernels; torch is reference-only.

CPU: TRITON_INTERPRET=1 python -m pytest -q tests/test_dense_triton.py
GPU: python -m pytest -q tests/test_dense_triton.py
"""
import os
import subprocess
import sys

import datasets  # Windows DLL import order required by AGENTS.md.
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('triton')
from model.dense_triton import elementwise as ew
from model.dense_triton.ops import BlockConfig, DenseBlock, DenseOps
from model.dense_triton.runtime import Tensor


INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
DEVICE = 'cpu' if INTERPRET else 'cuda'
kernel_test = pytest.mark.skipif(not INTERPRET and not torch.cuda.is_available(), reason='requires interpreter or CUDA')
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16,
          'fp8e4nv': torch.float8_e4m3fn, 'fp8e5': torch.float8_e5m2,
          'i32': torch.int32, 'i64': torch.int64}


class ReferenceBackend:
    """Test adapter, never imported by the implementation."""
    def empty(self, shape, dtype, zero=False):
        factory = torch.zeros if zero else torch.empty
        return self.wrap(factory(shape, dtype=DTYPES[dtype], device=DEVICE))

    def wrap(self, value):
        value = value.detach().contiguous().to(DEVICE)
        dtype = next(k for k, v in DTYPES.items() if v == value.dtype)
        return Tensor(tuple(value.shape), dtype, value, self)

    def launch(self, fn, grid, args, **constants):
        fn[grid](*[t.storage.view(t.shape) for t in args], **constants,
                 enable_fp_fusion=False, num_warps=4, num_stages=2)


def actual(tensor):
    return tensor.storage.view(tensor.shape).float().cpu()


def close(a, b, atol=1e-5, rtol=1e-4):
    a = actual(a) if isinstance(a, Tensor) else a.detach().float().cpu()
    b = actual(b) if isinstance(b, Tensor) else b.detach().float().cpu()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def random(shape, seed=42, scale=1.0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).mul(scale).to(DEVICE)


def test_imports_do_not_load_torch():
    code = '''
import sys
from model.dense_triton import ops, runtime, attention, linear, elementwise
assert 'torch' not in sys.modules and 'torchao' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_unsupported_dropout_and_shapes_rejected():
    with pytest.raises(ValueError, match='dropout'):
        BlockConfig(dropout=0.1)
    with pytest.raises(ValueError):
        BlockConfig(num_attention_heads=3, num_key_value_heads=2)
    with pytest.raises(ValueError):
        Tensor((2, 3), 'fp32', None, None).view(7)


@kernel_test
@pytest.mark.parametrize('rows,width', [(3, 96), (9, 768), (131, 33)])
def test_norm_forward_backward(rows, width):
    backend = ReferenceBackend()
    op = DenseOps(backend)
    x = random((rows, width)).requires_grad_()
    w = random((width,), seed=43).requires_grad_()
    dy = random(x.shape, seed=44)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    y.backward(dy)
    result, saved = op.norm(backend.wrap(x), backend.wrap(w))
    dx, dw = op.norm_backward(saved, backend.wrap(dy))
    close(result, y)
    close(dx, x.grad)
    close(dw, w.grad, atol=5e-5)


@kernel_test
@pytest.mark.parametrize('size', [1, 259, 2432])
def test_swiglu_forward_backward(size):
    backend = ReferenceBackend()
    op = DenseOps(backend)
    g = random((size,)).requires_grad_()
    u = random((size,), seed=43).requires_grad_()
    dy = random((size,), seed=44)
    y = F.silu(g) * u
    y.backward(dy)
    result, saved = op.swiglu(backend.wrap(g), backend.wrap(u))
    dg, du = op.swiglu_backward(saved, backend.wrap(dy))
    close(result, y)
    close(dg, g.grad)
    close(du, u.grad)


@kernel_test
def test_rope_split_half_and_nonzero_offset():
    backend = ReferenceBackend()
    op = DenseOps(backend)
    x = random((2, 7, 3, 96)).requires_grad_()
    angles = torch.arange(12, device=DEVICE)[:, None] / (1e6 ** (torch.arange(0, 96, 2, device=DEVICE) / 96))
    cos = torch.cat((angles.cos(), angles.cos()), -1) * 1.1
    sin = torch.cat((angles.sin(), angles.sin()), -1) * 1.1
    y = x * cos[3:10, None] + torch.cat((-x[..., 48:], x[..., :48]), -1) * sin[3:10, None]
    dy = random(x.shape, seed=44)
    y.backward(dy)
    result = op.rotary(backend.wrap(x), backend.wrap(cos), backend.wrap(sin), start=3)
    dx = op.rotary(backend.wrap(dy), backend.wrap(cos), backend.wrap(sin), start=3, backward=True)
    close(result, y)
    close(dx, x.grad)


@kernel_test
@pytest.mark.parametrize('seq,hq,hkv,dim,masked', [(1, 2, 1, 16, False), (35, 4, 2, 96, False), (37, 4, 1, 32, True)])
def test_causal_gqa_forward_and_all_gradients(seq, hq, hkv, dim, masked):
    backend = ReferenceBackend()
    op = DenseOps(backend)
    q = random((2, seq, hq, dim), scale=0.4).requires_grad_()
    k = random((2, seq, hkv, dim), seed=43, scale=0.4).requires_grad_()
    v = random((2, seq, hkv, dim), seed=44).requires_grad_()
    mask = torch.ones((2, seq), dtype=torch.int32, device=DEVICE)
    if masked:
        mask[0, :3] = 0  # Fully masked initial rows must return zeros, not NaNs.
        mask[1, -4:] = 0
    allowed = torch.ones((seq, seq), dtype=torch.bool, device=DEVICE).tril()
    allowed = allowed[None, None] & mask[:, None, None, :].bool()
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.repeat_interleave(hq // hkv, 2).transpose(1, 2),
                                       v.repeat_interleave(hq // hkv, 2).transpose(1, 2), attn_mask=allowed).transpose(1, 2)
    dy = random(y.shape, seed=45)
    y.backward(dy)
    out, saved = op.attention(backend.wrap(q), backend.wrap(k), backend.wrap(v), backend.wrap(mask) if masked else None)
    grads = op.attention_backward(saved, backend.wrap(dy))
    close(out, y, atol=3e-5)
    for result, ref in zip(grads, (q.grad, k.grad, v.grad)):
        close(result, ref, atol=5e-5, rtol=3e-4)


def fp8_reference(x, dtype):
    scale = (torch.finfo(dtype).max / x.abs().max().double().clamp_min(1e-12)).float()
    q = (x.float() * scale).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
    return q, scale


@kernel_test
@pytest.mark.parametrize('dtype', ['fp8e4nv', 'fp8e5'])
def test_fp8_quantization_zero_small_and_outlier(dtype):
    if INTERPRET:
        pytest.skip('Triton 3.4 interpreter FP8 RTNE lacks carry/tie handling; validate native casts on GPU')
    if not INTERPRET and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('native FP8 requires SM89+')
    backend = ReferenceBackend()
    op = DenseOps(backend)
    maximum = torch.finfo(DTYPES[dtype]).max
    # An exact maximum fixes scale=1: include rounding ties and exponent carries.
    edges = torch.tensor([0., -0., 1.0625, 1.1875, 1.9375, 1.875,
                          0.0009765625, 0.001953125, maximum, -maximum], device=DEVICE)
    for x in [torch.zeros(33, device=DEVICE), random((1027,)) * 1e-14, random((2051,)), edges]:
        ref, scale = fp8_reference(x, DTYPES[dtype])
        q, s = op.quantize(backend.wrap(x), dtype)
        close(s, scale.reshape(1), rtol=0, atol=0)
        close(q, ref, rtol=0, atol=0)


@kernel_test
@pytest.mark.parametrize('fp8', [False, True])
def test_linear_forward_dx_dw_with_tail_dimensions(fp8):
    if fp8 and INTERPRET:
        pytest.skip('native FP8 casts need GPU; scaled-GEMM recipe has a separate emulated test')
    if fp8 and not INTERPRET and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('native FP8 requires SM89+')
    backend = ReferenceBackend()
    op = DenseOps(backend, fp8=fp8)
    x, w, dy = random((2, 17, 48)), random((80, 48), seed=43), random((2, 17, 80), seed=44)
    if fp8:
        qx, sx = fp8_reference(x.view(-1, 48), torch.float8_e4m3fn)
        qw, sw = fp8_reference(w, torch.float8_e4m3fn)
        qdy, sdy = fp8_reference(dy.view(-1, 80), torch.float8_e5m2)
        y = qx.float() @ qw.float().T / sx / sw
        dx = qdy.float() @ qw.float() / sdy / sw
        dw = qdy.float().T @ qx.float() / sdy / sx
    else:
        y, dx, dw = x.view(-1, 48) @ w.T, dy.view(-1, 80) @ w, dy.view(-1, 80).T @ x.view(-1, 48)
    result, saved = op.linear(backend.wrap(x), backend.wrap(w))
    actual_dx, actual_dw = op.linear_backward(saved, backend.wrap(dy))
    close(result, y.view(2, 17, 80), atol=4e-5)
    close(actual_dx, dx.view(x.shape), atol=4e-5)
    close(actual_dw, dw, atol=4e-5)


@kernel_test
def test_fp8_scaled_gemm_recipe_against_torchao_emulation():
    """Check the three GEMMs using exact prequantized FP32 values.

    The quantization oracle is test-only. This tests our GEMM/dequantization and
    backward wiring, not Triton's native FP8 conversion or Tensor Core execution.
    """
    pytest.importorskip('torchao')
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from dataclasses import replace
    backend = ReferenceBackend()
    class EmulatedQuantOps(DenseOps):
        def quantize(self, x, dtype='fp8e4nv'):
            q, s = fp8_reference(x.storage.view(x.shape), DTYPES[dtype])
            return backend.wrap(q.float()), backend.wrap(s.reshape(1))
    op = EmulatedQuantOps(backend, fp8=True)
    torch.manual_seed(123)
    ref = torch.nn.Sequential(torch.nn.Linear(48, 80, bias=False)).to(DEVICE)
    cfg = replace(Float8LinearConfig.from_recipe_name('tensorwise'), emulate=True, pad_inner_dim=True)
    convert_to_float8_training(ref, config=cfg)
    x = random((2, 17, 48)).requires_grad_()
    dy = random((2, 17, 80), seed=44)
    y = ref(x)
    y.backward(dy)
    result, saved = op.linear(backend.wrap(x), backend.wrap(ref[0].weight))
    dx, dw = op.linear_backward(saved, backend.wrap(dy))
    close(result, y, atol=1e-5)
    close(dx, x.grad, atol=1e-5)
    close(dw, ref[0].weight.grad, atol=3e-5)


@kernel_test
def test_embedding_duplicates_tied_gradient_and_ignored_ce_labels():
    backend = ReferenceBackend()
    ids = torch.tensor([1, 3, 1, 2], dtype=torch.int64, device=DEVICE)
    weight = random((8, 33)).requires_grad_()
    dy = random((4, 33), seed=43)
    y = F.embedding(ids, weight)
    y.backward(dy)
    ti, tw, tdy = map(backend.wrap, (ids, weight, dy))
    out = backend.empty((4, 33), 'fp32')
    existing = random((8, 33), seed=44)
    dw = backend.wrap(existing.clone())
    options = dict(TOKENS=4, DIM=33, VOCAB=8)
    backend.launch(ew.embedding, (4, 1), (ti, tw, out), **options)
    backend.launch(ew.embedding_backward, (4, 1), (ti, tdy, dw), **options)
    close(out, y)
    close(dw, weight.grad + existing)
    logits = random((4, 73)).requires_grad_()
    labels = torch.tensor([1, -100, 72, 0], device=DEVICE)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss.backward()
    losses, valid, dl = backend.empty((4,), 'fp32'), backend.empty((4,), 'i32'), backend.empty((4, 73), 'fp32')
    backend.launch(ew.cross_entropy, (4,), (backend.wrap(logits), backend.wrap(labels), losses, valid, dl), VOCAB=73, BLOCK=128)
    close(actual(losses).sum(), loss)
    assert actual(valid).sum() == 3
    close(dl, logits.grad)


@kernel_test
@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_complete_dense_block_against_instinct_autograd(dtype):
    if dtype == 'bf16' and INTERPRET:
        pytest.skip('BF16 rounding and Tensor Core parity require a CUDA execution')
    from model.model_instinct import InstinctBlock, InstinctConfig, precompute_freqs_cis
    cfg = InstinctConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, intermediate_size=48,
                         flash_attn=False, dropout=0.0)
    torch.manual_seed(123)
    ref = InstinctBlock(0, cfg).to(DEVICE)
    x = random((1, 5, 32)).to(DTYPES[dtype]).requires_grad_()
    dy = random(x.shape, seed=44).to(x.dtype)
    cos, sin = [t.to(DEVICE) for t in precompute_freqs_cis(16, 5)]
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=dtype == 'bf16'):
        y = ref(x, (cos, sin))[0]
    y.backward(dy)
    backend = ReferenceBackend()
    weights = {n: backend.wrap(p) for n, p in ref.named_parameters()}
    config = BlockConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=16, intermediate_size=48)
    block = DenseBlock(DenseOps(backend, fp8=False), weights, config)
    out, saved = block.forward(backend.wrap(x), backend.wrap(cos), backend.wrap(sin))
    dx, grads = block.backward(saved, backend.wrap(dy))
    atol, rtol = (0.04, 0.04) if dtype == 'bf16' else (5e-5, 3e-4)
    close(out, y, atol=atol, rtol=rtol)
    close(dx, x.grad, atol=atol, rtol=rtol)
    assert grads.keys() == weights.keys()
    for name, p in ref.named_parameters():
        close(grads[name], p.grad, atol=max(atol, 8e-5), rtol=max(rtol, 5e-4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@pytest.mark.gpu
@pytest.mark.parametrize('fp8', [0, 1])
def test_direct_cuda_driver_smoke_without_torch(fp8):
    if fp8 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('native FP8 requires SM89+')
    env = dict(os.environ)
    env.pop('TRITON_INTERPRET', None)
    result = subprocess.run([sys.executable, '-m', 'experiments.smoke_dense_triton',
                             '--fp8', str(fp8), '--seq_len', '17'],
                            env=env, check=True, capture_output=True, text=True)
    import json
    report = json.loads(result.stdout)
    assert report['status'] == 'ok'
    assert not report['torch_imported'] and not report['torchao_imported']
    assert len(report['gradient_parameters']) == 11
