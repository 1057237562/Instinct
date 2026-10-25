"""Validate the V2 TMA path against compact-GQA Flex and the old FP8 compiler."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from model.packed_attention import (
    _flex_kernel_options, build_causal_flex_mask, build_packed_flex_mask, packed_flex_attention,
)


def test_tma_selector_is_limited_to_validated_training_buckets(monkeypatch):
    from model import cuda_fastpath
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: True)
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', 'auto')
    q = SimpleNamespace(shape=torch.Size((3, 4096, 16, 96)), dtype=torch.bfloat16,
                        is_cuda=True, device=torch.device('cuda'))
    k = SimpleNamespace(shape=torch.Size((3, 4096, 4, 96)))
    assert _flex_kernel_options(q, k)['bwd_USE_TMA'] is True
    monkeypatch.delenv('INSTINCT_V2_FLEX_TMA')
    assert _flex_kernel_options(q, k)['bwd_USE_TMA'] is True
    assert 'USE_TMA' not in _flex_kernel_options(q, k)  # forward keeps its original backend
    with torch.no_grad():
        assert _flex_kernel_options(q, k) == {'BACKEND': 'TRITON'}
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', 'auto')
    q.shape, k.shape = torch.Size((4, 3072, 16, 96)), torch.Size((4, 3072, 4, 96))
    assert _flex_kernel_options(q, k) == {'BACKEND': 'TRITON'}
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', '1')
    assert _flex_kernel_options(q, k)['bwd_USE_TMA'] is True
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', 'auto')
    q.shape, k.shape = torch.Size((1, 137, 16, 96)), torch.Size((1, 137, 4, 96))
    assert _flex_kernel_options(q, k) == {'BACKEND': 'TRITON'}
    q.shape, k.shape = torch.Size((3, 4096, 16, 96)), torch.Size((3, 4096, 4, 96))
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', '0')
    assert _flex_kernel_options(q, k) == {'BACKEND': 'TRITON'}
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', 'auto')
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: False)
    assert _flex_kernel_options(q, k) == {'BACKEND': 'TRITON'}
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', 'invalid')
    with pytest.raises(ValueError, match='INSTINCT_V2_FLEX_TMA'):
        _flex_kernel_options(q, k)


@pytest.mark.gpu
@pytest.mark.parametrize('batch,length,packed', [(4, 3072, True), (3, 4096, False)])
def test_tma_actual_buckets_output_gradients_and_isolation(monkeypatch, batch, length, packed):
    torch.manual_seed(607)
    q, k, v = [torch.randn(batch, length, heads, 96, device='cuda',
                          dtype=torch.bfloat16, requires_grad=True) for heads in (16, 4, 4)]
    ids = (torch.arange(length, device='cuda') // 997)[None].expand(batch, -1).clone()
    ids[:, -13:] = -1
    mask = build_packed_flex_mask(ids) if packed else build_causal_flex_mask(length, 'cuda:0')
    fn = torch.compile(lambda q, k, v: packed_flex_attention(q, k, v, mask),
                       fullgraph=True, dynamic=False, mode='max-autotune-no-cudagraphs')
    dy = torch.randn_like(q)
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', '0')
    expected = fn(q, k, v)
    reference_grads = torch.autograd.grad(expected, (q, k, v), dy)
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', '1')
    actual = fn(q, k, v)
    grads = torch.autograd.grad(actual, (q, k, v), dy)
    errors = []
    for a, e in zip((actual, *grads), (expected, *reference_grads)):
        assert torch.isfinite(a).all()
        errors.append(float((a.detach().float() - e.detach().float()).norm() / e.detach().float().norm()))
    assert max(errors) < .005
    # Preserve the input strides/grad flags so this exercises the SAME kernel.
    # Detached clones trigger another compilation/autotune whose reduction
    # tile can differ, confounding a bitwise causality check with rounding.
    with torch.no_grad():
        k[:, 997:] += 7
        v[:, 997:] -= 11
    changed = fn(q, k, v)
    torch.testing.assert_close(changed[:, :997], actual[:, :997], rtol=0, atol=0)
    print(json.dumps(dict(test='attention', length=length, relative_l2=errors)))


@pytest.mark.gpu
@pytest.mark.parametrize('batch,length,packed', [(4, 3072, True), (3, 4096, False)])
def test_fp8_cublas_tma_trunk_against_original_compiler(monkeypatch, batch, length, packed):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training
    from torch._inductor import list_mode_options
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    torch.manual_seed(613)
    cfg = InstinctConfig(hidden_size=1536, intermediate_size=5504, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96, vocab_size=128,
        max_position_embeddings=4096, use_grad_checkpoint=4)
    raw = cast_training_parameters(InstinctForCausalLM(cfg), 'bf16').cuda().train()
    apply_torchao_fp8_training(raw, SimpleNamespace(fp8_training='tensorwise', fp8_filter='auto',
        dtype='bfloat16', device='cuda', use_compile=1))
    reference = copy.deepcopy(raw)
    original = torch.compile(reference, mode='max-autotune-no-cudagraphs', dynamic=False)
    optimized = torch.compile(raw, options={**list_mode_options('max-autotune-no-cudagraphs'),
        'max_autotune_gemm_backends': 'ATEN'}, dynamic=False)
    tokens = torch.randint(3, 128, (batch, length), device='cuda')
    labels = tokens.clone()
    ids = None
    if packed:
        ids = (torch.arange(length, device='cuda') // 997)[None].expand(batch, -1).contiguous()
        labels[:, 1:][ids[:, 1:] != ids[:, :-1]] = -100
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', '0')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = original(tokens, labels=labels, sequence_ids=ids).loss
    expected.backward()
    monkeypatch.setenv('INSTINCT_V2_FLEX_TMA', 'auto')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = optimized(tokens, labels=labels, sequence_ids=ids).loss
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=.002, atol=.003)
    dot = norm_a = norm_e = error = 0.
    for a, e in zip(raw.parameters(), reference.parameters()):
        assert (a.grad is None) == (e.grad is None)
        if a.grad is not None:
            ga, ge = a.grad.float(), e.grad.float()
            assert torch.isfinite(ga).all()
            dot += float((ga * ge).sum())
            norm_a += float(ga.square().sum())
            norm_e += float(ge.square().sum())
            error += float((ga-ge).square().sum())
    cosine, ratio = dot/(norm_a*norm_e)**.5, (norm_a/norm_e)**.5
    assert cosine > .99
    assert .95 < ratio < 1.05
    print(json.dumps(dict(test='fp8_trunk', length=length, cosine=cosine,
        norm_ratio=ratio, relative_l2=(error/norm_e)**.5,
        loss=float(actual.detach()), reference_loss=float(expected.detach()))))
