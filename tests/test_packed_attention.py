"""Block-sparse packed attention preserves causal/segment and gradient semantics."""

import copy
import importlib

import pytest
import torch

from model.attention_mask import prepare_sdpa_attention_bias
from model.flash_attn_4 import flash_attention
from model.packed_attention import (
    PackedFlexMask, build_packed_flex_mask, document_mask, maybe_prepare_flex_mask,
)
from model.sequence_packing import merge_packed_attention_mask


def test_document_mask_matches_existing_semantics_and_bounds():
    # Repeated IDs, per-row document boundaries, and -1 padding retain exactly
    # the existing equality semantics (not a guessed contiguous-segment policy).
    ids = torch.tensor([[0, 0, 1, 1, -1, -1, -1], [0, 1, 1, 0, 2, 2, -1]])
    b = torch.arange(2)[:, None, None]
    q = torch.arange(9)[None, :, None]
    k = torch.arange(9)[None, None, :]
    actual = document_mask(ids)(b, 0, q, k)
    expected = torch.zeros(2, 9, 9, dtype=torch.bool)
    expected[:, :7, :7] = merge_packed_attention_mask(ids, None) & torch.ones(7, 7, dtype=torch.bool).tril()
    torch.testing.assert_close(actual, expected)


def test_cpu_and_disabled_paths_keep_sdpa(monkeypatch):
    ids = torch.zeros(2, 7, dtype=torch.long)
    ref = torch.empty(0)
    assert maybe_prepare_flex_mask(ids, ref) is None
    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'sdpa')
    assert maybe_prepare_flex_mask(ids, ref) is None
    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'invalid')
    with pytest.raises(ValueError, match='BACKEND'):
        maybe_prepare_flex_mask(ids, ref)


def test_compiled_mask_survives_streaming_bucket_shapes_on_cpu(monkeypatch):
    """Exercise real Dynamo guards without CUDA/Triton compilation costs."""
    import model.packed_attention as packed
    from torch.nn.attention.flex_attention import create_mask

    original_compile = torch.compile
    def cpu_compile(fn, **kwargs):
        return original_compile(fn, backend='eager', **kwargs)

    packed._flex_functions.cache_clear()
    monkeypatch.setattr(torch, 'compile', cpu_compile)
    torch.compiler.reset()
    try:
        with torch._dynamo.config.patch(
            cache_size_limit=8, accumulated_cache_size_limit=256,
            fail_on_recompile_limit_hit=True, suppress_errors=False,
        ):
            for index, length in enumerate(range(129, 1666, 128)):
                batch = 1 + index % 3
                ids = (torch.arange(length) // 31)[None].expand(batch, -1).clone()
                mask = packed.build_packed_flex_mask(ids)
                dense = create_mask(mask.block_mask.mask_mod, batch, 1, length, length, device='cpu')
                expected = merge_packed_attention_mask(ids, None) & torch.ones(length, length, dtype=torch.bool).tril()
                torch.testing.assert_close(dense[:, 0], expected)
    finally:
        packed._flex_functions.cache_clear()
        torch.compiler.reset()


def test_flex_compilation_preserves_kernel_options_and_sets_both_limits(monkeypatch):
    import model.packed_attention as packed
    calls = []
    def compile_region(fn, **kwargs):
        calls.append(kwargs)
        return fn
    monkeypatch.setattr(torch, 'compile', compile_region)
    packed._flex_functions.cache_clear()
    try:
        with torch._dynamo.config.patch(
            cache_size_limit=8, accumulated_cache_size_limit=256,
            fail_on_recompile_limit_hit=False, suppress_errors=False,
        ):
            packed._flex_functions()
            assert calls == [{'fullgraph': True}, {'fullgraph': True}]
            assert torch._dynamo.config.cache_size_limit == 128
            assert torch._dynamo.config.accumulated_cache_size_limit == 4096
            assert torch._dynamo.config.fail_on_recompile_limit_hit
    finally:
        packed._flex_functions.cache_clear()


@pytest.mark.gpu
@pytest.mark.parametrize('length,kv_heads', [(80, 2), (137, 4)])
def test_cuda_forward_backward_and_document_isolation(length, kv_heads):
    torch.manual_seed(17)
    ids = torch.arange(length, device='cuda')[None].expand(2, -1).clone() // 23
    ids[0, -9:] = -1
    ids[1, -17:] = -1
    q = torch.randn(2, length, 4, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, length, kv_heads, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    grad = torch.randn_like(q)
    block = build_packed_flex_mask(ids)
    bias = prepare_sdpa_attention_bias(merge_packed_attention_mask(ids, None), q, query_length=length)
    actual = flash_attention(q, k, v, attention_mask=block)
    expected = flash_attention(q, k, v, attention_mask=bias)
    ga = torch.autograd.grad(actual, (q, k, v), grad)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in [(actual, expected), *zip(ga, ge)]:
        assert torch.isfinite(a).all()
        assert ((a.float() - e.float()).norm() / e.float().norm()).item() < 0.01
    # Perturb another document and the query's future: neither can affect its
    # output. Test the actual sparse kernel, not only mask construction.
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, 10:] += 7
    changed_v[:, 10:] -= 11
    changed = flash_attention(q.detach(), changed_k, changed_v, attention_mask=block)
    torch.testing.assert_close(changed[:, :10], actual[:, :10], rtol=0, atol=0)
    # A subsequent batch with different IDs must not reuse stale block metadata.
    new_ids = torch.zeros_like(ids)
    fresh = build_packed_flex_mask(new_ids)
    fresh_bias = prepare_sdpa_attention_bias(merge_packed_attention_mask(new_ids, None), q,
                                           query_length=length)
    new_out = flash_attention(q, k, v, attention_mask=fresh)
    new_ref = flash_attention(q, k, v, attention_mask=fresh_bias)
    assert ((new_out.float() - new_ref.float()).norm() / new_ref.float().norm()).item() < 0.01


@pytest.mark.gpu
def test_flex_eligibility_and_explicit_mask_dropout_fallback(monkeypatch):
    ids = torch.zeros(1, 16, dtype=torch.long, device='cuda')
    ref = torch.empty(0, device='cuda', dtype=torch.float32)
    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'flex')
    assert maybe_prepare_flex_mask(ids, ref) is None  # FP32 compute
    with torch.autocast('cuda', dtype=torch.bfloat16):
        assert maybe_prepare_flex_mask(ids, ref, dropout_p=0.1) is None
        assert maybe_prepare_flex_mask(ids, ref, attention_mask=ids >= 0) is None
        assert isinstance(maybe_prepare_flex_mask(ids, ref), PackedFlexMask)


@pytest.mark.gpu
@pytest.mark.parametrize('variant,checkpoint', [('standard', 1), ('standard', 2), ('loop', 1)])
def test_model_shares_block_mask_and_matches_sdpa_gradients(monkeypatch, variant, checkpoint):
    module = importlib.import_module('model.model_instinct_loop' if variant == 'loop'
                                     else 'model.model_instinct')
    cfg = module.InstinctConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=128, flash_attn=True, use_moe=True,
        use_grad_checkpoint=checkpoint, dropout=0.0,
        prelude_layers=1, recurrent_layers=1, coda_layers=1, loop_iters=2,
        mean_backprop_depth=2, recurrence_sampling='fixed', state_init_std=0.0,
    )
    torch.manual_seed(19)
    model = module.InstinctForCausalLM(cfg).cuda().train()
    reference = copy.deepcopy(model)
    tokens = torch.randint(1, 128, (2, 80), device='cuda')
    ids = (torch.arange(80, device='cuda') // 20)[None].expand(2, -1)
    captured = []
    original = module.flash_attention

    def spy(*a, **kw):
        captured.append(kw['attention_mask'])
        return original(*a, **kw)

    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'flex')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'flash_attention', spy)
        def no_dense_mask(*a, **kw):
            raise AssertionError('Flex path materialized the dense packing mask')
        patch.setattr(module, 'merge_packed_attention_mask', no_dense_mask)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(tokens, labels=tokens, sequence_ids=ids)
        (result.loss + result.aux_loss).backward()
    assert len(captured) >= 3
    assert isinstance(captured[0], PackedFlexMask)
    assert all(mask is captured[0] for mask in captured)
    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'sdpa')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = reference(tokens, labels=tokens, sequence_ids=ids)
    (expected.loss + expected.aux_loss).backward()
    torch.testing.assert_close(result.loss, expected.loss, atol=0.003, rtol=0.001)
    grads, references = [], []
    for (name, p), (_, r) in zip(model.named_parameters(), reference.named_parameters()):
        assert (p.grad is None) == (r.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            grads.append(p.grad.flatten().float())
            references.append(r.grad.flatten().float())
    a, e = torch.cat(grads), torch.cat(references)
    assert ((a - e).norm() / e.norm()).item() < 0.02
    # No weights, buffers, or checkpoint layout changes.
    reference.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.gpu
@pytest.mark.slow
def test_compiled_moe_training_two_batches(monkeypatch):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'flex')
    cfg = InstinctConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=128, flash_attn=True, use_moe=True,
        use_grad_checkpoint=1, dropout=0.0,
    )
    torch.manual_seed(8)
    model = InstinctForCausalLM(cfg).cuda().train()
    reference = copy.deepcopy(model)
    compiled = torch.compile(model)
    tokens = torch.randint(1, 128, (2, 80), device='cuda')
    for segment_length in (20, 37):
        ids = (torch.arange(80, device='cuda') // segment_length)[None].expand(2, -1)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = compiled(tokens, labels=tokens, sequence_ids=ids)
            expected = reference(tokens, labels=tokens, sequence_ids=ids)
        (actual.loss + actual.aux_loss).backward()
        (expected.loss + expected.aux_loss).backward()
        torch.testing.assert_close(actual.loss, expected.loss, atol=0.003, rtol=0.001)
        grads, refs = [], []
        for p, r in zip(model.parameters(), reference.parameters()):
            assert p.grad is not None and r.grad is not None
            assert torch.isfinite(p.grad).all()
            grads.append(p.grad.flatten().float())
            refs.append(r.grad.flatten().float())
        a, b = torch.cat(grads), torch.cat(refs)
        assert ((a - b).norm() / b.norm()).item() < 0.02
