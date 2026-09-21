"""Packed mask sharing: semantics, gradients, checkpointing and storage reuse."""

import copy
import importlib

import pytest
import torch

from model.attention_mask import PreparedAttentionBias, prepare_sdpa_attention_bias
from model.flash_attn_4 import flash_attention
from model.sequence_packing import merge_packed_attention_mask


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_bias_combines_causality_segments_and_padding(dtype):
    ids = torch.tensor([[0, 0, 1, 1, -1, -1]])
    mask = merge_packed_attention_mask(ids, None)
    bias = prepare_sdpa_attention_bias(mask, torch.empty(0, dtype=dtype), query_length=6)
    expected = mask[:, None] & torch.ones(6, 6, dtype=torch.bool).tril()
    assert isinstance(bias, PreparedAttentionBias)
    assert bias.tensor.dtype == dtype
    assert bias.tensor.shape == (1, 1, 6, 6)
    assert bias.tensor.stride(-2) % 64 == 0
    assert not bias.tensor.requires_grad
    assert torch.equal(bias.tensor == 0, expected)
    assert torch.equal(torch.isneginf(bias.tensor), ~expected)


def test_autocast_uses_compute_dtype_without_reusing_previous_batch():
    ref = torch.empty(0, dtype=torch.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        first = prepare_sdpa_attention_bias(torch.ones(1, 7, 7), ref, query_length=7)
        second = prepare_sdpa_attention_bias(torch.eye(7)[None], ref, query_length=7)
    assert first.tensor.dtype == torch.bfloat16
    assert first.tensor.data_ptr() != second.tensor.data_ptr()
    assert first.tensor[0, 0, 6, 0] == 0
    assert torch.isneginf(second.tensor[0, 0, 6, 0])


@pytest.mark.parametrize('mask_kind', ['binary_float', 'packed', 'fully_masked_row'])
def test_sdpa_forward_and_gradients_match_boolean_path(mask_kind):
    torch.manual_seed(5)
    q = torch.randn(2, 7, 4, 8, requires_grad=True)
    k = torch.randn(2, 7, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    if mask_kind == 'binary_float':
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]] * 2, dtype=torch.float32)
    else:
        ids = torch.tensor([[0, 0, 0, 1, 1, -1, -1]] * 2)
        mask = merge_packed_attention_mask(ids, None)
        if mask_kind == 'fully_masked_row':
            mask[:, 3, :] = False
    expected = flash_attention(q, k, v, attention_mask=mask)
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    bias = prepare_sdpa_attention_bias(mask, q, query_length=7)
    actual = flash_attention(q, k, v, attention_mask=bias)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('variant,moe,checkpoint', [
    ('standard', False, 0), ('standard', True, 1), ('standard', False, 2),
    ('loop', False, 0), ('loop', True, 1),
])
def test_model_reuses_bias_and_matches_old_gradients(monkeypatch, variant, moe, checkpoint):
    module = importlib.import_module(
        'model.model_instinct_loop' if variant == 'loop' else 'model.model_instinct'
    )
    config = module.InstinctConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, dropout=0.0, flash_attn=True,
        use_moe=moe, use_grad_checkpoint=checkpoint,
        prelude_layers=1, recurrent_layers=1, coda_layers=1, loop_iters=2,
        state_init_std=0.0, recurrence_sampling='fixed', mean_backprop_depth=2,
    )
    torch.manual_seed(9)
    model = module.InstinctForCausalLM(config).train()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[2, 3, 4, 5, 6, 0, 0]])
    segments = torch.tensor([[0, 0, 0, 1, 1, -1, -1]])
    labels = ids.clone()
    labels[:, 3] = -100
    labels[:, 5:] = -100
    captured = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        captured.append(kwargs['attn_mask'])
        assert kwargs['is_causal'] is False
        return original_sdpa(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.functional, 'scaled_dot_product_attention', spy)
        result = model(ids, labels=labels, sequence_ids=segments)
        (result.loss + result.aux_loss).backward()
    assert len(captured) >= 3
    assert captured[0].is_floating_point()
    assert all(mask is captured[0] for mask in captured)
    # Compare against the former layer-local boolean-mask conversion path.
    with monkeypatch.context() as patch:
        patch.setattr(module, 'prepare_sdpa_attention_bias', lambda mask, *a, **kw: mask)
        expected = reference(ids, labels=labels, sequence_ids=segments)
        (expected.loss + expected.aux_loss).backward()
    torch.testing.assert_close(result.loss, expected.loss, atol=1e-6, rtol=1e-5)
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5,
                                       msg=lambda msg, name=name: f'{name}: {msg}')


def test_compiled_bias_pytree_and_backward():
    def forward(q, k, v, mask):
        bias = prepare_sdpa_attention_bias(mask, q, query_length=q.size(1))
        return flash_attention(q, k, v, attention_mask=bias)

    torch.manual_seed(4)
    q = torch.randn(1, 7, 2, 8, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = torch.ones(1, 7, 7, dtype=torch.bool)
    expected = forward(q, k, v, mask)
    expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))
    compiled = torch.compile(forward, backend='aot_eager', fullgraph=True)
    actual = compiled(q, k, v, mask)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want)


@pytest.mark.gpu
@pytest.mark.parametrize('compiled', [False, True])
def test_cuda_memory_efficient_backward_reuses_bias_storage(compiled):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(2)
    q = torch.randn(2, 80, 4, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 80, 2, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    segments = torch.arange(80, device='cuda')[None].expand(2, -1) // 20
    mask = merge_packed_attention_mask(segments, None)
    bias = prepare_sdpa_attention_bias(mask, q, query_length=80)
    attention = (torch.compile(flash_attention, fullgraph=True) if compiled
                 else flash_attention)
    saved_biases = []

    def pack(t):
        # Eager saves a head-expanded view; AOTAutograd can save the compact
        # [B, 1, Q, K] input and reconstruct that view in backward.
        if t.ndim == 4 and t.shape[-2:] == (80, 80):
            saved_biases.append(t)
        return t

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        expected = flash_attention(q, k, v, attention_mask=mask)
        expected_grads = torch.autograd.grad(expected.float().square().sum(), (q, k, v))
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            outputs = [attention(q, k, v, attention_mask=bias) for _ in range(3)]
        actual_grads = torch.autograd.grad(outputs[0].float().square().sum(), (q, k, v))
    assert len(saved_biases) == 3
    assert all(t.untyped_storage().data_ptr() == bias.tensor.untyped_storage().data_ptr()
               for t in saved_biases)
    torch.testing.assert_close(outputs[0], expected, atol=2e-3, rtol=2e-3)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.slow
def test_compiled_moe_model_shares_one_bias_with_ffn_checkpoint(monkeypatch):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'sdpa')
    config = InstinctConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, flash_attn=True,
        use_moe=True, use_grad_checkpoint=1, dropout=0.0,
    )
    model = torch.compile(InstinctForCausalLM(config).cuda().train())
    tokens = torch.randint(1, 128, (2, 80), device='cuda')
    segments = (torch.arange(80, device='cuda') // 20)[None].expand(2, -1)
    saved_biases = []

    def pack(t):
        if t.ndim == 4 and t.shape[-2:] == (80, 80) and t.dtype == torch.bfloat16:
            saved_biases.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(tokens, labels=tokens, sequence_ids=segments)
        loss = result.loss + result.aux_loss
        loss.backward()
    assert torch.isfinite(loss)
    assert len(saved_biases) == 3
    assert len({t.untyped_storage().data_ptr() for t in saved_biases}) == 1
    assert all(p.grad is not None for p in model.parameters())
