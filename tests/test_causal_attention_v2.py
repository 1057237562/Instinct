"""Single-document V2 inputs must retain the compiled causal Flex path."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.flash_attn_4 import flash_attention, resolve_flash_attn
from model.packed_attention import (
    PackedFlexMask, build_causal_flex_mask, maybe_prepare_causal_flex_mask,
)
from trainer.trainer_utils import prepare_lm_batch


OPTIONS = dict(head_dim=96, num_attention_heads=16, num_key_value_heads=4)


def test_single_document_loader_contract_is_preserved():
    tokens = torch.arange(13)[None]
    labels = tokens.clone()
    labels[:, -3:] = -100
    ids = torch.zeros_like(tokens)
    ids[:, -3:] = -1
    inputs, targets, sequences = prepare_lm_batch((tokens, labels, ids), 'cpu')
    assert sequences is None
    assert torch.equal(inputs, tokens) and torch.equal(targets, labels)
    ids[:, 5:10] = 1
    assert prepare_lm_batch((tokens, labels, ids), 'cpu')[2] is not None
    assert maybe_prepare_causal_flex_mask(torch.empty(1, 13, 16), **OPTIONS) is None


@pytest.mark.parametrize('mode', ['eval', 'cache', 'attnres'])
def test_model_keeps_inference_cache_and_other_residual_paths(monkeypatch, mode):
    import model.model_instinct as module
    cfg = module.InstinctConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96, vocab_size=32,
        max_position_embeddings=32, residual_type='attnres' if mode == 'attnres' else 'standard',
        attnres_block_size=2)
    model = module.InstinctForCausalLM(cfg)
    model.train(mode != 'eval')

    def unexpected(*args, **kwargs):
        raise AssertionError('V2 training selector was called on an excluded model path')

    monkeypatch.setattr(module, 'maybe_prepare_causal_flex_mask', unexpected)
    tokens = torch.randint(1, 32, (1, 7))
    model(tokens, use_cache=mode == 'cache')


@pytest.mark.gpu
def test_causal96_eligibility_and_cache(monkeypatch):
    from model import cuda_fastpath
    ref = torch.empty(2, 137, 1536, device='cuda', dtype=torch.bfloat16)
    monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'auto')
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: False)
    assert maybe_prepare_causal_flex_mask(ref, **OPTIONS) is None
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: True)
    mask = maybe_prepare_causal_flex_mask(ref, **OPTIONS)
    assert isinstance(mask, PackedFlexMask)
    assert mask.block_mask is maybe_prepare_causal_flex_mask(ref[:1], **OPTIONS).block_mask
    assert mask.block_mask is not build_causal_flex_mask(257, 'cuda:0').block_mask
    for kwargs in (dict(enabled=False), dict(dropout_p=.1),
                   dict(attention_mask=torch.ones(2, 137, device='cuda'))):
        assert maybe_prepare_causal_flex_mask(ref, **OPTIONS, **kwargs) is None
    assert maybe_prepare_causal_flex_mask(ref[:, :1], **OPTIONS) is None
    assert maybe_prepare_causal_flex_mask(ref.float(), **OPTIONS) is None
    assert maybe_prepare_causal_flex_mask(ref, **dict(OPTIONS, head_dim=32)) is None
    assert maybe_prepare_causal_flex_mask(ref, **dict(OPTIONS, num_attention_heads=8)) is None
    with torch.no_grad():
        assert maybe_prepare_causal_flex_mask(ref, **OPTIONS) is None
    monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'sdpa')
    assert maybe_prepare_causal_flex_mask(ref, **OPTIONS) is None
    monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'invalid')
    with pytest.raises(ValueError, match='BACKEND'):
        maybe_prepare_causal_flex_mask(ref, **OPTIONS)


@pytest.mark.gpu
@pytest.mark.parametrize('length', [137, 4096])
def test_causal96_output_gradients_and_future_isolation(monkeypatch, length):
    monkeypatch.setenv('INSTINCT_FLEX_AUTOTUNE', '0')
    monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'sdpa')
    resolve_flash_attn()
    torch.manual_seed(231)
    q, k, v = [torch.randn(1, length, heads, 96, device='cuda',
                          dtype=torch.bfloat16, requires_grad=True) for heads in (16, 4, 4)]
    mask = build_causal_flex_mask(length, 'cuda:0')
    actual = flash_attention(q, k, v, attention_mask=mask)
    expected = flash_attention(q, k, v)
    dy = torch.randn_like(q)
    ga = torch.autograd.grad(actual, (q, k, v), dy)
    ge = torch.autograd.grad(expected, (q, k, v), dy)
    for a, e in [(actual, expected), *zip(ga, ge)]:
        assert torch.isfinite(a).all()
        assert float((a.detach().float()-e.detach().float()).norm()/e.detach().float().norm()) < .01
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, 10:] += 7
    changed_v[:, 10:] -= 11
    changed = flash_attention(q.detach(), changed_k, changed_v, attention_mask=mask)
    torch.testing.assert_close(changed[:, :10], actual[:, :10], atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize('fp8', [False, True])
def test_single_document_compiled_checkpoint_has_flex_and_matches_sdpa(monkeypatch, fp8):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training
    from torch._dynamo.backends.registry import lookup_backend

    resolve_flash_attn()
    monkeypatch.setenv('INSTINCT_FLEX_AUTOTUNE', '0')
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    torch.manual_seed(233)
    # Two checkpoint regions catch graph breaks that a single attention test
    # cannot; use the actual V2 Q/KV/head widths with a smaller residual stream.
    cfg = InstinctConfig(hidden_size=192, intermediate_size=512, num_hidden_layers=2,
                         num_attention_heads=16, num_key_value_heads=4, head_dim=96,
                         vocab_size=128, max_position_embeddings=256, use_grad_checkpoint=2)
    model = cast_training_parameters(InstinctForCausalLM(cfg), 'bf16').cuda().train()
    if fp8:
        apply_torchao_fp8_training(model, SimpleNamespace(fp8_training='tensorwise',
            fp8_filter='auto', dtype='bfloat16', device='cuda', use_compile=1))
    reference = copy.deepcopy(model)
    graphs = []
    inductor = lookup_backend('inductor')

    def backend(gm, inputs):
        graphs.append(gm)
        return inductor(gm, inputs)

    compiled = torch.compile(model, backend=backend)
    compiled_reference = torch.compile(reference)
    for length in (128, 137):
        tokens = torch.randint(1, 128, (1, length))
        labels = tokens.clone()
        labels[:, -7:] = -100
        ids = torch.zeros_like(tokens)
        ids[:, -7:] = -1
        tokens, labels, ids = prepare_lm_batch((tokens, labels, ids), 'cuda')
        assert ids is None  # Exercise the actual loader's single-document branch.
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'auto')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = compiled(tokens, labels=labels, sequence_ids=ids)
        actual.loss.backward()
        monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'sdpa')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = compiled_reference(tokens, labels=labels, sequence_ids=ids)
        expected.loss.backward()
        torch.testing.assert_close(actual.loss, expected.loss, rtol=.002, atol=.003)
        difference = total = actual_norm = dot = 0.
        for p, q in zip(model.parameters(), reference.parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                a, e = p.grad.float(), q.grad.float()
                difference += float((a-e).square().sum())
                total += float(e.square().sum())
                actual_norm += float(a.square().sum())
                dot += float((a*e).sum())
        if fp8:
            assert dot/(actual_norm*total)**.5 > .99
            assert .95 < (actual_norm/total)**.5 < 1.05
        else:
            assert (difference/total)**.5 < .02
    trunk = [gm for gm in graphs if gm.code.count('tag_activation_checkpoint') >= 2]
    assert trunk, 'Causal Flex must preserve both full-layer checkpoint regions'
    assert any('flex_attention' in module.code for gm in trunk for module in gm.modules()
               if isinstance(module, torch.fx.GraphModule))
    # The optimization does not add persistent state to checkpoints.
    assert model.state_dict().keys() == reference.state_dict().keys()
