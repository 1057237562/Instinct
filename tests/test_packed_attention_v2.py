"""Native 96-wide packed GQA and its traceable full-layer checkpoint path."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.attention_mask import prepare_sdpa_attention_bias
from model.flash_attn_4 import flash_attention, resolve_flash_attn
from model.packed_attention import build_packed_flex_mask, maybe_prepare_flex_mask, PackedFlexMask
from model.sequence_packing import merge_packed_attention_mask


@pytest.mark.gpu
@pytest.mark.parametrize('batch,length', [(2, 137), (1, 257)])
def test_head96_output_gradients_and_document_isolation(batch, length, monkeypatch):
    monkeypatch.setenv('INSTINCT_FLEX_AUTOTUNE', '0')
    torch.manual_seed(217)
    ids = (torch.arange(length, device='cuda')//31)[None].expand(batch, -1).clone()
    ids[:, -13:] = -1
    q, k, v = [torch.randn(batch, length, h, 96, device='cuda', dtype=torch.bfloat16,
                          requires_grad=True) for h in (16, 4, 4)]
    dy = torch.randn_like(q)
    mask = build_packed_flex_mask(ids)
    bias = prepare_sdpa_attention_bias(merge_packed_attention_mask(ids, None), q,
                                       query_length=length)
    actual = flash_attention(q, k, v, attention_mask=mask)
    expected = flash_attention(q, k, v, attention_mask=bias)
    ga = torch.autograd.grad(actual, (q, k, v), dy)
    ge = torch.autograd.grad(expected, (q, k, v), dy)
    for a, e in [(actual, expected), *zip(ga, ge)]:
        assert torch.isfinite(a).all()
        assert float((a.detach().float()-e.detach().float()).norm()/e.detach().float().norm()) < .01
    k2, v2 = k.detach().clone(), v.detach().clone()
    k2[:, 10:] += 7
    v2[:, 10:] -= 11
    changed = flash_attention(q.detach(), k2, v2, attention_mask=mask)
    torch.testing.assert_close(changed[:, :10], actual[:, :10], rtol=0, atol=0)


@pytest.mark.gpu
def test_head96_auto_eligibility_is_scoped_to_validated_stack(monkeypatch):
    from model import cuda_fastpath
    ids = torch.zeros(1, 17, device='cuda', dtype=torch.long)
    ref = torch.empty(0, device='cuda', dtype=torch.bfloat16)
    options = dict(head_dim=96, num_attention_heads=16, num_key_value_heads=4)
    monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'auto')
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: False)
    assert maybe_prepare_flex_mask(ids, ref, **options) is None
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: True)
    assert isinstance(maybe_prepare_flex_mask(ids, ref, **options), PackedFlexMask)
    assert maybe_prepare_flex_mask(ids, ref, head_dim=96, num_attention_heads=8,
                                   num_key_value_heads=4) is None
    assert maybe_prepare_flex_mask(ids, ref, head_dim=72) is None
    assert maybe_prepare_flex_mask(ids, ref, head_dim=96, dropout_p=.1) is None
    assert maybe_prepare_flex_mask(ids, ref, head_dim=96, attention_mask=ids >= 0) is None


@pytest.mark.gpu
@pytest.mark.parametrize('fp8', [False, True])
def test_compiled_checkpoint_retains_trunk_and_matches_sdpa(monkeypatch, fp8):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training
    from torch._dynamo.backends.registry import lookup_backend

    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    monkeypatch.setenv('INSTINCT_FLEX_AUTOTUNE', '0')
    resolve_flash_attn()
    torch.manual_seed(223)
    cfg = InstinctConfig(hidden_size=1536, intermediate_size=5504, num_hidden_layers=1,
                         num_attention_heads=16, num_key_value_heads=4, head_dim=96,
                         vocab_size=128, max_position_embeddings=256, use_grad_checkpoint=2)
    model = cast_training_parameters(InstinctForCausalLM(cfg), 'bf16').cuda().train()
    if fp8:
        apply_torchao_fp8_training(model, SimpleNamespace(fp8_training='tensorwise', fp8_filter='auto',
                                   dtype='bfloat16', device='cuda', use_compile=1))
    reference = copy.deepcopy(model)
    graphs = []
    inductor = lookup_backend('inductor')
    def backend(gm, inputs):
        graphs.append(gm.code)
        return inductor(gm, inputs)
    compiled = torch.compile(model, backend=backend)
    # Compare attention backends under the same FP8 compiler arithmetic. An
    # eager-vs-compiled FP8 comparison also changes quantization/fusion rounding.
    compiled_reference = torch.compile(reference)
    tokens = torch.randint(1, 128, (2, 128), device='cuda')
    for span in (31, 53):
        ids = (torch.arange(128, device='cuda')//span)[None].expand(2, -1).clone()
        ids[:, -7:] = -1
        labels = tokens.clone()
        labels[:, -7:] = -100
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'flex')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = compiled(tokens, labels=labels, sequence_ids=ids)
        actual.loss.backward()
        monkeypatch.setenv('INSTINCT_PACKED_ATTENTION_BACKEND', 'sdpa')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = compiled_reference(tokens, labels=labels, sequence_ids=ids)
        expected.loss.backward()
        torch.testing.assert_close(actual.loss, expected.loss, rtol=.002, atol=.003)
        difference = total = actual_norm = dot = 0.
        for p, q in zip(model.parameters(), reference.parameters()):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                difference += float((p.grad.float()-q.grad.float()).square().sum())
                total += float(q.grad.float().square().sum())
                actual_norm += float(p.grad.float().square().sum())
                dot += float((p.grad.float()*q.grad.float()).sum())
        if fp8:
            # E4M3 bin crossings amplify small attention-rounding changes. The
            # independent BF16 and attention-op tests gate the derivative;
            # here check FP8 gradient direction/norm and the loss above. Real
            # packed training additionally uses the held-out loss gate.
            assert dot/(actual_norm*total)**.5 > .99
            assert .95 < (actual_norm/total)**.5 < 1.05
        else:
            assert (difference/total)**.5 < .02
    assert any('tag_activation_checkpoint' in code for code in graphs)
