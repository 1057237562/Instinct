from datasets import load_dataset  # noqa: F401
import pytest
import torch
import torch.nn as nn

from model.inference_runtime import (
    CompiledTrunk,
    FusedKernel,
    _trunk_compile_eligible,
    rms_kernel,
    select_inference_dtype,
    swiglu_kernel,
)


def test_compiler_fallback_is_one_time_and_preserves_results():
    calls = []
    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('compiler unavailable')
    kernel = FusedKernel(swiglu_kernel, compiler=broken)
    a, b = torch.randn(2, 4), torch.randn(2, 4)
    expected = torch.nn.functional.silu(a) * b
    torch.testing.assert_close(kernel(a, b), expected)
    torch.testing.assert_close(kernel(a, b), expected)
    assert calls == [1]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_norm_formula_and_inference_dispatch(dtype):
    from model.model_instinct import RMSNorm
    layer = RMSNorm(32).to(dtype).eval()
    x = torch.randn(2, 3, 32).to(dtype)
    expected = layer(x)
    layer._inference_norm = rms_kernel
    torch.testing.assert_close(layer(x), expected)
    layer.train()
    layer._inference_norm = lambda *args: pytest.fail('Training must stay on original path')
    torch.testing.assert_close(layer(x), expected)


def test_select_inference_dtype_prefers_bf16_on_ampere_plus(monkeypatch):
    class FakeCapability:
        def __init__(self, major):
            self.major = major
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability',
                        lambda device=None: (FakeCapability.MAJOR, 0))
    FakeCapability.MAJOR = 8
    assert select_inference_dtype('cuda') == torch.bfloat16
    FakeCapability.MAJOR = 7
    assert select_inference_dtype('cuda') == torch.float16
    assert select_inference_dtype('cpu') == torch.float16


def test_trunk_compile_eligible_only_for_standard_topology():
    from model.model_instinct import InstinctConfig
    assert _trunk_compile_eligible(InstinctConfig(model_architecture='standard'))
    assert not _trunk_compile_eligible(InstinctConfig(model_architecture='looped'))
    assert not _trunk_compile_eligible(InstinctConfig(residual_type='mhc'))


def test_compiled_trunk_routes_diagnostic_kwargs_to_eager_module():
    """Per-call callbacks must never enter Dynamo; plain calls use the compiled path."""
    calls = {'compiled': 0, 'eager': 0}

    class Fake(nn.Module):
        def forward(self, x, **kwargs):
            return x * 2

    def compiled(x, **kwargs):
        calls['compiled'] += 1
        return x * 2

    trunk = CompiledTrunk(compiled, Fake())
    x = torch.randn(2, 2)
    torch.testing.assert_close(trunk(x), x * 2)
    assert calls['compiled'] == 1
    trunk(x, layer_callback=lambda *a: None)
    trunk(x, exit_check_fn=lambda *a: False)
    trunk(x, return_intermediate=True)
    assert calls['compiled'] == 1  # all three went to the eager module


@pytest.mark.gpu
def test_full_mode_compiles_trunk_and_matches_eager_generation():
    """End to end on CUDA: optimize_inference(mode='full') keeps greedy output."""
    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    torch.manual_seed(3)
    config = InstinctConfig(hidden_size=64, num_hidden_layers=2, vocab_size=128,
                            num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, use_moe=True)
    model = InstinctForCausalLM(config).cuda().to(torch.bfloat16).eval()
    eager = InstinctForCausalLM(config).cuda().to(torch.bfloat16).eval()
    eager.load_state_dict(model.state_dict())

    prompt = torch.randint(3, 100, (1, 6), device='cuda')
    with torch.inference_mode():
        expected = eager.generate(input_ids=prompt, max_new_tokens=4,
                                  do_sample=False, eos_token_id=None)

        from model.inference_runtime import optimize_inference
        optimized = optimize_inference(model, 'full')
        from model.inference_runtime import CompiledTrunk as Trunk
        assert isinstance(optimized.model, Trunk)
        actual = optimized.generate(input_ids=prompt, max_new_tokens=4,
                                    do_sample=False, eos_token_id=None)

    torch.testing.assert_close(actual, expected)
