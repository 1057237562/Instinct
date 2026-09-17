from datasets import load_dataset  # noqa: F401
import pytest
import torch

from model.inference_runtime import FusedKernel, rms_kernel, swiglu_kernel


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
