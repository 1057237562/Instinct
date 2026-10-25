"""Keep the measured V2 GEMM policy scoped and honor explicit compiler settings."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.compile_policy import training_compile_kwargs


@pytest.fixture
def v2(monkeypatch):
    from model import cuda_fastpath
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: True)
    monkeypatch.delenv('INSTINCT_V2_GEMM_BACKEND', raising=False)
    monkeypatch.delenv('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS', raising=False)
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(hidden_size=1536, intermediate_size=5504, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96,
        use_moe=False, residual_type='standard', param_dtype='bf16')
    model._instinct_fp8_recipe = 'tensorwise'
    model._instinct_fp8_filter = 'auto'
    return model


@pytest.mark.parametrize('mode', ['max-autotune', 'max-autotune-no-cudagraphs'])
def test_v2_options_preserve_compile_mode_without_global_mutation(v2, mode):
    from torch._inductor import config, list_mode_options
    before = config.max_autotune_gemm_backends
    kwargs = training_compile_kwargs(v2, mode=mode)
    assert kwargs == {'options': {**list_mode_options(mode), 'max_autotune_gemm_backends': 'ATEN'}}
    assert config.max_autotune_gemm_backends == before


@pytest.mark.parametrize('field,value', [('hidden_size', 512), ('head_dim', 32),
    ('num_hidden_layers', 8), ('use_moe', True), ('residual_type', 'attnres'),
    ('param_dtype', 'fp32'), ('model_architecture', 'linear')])
def test_other_models_keep_existing_backend(v2, field, value):
    setattr(v2.config, field, value)
    assert training_compile_kwargs(v2, mode='max-autotune') == {'mode': 'max-autotune'}


@pytest.mark.parametrize('recipe,filter_mode', [('off', 'auto'), ('rowwise', 'auto'), ('tensorwise', 'eligible')])
def test_other_precision_paths_keep_existing_backend(v2, recipe, filter_mode):
    v2._instinct_fp8_recipe, v2._instinct_fp8_filter = recipe, filter_mode
    assert training_compile_kwargs(v2, mode='max-autotune') == {'mode': 'max-autotune'}


def test_explicit_overrides_and_unvalidated_hardware(v2, monkeypatch):
    from model import cuda_fastpath
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: False)
    assert training_compile_kwargs(v2, mode='max-autotune') == {'mode': 'max-autotune'}
    monkeypatch.setattr(cuda_fastpath, 'blackwell_training_kernels', lambda device: True)
    monkeypatch.setenv('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS', 'TRITON')
    assert training_compile_kwargs(v2, mode='max-autotune') == {'mode': 'max-autotune'}
    monkeypatch.delenv('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS')
    monkeypatch.setenv('INSTINCT_V2_GEMM_BACKEND', 'default')
    assert training_compile_kwargs(v2, mode='max-autotune') == {'mode': 'max-autotune'}
    monkeypatch.setenv('INSTINCT_V2_GEMM_BACKEND', 'invalid')
    with pytest.raises(ValueError, match='INSTINCT_V2_GEMM_BACKEND'):
        training_compile_kwargs(v2)


def test_every_compiled_trainer_selects_options_for_its_actual_model():
    root = Path(__file__).resolve().parents[1]
    calls = 0
    for path in (root / 'trainer').glob('train_*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func) == 'torch.compile'):
                options = [kw.value for kw in node.keywords if kw.arg is None]
                assert len(options) == 1, path
                assert ast.unparse(options[0].func) == 'training_compile_kwargs', path
                assert ast.unparse(options[0].args[0]) == ast.unparse(node.args[0]), path
                assert any(kw.arg == 'dynamic' for kw in node.keywords), path
                calls += 1
    assert calls == 8
