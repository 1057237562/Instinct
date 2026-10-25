"""Compilation budgets and measured training kernel selection.

Import/call after the entry point's datasets-before-torch initialization.
Limits govern lazy specializations, not optimizer steps or total GPU kernels.
"""

import os

import torch


def training_compile_kwargs(model, *, mode='default'):
    """Keep V2 tensorwise FP8 GEMMs on cuBLAS with the validated local stack.

    Inductor's default autotuner can select slower Triton FP8 GEMMs for V2's
    1536/5504-wide projections. This is a per-compiled-model option: it does
    not mutate global Inductor settings, precision, attention, or Muon kernels.
    Explicit PyTorch backend settings take precedence over automatic selection.
    """
    setting = os.environ.get('INSTINCT_V2_GEMM_BACKEND', 'auto').strip().lower()
    if setting not in ('auto', 'aten', 'default'):
        raise ValueError('INSTINCT_V2_GEMM_BACKEND must be auto, aten or default')
    if setting == 'default' or (setting == 'auto' and
            'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS' in os.environ):
        return {'mode': mode}
    if setting == 'auto':
        config = getattr(model, 'config', None)
        dimensions = dict(hidden_size=1536, intermediate_size=5504, num_hidden_layers=32,
                          num_attention_heads=16, num_key_value_heads=4, head_dim=96)
        if (mode not in ('max-autotune', 'max-autotune-no-cudagraphs')
                or getattr(model, '_instinct_fp8_recipe', None) != 'tensorwise'
                or getattr(model, '_instinct_fp8_filter', None) != 'auto'
                or getattr(config, 'param_dtype', None) != 'bf16'
                or getattr(config, 'model_architecture', 'standard') != 'standard'
                or getattr(config, 'use_moe', False)
                or getattr(config, 'residual_type', 'standard') != 'standard'
                or any(getattr(config, key, None) != value for key, value in dimensions.items())):
            return {'mode': mode}
        from model.cuda_fastpath import blackwell_training_kernels
        if not blackwell_training_kernels(next(model.parameters()).device):
            return {'mode': mode}
    # torch.compile does not allow both mode and options. Expand the mode's
    # options so graph capture and every other mode-specific setting survive.
    from torch._inductor import list_mode_options
    options = dict(list_mode_options(mode))
    options['max_autotune_gemm_backends'] = 'ATEN'
    print('[Compile GEMMs] V2 tensorwise FP8: ATEN/cuBLAS; '
          'INSTINCT_V2_GEMM_BACKEND=default restores compiler selection.', flush=True)
    return {'options': options}


def _positive_budget(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer, got {raw!r}') from exc
    if value < 1:
        raise ValueError(f'{name} must be a positive integer, got {raw!r}')
    return value


def configure_compile_limits():
    """Set both Dynamo limits once at setup, before any compiled calls.

    Accumulated counts include different instances/regions sharing a code
    object and compilation attempts after invalidation, not just live entries.
    A 32-layer trunk can therefore need more accumulated entries than one gate.
    No per-batch patch, cache reset, eager fallback or shape-policy change.
    """
    per_region = _positive_budget('INSTINCT_COMPILE_RECOMPILE_LIMIT', 128)
    accumulated = _positive_budget('INSTINCT_COMPILE_ACCUMULATED_LIMIT', 4096)
    if accumulated < per_region:
        raise ValueError('INSTINCT_COMPILE_ACCUMULATED_LIMIT must be >= INSTINCT_COMPILE_RECOMPILE_LIMIT')
    config = torch._dynamo.config
    def key(modern, legacy):
        return modern if hasattr(config, modern) else legacy
    changes = {
        key('recompile_limit', 'cache_size_limit'): per_region,
        key('accumulated_recompile_limit', 'accumulated_cache_size_limit'): accumulated,
        key('fail_on_recompile_limit_hit', 'fail_on_cache_limit_hit'): True,
        'suppress_errors': False,
    }
    changed = any(getattr(config, name) != value for name, value in changes.items())
    for name, value in changes.items():
        setattr(config, name, value)
    if changed and os.environ.get('RANK', '0') == '0':
        print(
            f'[Compile limits] per-region={per_region}, accumulated={accumulated}; '
            'shape policy unchanged, limit hits raise instead of silently falling back to eager. '
            'Override with INSTINCT_COMPILE_RECOMPILE_LIMIT / INSTINCT_COMPILE_ACCUMULATED_LIMIT.',
            flush=True,
        )
    return per_region, accumulated
