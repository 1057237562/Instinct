"""Compilation budgets for repeated training buckets, without kernel changes.

Import/call after the entry point's datasets-before-torch initialization.
Limits govern lazy specializations, not optimizer steps or total GPU kernels.
"""

import os

import torch


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
