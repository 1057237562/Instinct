"""Exercise real Dynamo limit accounting on CPU, without Inductor/GPU cost."""

from types import SimpleNamespace

import pytest
import torch

from model.compile_policy import configure_compile_limits


@pytest.fixture(autouse=True)
def isolated_limits(monkeypatch):
    for name in ('INSTINCT_COMPILE_RECOMPILE_LIMIT', 'INSTINCT_COMPILE_ACCUMULATED_LIMIT'):
        monkeypatch.delenv(name, raising=False)
    torch.compiler.reset()
    with torch._dynamo.config.patch(
        cache_size_limit=8, accumulated_cache_size_limit=256,
        fail_on_recompile_limit_hit=False, suppress_errors=False,
    ):
        yield
    torch.compiler.reset()


def test_defaults_set_both_limits_without_changing_shape_policy():
    before = (torch._dynamo.config.automatic_dynamic_shapes,
              torch._dynamo.config.assume_static_by_default)
    assert configure_compile_limits() == (128, 4096)
    assert torch._dynamo.config.cache_size_limit == 128
    assert torch._dynamo.config.accumulated_cache_size_limit == 4096
    assert torch._dynamo.config.fail_on_recompile_limit_hit is True
    assert before == (torch._dynamo.config.automatic_dynamic_shapes,
                      torch._dynamo.config.assume_static_by_default)


def test_legacy_config_names_supported(monkeypatch):
    legacy = SimpleNamespace(cache_size_limit=8, accumulated_cache_size_limit=256,
                             fail_on_cache_limit_hit=False, suppress_errors=False)
    monkeypatch.setattr(torch._dynamo, 'config', legacy)
    configure_compile_limits()
    assert (legacy.cache_size_limit, legacy.accumulated_cache_size_limit) == (128, 4096)
    assert legacy.fail_on_cache_limit_hit is True


@pytest.mark.parametrize('name,value', [
    ('INSTINCT_COMPILE_RECOMPILE_LIMIT', '0'),
    ('INSTINCT_COMPILE_RECOMPILE_LIMIT', 'unlimited'),
    ('INSTINCT_COMPILE_ACCUMULATED_LIMIT', '-1'),
    ('INSTINCT_COMPILE_ACCUMULATED_LIMIT', '16'),
])
def test_bad_budgets_fail_before_mutating_config(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        configure_compile_limits()
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 256


def test_more_than_256_specializations_really_compile(monkeypatch):
    # The old per-call-only fix can bypass 8/128 but still fails at 256.
    monkeypatch.setenv('INSTINCT_COMPILE_RECOMPILE_LIMIT', '512')
    configure_compile_limits()
    compilations = []
    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward
    def fn(x, index):
        return x + index
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
    x = torch.ones(2)
    for index in range(260):
        torch.testing.assert_close(compiled(x, index), x + index)
    assert len(compilations) == 260  # no hidden eager fallback
    torch.testing.assert_close(compiled(x, 259), x + 259)
    assert len(compilations) == 260  # retain and reuse the warm cache


def test_trunk_limit_hit_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv('INSTINCT_COMPILE_RECOMPILE_LIMIT', '2')
    monkeypatch.setenv('INSTINCT_COMPILE_ACCUMULATED_LIMIT', '4')
    configure_compile_limits()
    compiled = torch.compile(lambda x, index: x + index, backend='eager', dynamic=False)
    compiled(torch.ones(2), 0)
    compiled(torch.ones(2), 1)
    with pytest.raises(Exception, match='[Rr]ecompile|[Cc]ache|[Ll]imit'):
        compiled(torch.ones(2), 2)
