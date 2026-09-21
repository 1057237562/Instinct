"""MoE routing utilization snapshots and reporting."""

import pytest
import torch

from trainer.moe_monitor import MoeRoutingStats, collect_moe_routing_stats
from tests.helpers import make_tiny_config


@pytest.mark.parametrize(
    "variant,module_name",
    [
        ("dense", "model.model_instinct"),
        ("loop", "model.model_instinct_loop"),
        ("linear", "model.model_instinct_linear"),
    ],
)
def test_moe_forward_records_normalized_detached_load(variant, module_name):
    module = __import__(module_name, fromlist=["MOEFeedForward"])
    config = make_tiny_config(use_moe=True, variant=variant)
    moe = module.MOEFeedForward(config).train()
    x = torch.randn(2, 5, config.hidden_size, requires_grad=True)

    output = moe(x)

    assert output.shape == x.shape
    assert moe.router_load.shape == (config.num_experts,)
    assert not moe.router_load.requires_grad
    torch.testing.assert_close(moe.router_load.sum(), torch.tensor(1.0))
    assert torch.count_nonzero(moe.router_load).item() >= 1


def test_collect_summarizes_layers_and_detects_collapsed_layer():
    class FakeMoe(torch.nn.Module):
        def __init__(self, load):
            super().__init__()
            self.router_load = torch.tensor(load)

    model = torch.nn.ModuleList([
        FakeMoe([0.25, 0.25, 0.25, 0.25]),
        FakeMoe([1.0, 0.0, 0.0, 0.0]),
    ])

    stats = collect_moe_routing_stats(model)

    assert stats is not None
    assert stats.num_layers == 2
    assert stats.num_experts == 4
    torch.testing.assert_close(
        stats.global_load, torch.tensor([0.625, 0.125, 0.125, 0.125])
    )
    metrics = stats.metrics()
    assert metrics["moe/active_expert_rate"] == 1.0
    assert metrics["moe/inactive_layer_expert_pairs"] == 3.0
    assert metrics["moe/worst_layer_max_route_pct"] == 100.0
    line = stats.format_line()
    assert "WARNING: routing imbalance" in line
    assert "active=4/4" in line
    assert "E0=62.5%" in line


def test_balanced_snapshot_has_expected_metrics_and_no_warning():
    loads = torch.full((3, 8), 1.0 / 8.0)
    stats = MoeRoutingStats(("layer0", "layer1", "layer2"), loads)

    metrics = stats.metrics()

    assert metrics["moe/active_expert_rate"] == 1.0
    assert metrics["moe/load_cv"] == 0.0
    assert metrics["moe/normalized_entropy"] == pytest.approx(1.0)
    assert "WARNING" not in stats.format_line()


def test_collect_returns_none_without_moe_snapshots():
    assert collect_moe_routing_stats(torch.nn.Linear(4, 4)) is None


def test_collect_deduplicates_compile_style_attribute_proxy():
    class FakeMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router_load = torch.tensor([0.5, 0.5])

    class Proxy(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original
            self.router_load = original.router_load

    stats = collect_moe_routing_stats(Proxy(FakeMoe()))

    assert stats is not None
    assert stats.num_layers == 1
