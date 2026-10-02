"""Low-overhead MoE router utilization reporting for training loops."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist


def _display_name(name: str) -> str:
    """Remove wrappers added by DDP/torch.compile from a module name."""
    parts = [part for part in name.split(".") if part not in {"module", "_orig_mod"}]
    return ".".join(parts) or "<root>"


@dataclass(frozen=True)
class MoeRoutingStats:
    """One logging-step snapshot of per-layer expert route fractions."""

    layer_names: tuple[str, ...]
    layer_loads: torch.Tensor  # CPU float32, [layers, experts], each row sums to 1

    @property
    def num_layers(self) -> int:
        return int(self.layer_loads.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.layer_loads.shape[1])

    @property
    def global_load(self) -> torch.Tensor:
        return self.layer_loads.mean(dim=0)

    def _entropy(self, loads: torch.Tensor) -> torch.Tensor:
        if self.num_experts <= 1:
            return torch.ones(loads.shape[:-1])
        terms = torch.where(loads > 0, loads * loads.clamp_min(1e-30).log(), 0.0)
        return -terms.sum(dim=-1) / math.log(self.num_experts)

    def metrics(self) -> dict[str, float]:
        """Return compact SwanLab/W&B metrics without per-layer metric spam."""
        global_load = self.global_load
        layer_entropy = self._entropy(self.layer_loads)
        layer_max, _ = self.layer_loads.max(dim=1)
        worst_layer = int(layer_max.argmax().item())
        ideal = 1.0 / self.num_experts
        metrics = {
            "moe/active_expert_rate": float((global_load > 0).float().mean().item()),
            "moe/inactive_experts": float((global_load == 0).sum().item()),
            "moe/min_route_pct": float(global_load.min().item() * 100.0),
            "moe/max_route_pct": float(global_load.max().item() * 100.0),
            "moe/load_cv": float(global_load.std(unbiased=False).item() / ideal),
            "moe/normalized_entropy": float(self._entropy(global_load).item()),
            "moe/inactive_layer_expert_pairs": float((self.layer_loads == 0).sum().item()),
            "moe/worst_layer_max_route_pct": float(layer_max[worst_layer].item() * 100.0),
            "moe/worst_layer_entropy": float(layer_entropy[worst_layer].item()),
        }
        metrics.update({
            f"moe/expert_{expert_id}_route_pct": float(load.item() * 100.0)
            for expert_id, load in enumerate(global_load)
        })
        return metrics

    def format_line(self) -> str:
        global_load = self.global_load
        layer_max, layer_top = self.layer_loads.max(dim=1)
        worst = int(layer_max.argmax().item())
        active = int((global_load > 0).sum().item())
        inactive_pairs = int((self.layer_loads == 0).sum().item())
        entropy = float(self._entropy(global_load).item())
        route_text = ", ".join(
            f"E{i}={value.item() * 100.0:.1f}%"
            for i, value in enumerate(global_load)
        )
        warning = (
            inactive_pairs > 0
            or float(layer_max[worst].item()) >= max(0.5, 4.0 / self.num_experts)
        )
        status = " WARNING: routing imbalance" if warning else ""
        return (
            f"[MoE Router]{status} active={active}/{self.num_experts}, "
            f"routes=[{route_text}], entropy={entropy:.3f}, "
            f"inactive_layer_expert_pairs={inactive_pairs}, "
            f"worst={self.layer_names[worst]}:E{int(layer_top[worst].item())}="
            f"{layer_max[worst].item() * 100.0:.1f}%"
        )


@torch.no_grad()
def collect_moe_routing_stats(model: torch.nn.Module) -> MoeRoutingStats | None:
    """Collect the most recent router load from every executed MoE module.

    Each MoE block stores an already-computed, detached expert-load vector while
    evaluating its load-balancing loss.  This function only stacks those tiny
    vectors.  In DDP it performs one all-reduce per logging event, then one
    device-to-host copy for the complete layer-by-expert matrix.
    """
    names: list[str] = []
    loads: list[torch.Tensor] = []
    seen_loads: set[int] = set()
    for name, module in model.named_modules():
        load = getattr(module, "router_load", None)
        # OptimizedModule proxies attributes from ``_orig_mod``.  When a bare
        # MoE block itself is compiled, both wrapper and original therefore
        # expose the exact same snapshot; count that tensor only once.
        if (
            isinstance(load, torch.Tensor)
            and load.ndim == 1
            and load.numel() > 0
            and id(load) not in seen_loads
        ):
            seen_loads.add(id(load))
            names.append(_display_name(name))
            loads.append(load.detach().float())

    if not loads:
        return None
    expert_counts = {int(load.numel()) for load in loads}
    if len(expert_counts) != 1:
        raise RuntimeError(
            f"cannot summarize MoE modules with different expert counts: {sorted(expert_counts)}"
        )

    layer_loads = torch.stack(loads)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(layer_loads, op=dist.ReduceOp.SUM)
        layer_loads.div_(dist.get_world_size())
    return MoeRoutingStats(tuple(names), layer_loads.cpu())
