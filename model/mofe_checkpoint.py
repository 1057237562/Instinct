"""Manifest-driven assembly and compact checkpoint I/O for MoFE models."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch


FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class FrozenExpert:
    name: str
    path: str
    domain: str = ""


@dataclass(frozen=True)
class MoFEManifest:
    base_model: str
    experts: tuple[FrozenExpert, ...]
    source: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "experts": [expert.__dict__.copy() for expert in self.experts],
            "source": self.source,
            "fingerprint": self.fingerprint,
        }


def _absolute(path: str, root: str) -> str:
    path = os.path.expandvars(os.path.expanduser(path.strip()))
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(root, path))


def load_manifest(path: str) -> MoFEManifest:
    manifest_path = os.path.abspath(path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    root = os.path.dirname(manifest_path)
    base = payload.get("base_model") or payload.get("shared_base")
    if not isinstance(base, str) or not base.strip():
        raise ValueError("MoFE manifest requires a non-empty 'base_model' path")
    raw_experts = payload.get("experts")
    if not isinstance(raw_experts, list) or not raw_experts:
        raise ValueError("MoFE manifest requires a non-empty 'experts' list")

    experts = []
    for index, item in enumerate(raw_experts):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"expert {index} must be a path string or an object with 'path'")
        experts.append(FrozenExpert(
            name=str(item.get("name") or f"expert_{index:02d}"),
            path=_absolute(item["path"], root),
            domain=str(item.get("domain") or ""),
        ))
    resolved_base = _absolute(base, root)
    missing = [candidate for candidate in [resolved_base, *(e.path for e in experts)]
               if not os.path.isfile(candidate)]
    if missing:
        raise FileNotFoundError("Missing MoFE checkpoint(s): " + ", ".join(missing))
    def identity(candidate: str) -> dict[str, Any]:
        stat = os.stat(candidate)
        return {
            "path": candidate,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    canonical = {
        "base_model": identity(resolved_base),
        "experts": [
            {**expert.__dict__, "file": identity(expert.path)} for expert in experts
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return MoFEManifest(resolved_base, tuple(experts), manifest_path, fingerprint)


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a state-dict-like mapping")
    for key in ("model", "state_dict"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload = nested
            break
    state = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        state[key] = value
    return state


def load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    return _unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=False))


def _is_ffn_key(key: str) -> bool:
    return ".mlp." in key and any(key.endswith(f".{projection}.weight") for projection in FFN_PROJECTIONS)


def preflight_manifest(manifest: MoFEManifest, expected_layers: int | None = None) -> dict[str, Any]:
    """Validate a dense checkpoint bank before allocating the combined MoFE."""
    base_state = load_checkpoint_state(manifest.base_model)
    layer_indices = sorted({
        int(key.split(".")[2])
        for key in base_state
        if key.startswith("model.layers.") and key.split(".")[2].isdigit()
    })
    if not layer_indices or layer_indices != list(range(layer_indices[-1] + 1)):
        raise ValueError("base checkpoint does not contain a contiguous Transformer layer stack")
    layer_count = len(layer_indices)
    if expected_layers is not None and layer_count != expected_layers:
        raise ValueError(f"base checkpoint has {layer_count} layers, expected {expected_layers}")

    reference_shapes = {}
    for layer_index in layer_indices:
        for projection in FFN_PROJECTIONS:
            key = f"model.layers.{layer_index}.mlp.{projection}.weight"
            if key not in base_state:
                raise KeyError(f"base checkpoint is missing {key}")
            reference_shapes[key] = tuple(base_state[key].shape)

    tensors = 0
    for expert in manifest.experts:
        state = load_checkpoint_state(expert.path)
        for key, shape in reference_shapes.items():
            if key not in state:
                raise KeyError(f"{expert.name} is missing {key}")
            if tuple(state[key].shape) != shape:
                raise ValueError(
                    f"{expert.name} has incompatible {key}: {tuple(state[key].shape)} != {shape}"
                )
            tensors += 1
    return {
        "ok": True,
        "experts": len(manifest.experts),
        "layers": layer_count,
        "validated_ffn_tensors": tensors,
        "fingerprint": manifest.fingerprint,
    }


@torch.no_grad()
def assemble_mofe(model, manifest: MoFEManifest) -> dict[str, Any]:
    """Load shared weights from base and one dense FFN per frozen expert/layer."""
    expected_experts = model.config.num_experts
    if len(manifest.experts) != expected_experts:
        raise ValueError(
            f"manifest has {len(manifest.experts)} experts, model expects {expected_experts}"
        )
    base_state = load_checkpoint_state(manifest.base_model)
    target_state = model.state_dict()
    required_shared = {key: value for key, value in target_state.items() if ".mlp." not in key}
    for key, target_value in required_shared.items():
        if key not in base_state:
            raise KeyError(f"base checkpoint is missing shared tensor {key}")
        if base_state[key].shape != target_value.shape:
            raise ValueError(
                f"shared tensor shape mismatch for {key}: "
                f"{tuple(base_state[key].shape)} != {tuple(target_value.shape)}"
            )
    shared_state = {key: value for key, value in base_state.items() if not _is_ffn_key(key)}
    incompatible = model.load_state_dict(shared_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if ".mlp." not in key]
    if unexpected:
        raise ValueError(f"unexpected shared checkpoint keys: {unexpected[:8]}")

    copied = 0
    for expert_index, expert in enumerate(manifest.experts):
        expert_state = load_checkpoint_state(expert.path)
        for layer_index in range(model.config.num_hidden_layers):
            for projection in FFN_PROJECTIONS:
                source = f"model.layers.{layer_index}.mlp.{projection}.weight"
                target = f"model.layers.{layer_index}.mlp.experts.{expert_index}.{projection}.weight"
                if source not in expert_state:
                    raise KeyError(f"{expert.path} is missing {source}")
                if target not in target_state:
                    raise KeyError(f"MoFE target is missing {target}")
                if expert_state[source].shape != target_state[target].shape:
                    raise ValueError(
                        f"shape mismatch for {expert.name} {source}: "
                        f"{tuple(expert_state[source].shape)} != {tuple(target_state[target].shape)}"
                    )
                target_state[target].copy_(expert_state[source].to(target_state[target].dtype))
                copied += 1
    model.load_state_dict(target_state, strict=False)
    for block in model.model.layers:
        block.mlp.experts.requires_grad_(False)
    return {
        "manifest": manifest.source,
        "fingerprint": manifest.fingerprint,
        "num_experts": len(manifest.experts),
        "copied_ffn_tensors": copied,
    }


def set_mofe_train_scope(model, scope: str) -> list[str]:
    """Freeze experts always; optionally train only routers or all shared weights."""
    if scope not in {"router_only", "router_shared"}:
        raise ValueError("train_scope must be router_only or router_shared")
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "router_shared")
    trainable = []
    for name, parameter in model.named_parameters():
        if ".mlp.experts." in name:
            parameter.requires_grad_(False)
        elif ".mlp.gate." in name or name.endswith(".mlp.gate.weight"):
            parameter.requires_grad_(True)
        if parameter.requires_grad:
            trainable.append(name)
    return trainable


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in trainable
    }


def load_trainable_state_dict(model, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    provided = set(state)
    missing = sorted(expected - provided)
    if missing:
        raise ValueError(f"MoFE delta is missing trainable keys: {missing[:8]}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected MoFE delta keys: {incompatible.unexpected_keys[:8]}")


def load_mofe_delta(path: str, device: str = "cpu"):
    """Reconstruct a usable MoFE model from its external expert bank and delta."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "instinct-mofe-delta-v1":
        raise ValueError(f"unsupported MoFE delta format in {path}")
    manifest_data = payload.get("manifest") or {}
    manifest_source = manifest_data.get("source")
    if not manifest_source:
        raise ValueError("MoFE delta does not identify its source manifest")
    manifest = load_manifest(manifest_source)
    if manifest.fingerprint != payload.get("manifest_fingerprint"):
        raise ValueError("frozen expert bank differs from the one used to train this delta")

    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    config = InstinctConfig(**payload["config"])
    model = InstinctForCausalLM(config)
    assemble_mofe(model, manifest)
    set_mofe_train_scope(model, payload.get("train_scope", "router_shared"))
    load_trainable_state_dict(model, payload["model_delta"])
    return model.to(device), payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an Instinct MoFE expert manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected_layers", type=int)
    cli_args = parser.parse_args()
    try:
        result = preflight_manifest(load_manifest(cli_args.manifest), cli_args.expected_layers)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)
