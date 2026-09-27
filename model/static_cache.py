"""Preallocated, in-place KV storage for single-sequence inference.

The default cache in ``Attention`` grows with ``torch.cat`` and, for quantized
configs, re-quantizes the whole sequence on every decoded token. Both make the
cache length part of the graph's shape, which is why the decode step cannot be
captured as a CUDA graph: Inductor would need a fresh capture for every length.

A ``StaticKVCache`` fixes the shape for the whole generation instead. Each step
writes its slice in place with ``index_copy_`` and attention reads the full
buffer through a mask built from the current position, so every decode step has
identical shapes and no host-derived offsets.
"""
from __future__ import annotations

from typing import List, Sequence

import torch


class StaticKVCacheLayer:
    """One layer's buffers, shaped for ``Attention``'s ``(k, v)`` convention."""

    __slots__ = ("key", "value")

    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        self.key = key
        self.value = value

    def append(self, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor):
        """Write one step's keys/values in place and return the whole buffer.

        ``index_copy_`` rather than a slice assignment: the offset has to be a
        device tensor for the step to stay capturable. ``positions`` arrives as
        ``[1, seq]`` from the model and covers the single cached row.
        """
        positions = positions.reshape(-1)
        self.key.index_copy_(1, positions, keys)
        self.value.index_copy_(1, positions, values)
        return self.key, self.value


class StaticKVCache:
    """Fixed-capacity KV storage shared by every layer of one generation.

    Only the length is static, not the contents: ``key_positions`` enumerates the
    buffer so a mask can hide the untouched tail.
    """

    def __init__(self, max_cache_len: int, layer_specs: Sequence, dtype, device):
        self.max_cache_len = max_cache_len
        self.key_positions = torch.arange(max_cache_len, device=device)
        self.layers: List[StaticKVCacheLayer] = [
            StaticKVCacheLayer(
                torch.zeros(1, max_cache_len, num_kv_heads, head_dim, dtype=dtype, device=device),
                torch.zeros(1, max_cache_len, num_kv_heads, head_dim, dtype=dtype, device=device),
            )
            for num_kv_heads, head_dim in layer_specs
        ]
        self._mark_static()

    def _mark_static(self) -> None:
        """Tell Inductor these buffers keep their address and their contents.

        Without it a captured graph would treat the cache as an ordinary input:
        the in-place writes would land in the pool copy and be discarded, losing
        the context between steps.
        """
        marker = getattr(torch._dynamo, "mark_static_address", None)
        if marker is None:
            return
        for layer in self.layers:
            marker(layer.key)
            marker(layer.value)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> StaticKVCacheLayer:
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def mask_for(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Bool attention mask for the current step.

        A key is visible when its cache slot is not past the query's own
        position, which is causal for a prompt and length-limiting for decode.
        Shape is ``[batch, 1, query, max_cache_len]`` on every step.
        """
        return self.key_positions[None, None, None, :] <= position_ids[:, None, :, None]


def build_static_cache(model, max_cache_len: int, dtype=None, device=None) -> StaticKVCache:
    """Allocate a cache sized for ``max_cache_len`` on the model's device.

    Unwraps ``inference_runtime.CompiledTrunk``: the loader warms the compiled
    model up with a real ``generate`` call, which lands here while ``model.model``
    is still the wrapper.
    """
    parameters = next(model.parameters())
    dtype = dtype or parameters.dtype
    device = device or parameters.device
    return StaticKVCache(max_cache_len, layer_specs(model), dtype, device)


def layer_specs(model):
    """``(kv_heads, head_dim)`` per layer, read from the unwrapped trunk."""
    trunk = getattr(model.model, "original", model.model)
    return [
        (block.self_attn.n_local_kv_heads, block.self_attn.head_dim)
        for block in trunk.layers
    ]


_CAPACITY_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384, 32768)

# Reserve enough room for the common short/medium answer without sizing every
# decode step for the UI's full max_new_tokens allowance.  Generation grows to
# the next bucket only when it actually reaches the current capacity.
STATIC_CACHE_INITIAL_DECODE_TOKENS = 256


def bucket_capacity(needed: int, limit: int) -> int | None:
    """Round a required length up to a reusable capacity, or None if too long.

    Reusing one allocation across turns is what keeps the compiled graph and the
    captured decode step valid: a fresh cache means fresh buffer addresses, so
    ``torch.compile`` recompiles and the CUDA graph has to be recorded again.
    """
    for size in _CAPACITY_BUCKETS:
        if size >= needed:
            return size if size <= limit else None
    return None


class DecodeState:
    """Reusable decode resources: the cache, its step buffers and the graph.

    Held by the model across ``generate`` calls. Contents are per-sequence
    (positions restart at 0) and the mask hides any slot past the current
    position, so a later, shorter prompt can reuse the same buffers safely.
    """

    def __init__(self, capacity: int, specs, dtype, device):
        self.cache = StaticKVCache(capacity, specs, dtype, device)
        self.step_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.step_position = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.graph = None          # (CUDAGraph, outputs) once recorded
        self.capture_failed = False

    @property
    def capacity(self) -> int:
        return self.cache.max_cache_len

    def accepts(self, needed: int) -> bool:
        return self.cache.max_cache_len >= needed
