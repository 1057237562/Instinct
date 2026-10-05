"""Preallocated, in-place KV storage for single-sequence and batched inference.

The default cache in ``Attention`` grows with ``torch.cat`` and, for quantized
configs, re-quantizes the whole sequence on every decoded token. Both make the
cache length part of the graph's shape, which is why the decode step cannot be
captured as a CUDA graph: Inductor would need a fresh capture for every length.

A ``StaticKVCache`` fixes the shape for the whole generation instead. Each step
writes its slice in place and attention reads the full buffer through a mask
built from the current position, so every decode step has identical shapes and
no host-derived offsets. Buffers carry a batch dimension, so a left-padded
batch gets the same treatment as one sequence: every row's prompt ends at the
same storage column, the step's slot is uniform across the batch, and the only
per-row state is the ``pad_starts`` offset that the mask folds in.
"""
from __future__ import annotations

import os
from typing import List, Sequence

import torch


class StaticKVCacheLayer:
    """One layer's buffers, shaped for ``Attention``'s ``(k, v)`` convention."""

    __slots__ = ("key", "value", "row_index")

    def __init__(self, key: torch.Tensor, value: torch.Tensor, row_index: torch.Tensor):
        self.key = key
        self.value = value
        self.row_index = row_index

    def append(self, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor):
        """Write one step's keys/values in place and return the whole buffer.

        Device-tensor indexing rather than a slice assignment: the offset has to
        live on the GPU for the step to stay capturable. ``positions`` holds the
        physical cache slot of every ``[batch, seq]`` write and ``row_index``
        pairs each slot with its batch row, so one call writes the whole batch.
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        rows = self.row_index.expand_as(positions)
        self.key[rows, positions] = keys
        self.value[rows, positions] = values
        return self.key, self.value


class StaticKVCache:
    """Fixed-capacity KV storage shared by every layer of one generation.

    Only the length is static, not the contents: ``key_positions`` enumerates
    the buffer so a mask can hide the untouched tail, and ``pad_starts`` hides
    the pad prefix a left-padded batch wrote into the same slots.
    """

    def __init__(self, max_cache_len: int, layer_specs: Sequence, dtype, device,
                 batch_size: int = 1):
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.key_positions = torch.arange(max_cache_len, device=device)
        # Physical slot of each row's first real token. Zero for unpadded use;
        # a left-padded prompt writes its pad prefix into the buffer and the
        # mask drops those slots through this offset instead.
        self.pad_starts = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        row_index = torch.arange(batch_size, device=device).view(batch_size, 1)
        self.layers: List[StaticKVCacheLayer] = [
            StaticKVCacheLayer(
                torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim,
                            dtype=dtype, device=device),
                torch.zeros(batch_size, max_cache_len, num_kv_heads, head_dim,
                            dtype=dtype, device=device),
                row_index,
            )
            for num_kv_heads, head_dim in layer_specs
        ]
        self._mark_static()

    def _mark_static(self) -> None:
        """Tell Inductor these buffers keep their address and their contents.

        Without it a captured graph would treat the cache as an ordinary input:
        the in-place writes would land in the pool copy and be discarded, losing
        the context between steps. ``pad_starts`` is rewritten per generation,
        so a replay must read its live value too.
        """
        marker = getattr(torch._dynamo, "mark_static_address", None)
        if marker is None:
            return
        for layer in self.layers:
            marker(layer.key)
            marker(layer.value)
        marker(self.pad_starts)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> StaticKVCacheLayer:
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def mask_for(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Bool attention mask for the current step.

        A key is visible when its cache slot is not past the query's own
        position — causal for a prompt, length-limiting for decode — and not
        inside the row's pad prefix. Shape is ``[batch, 1, query,
        max_cache_len]`` on every step.
        """
        slots = self.key_positions[None, None, None, :]
        mask = slots <= position_ids[:, None, :, None]
        return mask & (slots >= self.pad_starts[:, None, None, :])


def build_static_cache(model, max_cache_len: int, dtype=None, device=None,
                       batch_size: int = 1) -> StaticKVCache:
    """Allocate a cache sized for ``max_cache_len`` on the model's device.

    Unwraps ``inference_runtime.CompiledTrunk``: the loader warms the compiled
    model up with a real ``generate`` call, which lands here while ``model.model``
    is still the wrapper.
    """
    parameters = next(model.parameters())
    dtype = dtype or parameters.dtype
    device = device or parameters.device
    return StaticKVCache(max_cache_len, layer_specs(model), dtype, device, batch_size)


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


def batch_capacity_guard(batch_size: int) -> float:
    """Share of free VRAM a new static bucket may claim.

    A batched state serves a short-lived eval batch and the pool holds at most
    two capacities per batch size, so it can budget more aggressively than the
    desktop single-sequence pool, whose states live for the whole session.
    """
    return 0.6 if batch_size > 1 else 0.25


def batched_static_decode_enabled() -> bool:
    """Kill switch for the batched fast path; single-sequence decoding is unaffected."""
    return os.environ.get('INSTINCT_BATCH_STATIC_DECODE', '1') != '0'


def left_pad_starts(attention_mask: torch.Tensor):
    """Per-row width of the left pad prefix, and whether padding is left-only.

    The decode step appends at the sequence end, so only a zeros-then-ones mask
    is admissible: right padding or interior holes would put real tokens after
    the append slot. One device-to-host read for the whole batch.
    """
    mask = attention_mask.to(torch.bool)
    pad_starts = (mask.shape[1] - mask.sum(-1, keepdim=True)).to(torch.long)
    leaks = (~mask & (mask.cumsum(-1) > 0)).any() | (pad_starts >= mask.shape[1]).any()
    return pad_starts, not bool(leaks)


class DecodeState:
    """Reusable decode resources: the cache, its step buffers and the graph.

    Held by the model across ``generate`` calls, pooled per batch size. Contents
    are per-generation (positions restart at 0, ``pad_starts`` is rewritten) and
    the mask hides any slot past the current position or inside the pad prefix,
    so a later, shorter prompt can reuse the same buffers safely.
    """

    def __init__(self, capacity: int, specs, dtype, device, batch_size: int = 1):
        self.batch_size = batch_size
        self.cache = StaticKVCache(capacity, specs, dtype, device, batch_size)
        self.step_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.step_position = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.graph = None          # (CUDAGraph, outputs) once recorded
        self.capture_failed = False

    @property
    def capacity(self) -> int:
        return self.cache.max_cache_len

    def accepts(self, needed: int) -> bool:
        return self.cache.max_cache_len >= needed
