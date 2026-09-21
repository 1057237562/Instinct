"""Ragged expert GEMMs without PyTorch's repeated host-offset fallback.

Keep original expert parameters/checkpoint keys. A plan belongs to one routing
invocation and is reused by all three projections and their backward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=None)
def _auto_backend(device):
    if device.type != 'cuda' or torch.version.hip is not None:
        return 'cached'
    major, _ = torch.cuda.get_device_capability(device)
    # Match the actual native BF16 fast-path whitelist, not just hasattr().
    if sys.platform != 'win32' and major in (9, 10) and hasattr(F, 'grouped_mm'):
        return 'native'
    if major >= 8 and importlib.util.find_spec('triton') is not None:
        return 'triton'
    return 'cached'


@dataclass(frozen=True)
class GroupedMMPlan:
    offsets: torch.Tensor
    backend: str
    ends: tuple[int, ...] | None = None


def make_grouped_mm_plan(offsets, *, backend=None):
    if offsets.ndim != 1 or offsets.dtype != torch.int32:
        raise ValueError('grouped GEMM offsets must be a 1D int32 tensor')
    backend = backend or os.environ.get('INSTINCT_GROUPED_MM_BACKEND', 'auto')
    if backend == 'auto':
        backend = _auto_backend(offsets.device)
    if backend not in ('native', 'triton', 'cached'):
        raise ValueError('INSTINCT_GROUPED_MM_BACKEND must be auto, native, triton or cached')
    if backend == 'triton' and offsets.device.type != 'cuda':
        raise ValueError('Triton grouped GEMM requires CUDA offsets')
    # One readback in the portable path; no readbacks in the Triton path.
    ends = tuple(offsets.cpu().tolist()) if backend == 'cached' else None
    return GroupedMMPlan(offsets, backend, ends)


class _RaggedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, plan):
        ctx.save_for_backward(x, weight, plan.offsets)
        ctx.plan = plan
        if plan.backend == 'triton':
            from model.grouped_mm_triton import ragged_mm
            return ragged_mm(x, weight, plan.offsets)
        result = x.new_empty((x.size(0), weight.size(2)))
        start = 0
        for expert, end in enumerate(plan.ends):
            torch.mm(x[start:end], weight[expert], out=result[start:end])
            start = end
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, offsets = ctx.saved_tensors
        plan = ctx.plan
        dx = dw = None
        # These are BF16 GEMMs regardless of the surrounding activation context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if plan.backend == 'triton':
                from model.grouped_mm_triton import ragged_mm, ragged_weight_grad
                if ctx.needs_input_grad[0]:
                    dx = ragged_mm(grad, weight.transpose(1, 2), offsets)
                if ctx.needs_input_grad[1]:
                    dw = ragged_weight_grad(x, grad, offsets, weight.size(0))
            else:
                dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
                dw = torch.empty_like(weight) if ctx.needs_input_grad[1] else None
                start = 0
                for expert, end in enumerate(plan.ends):
                    if dx is not None:
                        torch.mm(grad[start:end], weight[expert].T, out=dx[start:end])
                    if dw is not None:
                        torch.mm(x[start:end].T, grad[start:end], out=dw[expert])
                    start = end
        return dx, dw, None


def expert_grouped_mm(x, weight, plan):
    """[tokens,K] @ [experts,K,N]; offsets are trusted sorted routing ends.

    Offsets must be monotone and end at x.size(0); the router guarantees this.
    Tensor metadata is validated without reading device contents. First-order
    autograd is supported; the native backend remains available for higher-order
    differentiation and reference comparisons.
    """
    if x.ndim != 2 or weight.ndim != 3 or x.size(1) != weight.size(1):
        raise ValueError('expected [tokens,K] and [experts,K,N]')
    if weight.size(0) != plan.offsets.numel():
        raise ValueError('expert count must match offsets')
    if x.device != weight.device or x.device != plan.offsets.device:
        raise ValueError('inputs and offsets must be on the same device')
    if x.dtype != weight.dtype:
        raise ValueError('grouped GEMM input dtypes must match')
    if plan.backend == 'native':
        return F.grouped_mm(x, weight, offs=plan.offsets)
    if plan.backend == 'triton' and x.dtype != torch.bfloat16:
        raise ValueError('Triton grouped GEMM currently requires BF16 inputs')
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _RaggedMM.apply(x, weight, plan)
