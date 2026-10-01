"""Block-sparse causal document attention with compact GQA K/V.

The optional FlexAttention path is isolated from the outer model compiler.
One block mask is built per batch and shared across every transformer layer.
"""

from functools import lru_cache
import importlib.util
import os
from typing import Any, NamedTuple

import torch
from model.compile_policy import configure_compile_limits

_FLEX_AVAILABLE = (
    importlib.util.find_spec('triton') is not None
    and importlib.util.find_spec('torch.nn.attention.flex_attention') is not None
)


class PackedFlexMask(NamedTuple):
    block_mask: Any


def document_mask(sequence_ids):
    """Match the existing same-ID + causal mask, including padding ID -1."""
    length = sequence_ids.size(1)

    def allowed(batch, head, query, key):
        # Block-mask builders/kernels can visit a padded tile. Clamp the gather
        # as well as masking its result; tensor '&' does not short-circuit.
        qid = sequence_ids[batch, query.clamp(max=length - 1)]
        kid = sequence_ids[batch, key.clamp(max=length - 1)]
        return (query < length) & (key < length) & (query >= key) & (qid == kid)

    return allowed


@lru_cache(maxsize=1)
def _flex_functions():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # Flex also compiles when the outer trainer uses --use_compile 0.
    # Configure both Dynamo limits before constructing its lazy wrappers.
    configure_compile_limits()
    return (
        torch.compile(create_block_mask, fullgraph=True),
        torch.compile(flex_attention, fullgraph=True),
    )


@torch.compiler.disable
def build_packed_flex_mask(sequence_ids):
    if sequence_ids.ndim != 2 or sequence_ids.size(1) == 0:
        raise ValueError('sequence_ids must have shape [batch, nonempty sequence]')
    create_mask, _ = _flex_functions()
    batch, length = sequence_ids.shape
    block_mask = create_mask(
        document_mask(sequence_ids), batch, None, length, length,
        device=sequence_ids.device, BLOCK_SIZE=128,
    )
    return PackedFlexMask(block_mask)


def maybe_prepare_flex_mask(
    sequence_ids, reference, *, attention_mask=None, enabled=True,
    dropout_p=0.0, head_dim=32,
):
    """Select only the validated packed CUDA BF16, dropout-free fast path.

    Arbitrary caller masks and unsupported compute modes retain the existing
    shared-bias SDPA implementation. No dense mask is built when Flex is used.
    """
    backend = os.environ.get('INSTINCT_PACKED_ATTENTION_BACKEND', 'auto').strip().lower()
    if backend not in ('auto', 'flex', 'sdpa'):
        raise ValueError('INSTINCT_PACKED_ATTENTION_BACKEND must be auto, flex or sdpa')
    if backend == 'sdpa':
        return None
    if (not enabled or not reference.is_cuda or torch.version.hip is not None
            or attention_mask is not None or dropout_p != 0.0
            or sequence_ids.size(1) <= 1 or head_dim not in (16, 32, 64, 128)):
        return None
    dtype = (torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda')
             else reference.dtype)
    if dtype != torch.bfloat16:
        return None
    if not _FLEX_AVAILABLE:
        if backend == 'flex':
            raise RuntimeError('Flex packed attention requires PyTorch FlexAttention and Triton')
        return None
    return build_packed_flex_mask(sequence_ids)


@torch.compiler.disable
def packed_flex_attention(q, k, v, mask: PackedFlexMask):
    _, attention = _flex_functions()
    # Do not repeat KV heads: FlexAttention implements GQA in its kernels and
    # accumulates gradients into the original compact K/V tensors.
    output = attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        block_mask=mask.block_mask, enable_gqa=True,
        kernel_options={'BACKEND': 'TRITON'},
    )
    return output.transpose(1, 2)
