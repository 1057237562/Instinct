"""Block-sparse causal document attention with compact GQA K/V.

Block masks are prepared outside checkpointed layers and shared across them.
V2's 96-wide attention remains traceable inside the outer model compiler.
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


def _flex_kernel_options(q, k):
    """TMA backward for the measured long V2 bucket; keep compact GQA storage.

    TMA slowed forward while accelerating long-context backward. Configure
    the two passes independently instead of changing the proven forward tile.
    """
    options = {'BACKEND': 'TRITON'}
    setting = os.environ.get('INSTINCT_V2_FLEX_TMA', 'auto').strip().lower()
    if setting not in ('auto', '0', '1'):
        raise ValueError('INSTINCT_V2_FLEX_TMA must be auto, 0 or 1')
    if (setting == '0' or not torch.is_grad_enabled() or not q.is_cuda
            or q.dtype != torch.bfloat16 or q.shape[2:] != (16, 96)
            or k.shape[2:] != (4, 96) or q.shape[:2] != k.shape[:2]
            or q.shape[1] % 128 != 0):
        return options
    if setting == 'auto' and q.shape[:2] != (3, 4096):
        return options
    from model.cuda_fastpath import blackwell_training_kernels
    if blackwell_training_kernels(q.device):
        options.update(bwd_USE_TMA=True, bwd_BLOCK_M1=32, bwd_BLOCK_N1=32,
                       bwd_BLOCK_M2=32, bwd_BLOCK_N2=32,
                       bwd_num_warps=4, bwd_num_stages=3)
    return options


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


@lru_cache(maxsize=2)
def _flex_functions(autotune=False):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # Flex also compiles when the outer trainer uses --use_compile 0.
    # Configure both Dynamo limits before constructing its lazy wrappers.
    configure_compile_limits()
    return (
        torch.compile(create_block_mask, fullgraph=True),
        torch.compile(flex_attention, fullgraph=True,
                      **({'mode':'max-autotune-no-cudagraphs'} if autotune else {})),
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
    dropout_p=0.0, head_dim=32, num_attention_heads=None, num_key_value_heads=None,
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
            or sequence_ids.size(1) <= 1):
        return None
    if head_dim not in (16, 32, 64, 128):
        from model.cuda_fastpath import blackwell_training_kernels
        # Native non-power-of-two Flex kernels are validated here for V2's
        # 96-wide heads. Older stacks/devices retain their existing SDPA path.
        validated_v2 = (num_attention_heads == 16 and num_key_value_heads == 4
                        and blackwell_training_kernels(reference.device))
        if head_dim != 96 or (backend != 'flex' and not validated_v2):
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


def maybe_prepare_causal_flex_mask(
    reference, *, enabled=True, attention_mask=None, dropout_p=0.0,
    head_dim=32, num_attention_heads=None, num_key_value_heads=None,
):
    """Prepare V2's plain causal mask before entering checkpointed layers.

    Single-document batches intentionally discard their sequence IDs in the
    loader. They still need the 96-wide Flex path, but creating its mask inside
    a layer would introduce a compiler-disable boundary inside checkpointing.
    Keep ordinary inference, caller masks and other model shapes unchanged.
    """
    if (not enabled or head_dim != 96 or num_attention_heads != 16
            or num_key_value_heads != 4 or not reference.is_cuda
            or torch.version.hip is not None or attention_mask is not None
            or dropout_p != 0.0 or reference.shape[1] <= 1
            or not torch.is_grad_enabled()):
        return None
    backend = os.environ.get('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'auto').strip().lower()
    if backend not in ('auto', 'flex', 'sdpa'):
        raise ValueError('INSTINCT_CAUSAL_ATTENTION_BACKEND must be auto, flex or sdpa')
    if backend == 'sdpa':
        return None
    dtype = (torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda')
             else reference.dtype)
    if dtype != torch.bfloat16:
        return None
    if backend == 'auto':
        from model.cuda_fastpath import blackwell_training_kernels
        if not blackwell_training_kernels(reference.device):
            return None
    if not _FLEX_AVAILABLE:
        if backend == 'flex':
            raise RuntimeError('Flex causal attention requires PyTorch FlexAttention and Triton')
        return None
    return build_causal_flex_mask(reference.shape[1], str(reference.device))


def packed_flex_attention(q, k, v, mask: PackedFlexMask):
    if q.shape[-1] == 96 and torch.compiler.is_compiling():
        # A disable boundary inside full-layer checkpointing leaves the entire
        # layer eager. Flex is a traceable higher-order op: keep V2's attention
        # inside the outer compiled checkpoint, including FP8 projections.
        from torch.nn.attention.flex_attention import flex_attention
        return flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            block_mask=mask.block_mask, enable_gqa=True,
            kernel_options=_flex_kernel_options(q, k),
        ).transpose(1, 2)
    return _packed_flex_attention_region(q, k, v, mask)


@torch.compiler.disable
def _packed_flex_attention_region(q, k, v, mask: PackedFlexMask):
    _, attention = _flex_functions(_training_autotune(q))
    # Do not repeat KV heads: FlexAttention implements GQA in its kernels and
    # accumulates gradients into the original compact K/V tensors.
    output = attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        block_mask=mask.block_mask, enable_gqa=True,
        kernel_options=_flex_kernel_options(q, k),
    )
    return output.transpose(1, 2)


@lru_cache(maxsize=8)
def _causal_flex_mask(length, device):
    create_mask, _ = _flex_functions()
    def causal(batch, head, query, key):
        return query >= key
    return create_mask(causal, None, None, length, length, device=device, BLOCK_SIZE=128)


@torch.compiler.disable
def build_causal_flex_mask(length, device):
    """Cache batch/head-independent causal metadata outside the layer graph."""
    return PackedFlexMask(_causal_flex_mask(length, device))


@torch.compiler.disable
def causal_flex_attention(q, k, v):
    """Full causal attention with compact GQA, including single-document batches."""
    _, attention = _flex_functions(_training_autotune(q))
    mask = _causal_flex_mask(q.shape[1], str(q.device))
    return attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                     block_mask=mask, enable_gqa=True,
                     kernel_options=_flex_kernel_options(q, k)).transpose(1,2)


def _training_autotune(q):
    setting=os.environ.get('INSTINCT_FLEX_AUTOTUNE','auto')
    if setting not in ('auto','0','1'):
        raise ValueError('INSTINCT_FLEX_AUTOTUNE must be auto, 0 or 1')
    from model.cuda_fastpath import blackwell_training_kernels
    return torch.is_grad_enabled() and (setting=='1' or
        (setting=='auto' and blackwell_training_kernels(q.device)))
