"""
FlashAttention-4 (FA4) 集成层。

设计目标:让 Instinct 所有注意力变体(Dense / MoE / Loop / Linear 混合)在训练时
优先走 FA4 fused kernel(Blackwell / Hopper, `flash-attn >= 2.8` 的
`flash_attn_interface`),不可用时自动回退到 PyTorch 原生 SDPA flash /
mem-efficient backend(在 sm_120 上实测约 20x 于 math),极端情况再回退 eager。

使用约定:
- 输入 / 输出布局均为 (batch, seqlen, nheads, head_dim),与 FA4 / SDPA 统一,
  调用方无需手动 transpose 或 repeat_kv(集成层内部适配 GQA)。
- PackedFlexMask 使用文档分块 FlexAttention 和紧凑 GQA；不适用时保持共享 bias SDPA。
- FA4 仅处理无 attention_mask 的快路径；SDPA fallback 显式展开 K/V heads。
- FA4 仅支持 bf16 / fp8,本项目训练默认 bf16;fp16 / fp32 自动走 SDPA。
- seq_len 非 128 倍数时(FA4 non-varlen 接口的约束),causal 场景自动 pad 到
  128 倍数再截断(因果掩码下数值精确等价);非 causal 场景回退 SDPA。
"""
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from model.attention_mask import PreparedAttentionBias, normalize_attention_mask
from model.packed_attention import PackedFlexMask, packed_flex_attention

_flash_attn_4_func = None
_tried_import = False


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand compact GQA K/V heads for the proven SDPA fallback path."""
    batch, seq_len, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(
        batch, seq_len, kv_heads, n_rep, head_dim
    ).reshape(batch, seq_len, kv_heads * n_rep, head_dim)


def _get_fa4() -> Optional[Callable[..., Any]]:
    """惰性探测 flash_attn_interface.flash_attn_func(FA4),缓存结果。"""
    global _flash_attn_4_func, _tried_import
    if not _tried_import:
        _tried_import = True
        try:
            from flash_attn_interface import flash_attn_func as _fa4
            _flash_attn_4_func = _fa4
        except Exception:
            _flash_attn_4_func = None
    return _flash_attn_4_func


def resolve_flash_attn() -> Optional[Callable[..., Any]]:
    """探测 FA4 并缓存结果,必须在 torch.compile 之前调用一次。

    Dynamo 会执行被追踪帧里的 ``from flash_attn_interface import ...``: 未安装
    flash-attn 时这条 import 抛错,于是**每一层注意力**都产生一次图断裂。提前
    探测让追踪期只走 ``if not _tried_import`` 的短路分支。
    """
    return _get_fa4()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    attention_mask: Optional[torch.Tensor | PreparedAttentionBias | PackedFlexMask] = None,
) -> torch.Tensor:
    """Fused attention 快路径。

    Args:
        q: (batch, seqlen, nheads, head_dim),支持 GQA(nheads >= kv_heads)
        k, v: (batch, seqlen, kv_heads, head_dim)
        dropout_p: attention dropout 概率(训练时)
        is_causal: 是否因果掩码

    Returns:
        (batch, seqlen, nheads, head_dim)
    """
    if isinstance(attention_mask, PackedFlexMask):
        if dropout_p != 0.0:
            raise ValueError('Packed FlexAttention does not support attention dropout')
        return packed_flex_attention(q, k, v, attention_mask)
    fa4 = _get_fa4()
    if (
        attention_mask is None
        and
        fa4 is not None
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and q.size(-1) % 8 == 0
    ):
        seqlen = q.size(1)
        pad = (-seqlen) % 128
        # 非 causal 场景 pad 会改变真实 token 的输出(后续 token 影响前序),故不允许
        if pad == 0 or is_causal:
            try:
                if pad:
                    pad_spec = (0, 0, 0, pad)
                    q, k, v = F.pad(q, pad_spec), F.pad(k, pad_spec), F.pad(v, pad_spec)
                out = fa4(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
                if isinstance(out, tuple):
                    out = out[0]
                return out[:, :seqlen] if pad else out
            except Exception:
                pass  # FA4 kernel 不支持当前 shape/dtype,回退 SDPA
    # 回退: 显式展开 K/V 可让当前 torch.compile + packed mask 保持已验证的
    # memory-efficient SDPA backend；native enable_gqa 会被 Inductor 分解为 dense BMM。
    n_rep = q.size(2) // k.size(2)
    q = q.transpose(1, 2)
    k = _repeat_kv(k, n_rep).transpose(1, 2)
    v = _repeat_kv(v, n_rep).transpose(1, 2)
    sdpa_mask = None
    sdpa_is_causal = is_causal
    if isinstance(attention_mask, PreparedAttentionBias):
        # Already combined, cast and aligned once at the model entry point.
        # Every layer's backward can reference the same storage.
        sdpa_mask = attention_mask.tensor
        sdpa_is_causal = False
    elif attention_mask is not None:
        sdpa_mask = normalize_attention_mask(attention_mask)
        if is_causal:
            query_len, key_len = q.size(-2), k.size(-2)
            causal = torch.ones(
                (query_len, key_len), dtype=torch.bool, device=q.device
            ).tril(diagonal=key_len - query_len)
            sdpa_mask = sdpa_mask & causal
        # Combining the masks explicitly works across torch versions that do
        # not permit attn_mask and is_causal=True at the same time.
        sdpa_is_causal = False
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=sdpa_is_causal,
    ).transpose(1, 2)
