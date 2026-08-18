"""
FlashAttention-4 (FA4) 集成层。

设计目标:让 Instinct 所有注意力变体(Dense / MoE / Loop / Linear 混合)在训练时
优先走 FA4 fused kernel(Blackwell / Hopper, `flash-attn >= 2.8` 的
`flash_attn_interface`),不可用时自动回退到 PyTorch 原生 SDPA flash /
mem-efficient backend(在 sm_120 上实测约 20x 于 math),极端情况再回退 eager。

使用约定:
- 输入 / 输出布局均为 (batch, seqlen, nheads, head_dim),与 FA4 / SDPA 统一,
  调用方无需手动 transpose 或 repeat_kv(GQA 由 kernel 原生支持)。
- 仅加速训练快路径(无 attention_mask、无 KV cache、seq_len > 1),
  推理 / masked 路径仍走原 eager 实现。
- FA4 仅支持 bf16 / fp8,本项目训练默认 bf16;fp16 / fp32 自动走 SDPA。
- seq_len 非 128 倍数时(FA4 non-varlen 接口的约束),causal 场景自动 pad 到
  128 倍数再截断(因果掩码下数值精确等价);非 causal 场景回退 SDPA。
"""
import torch
import torch.nn.functional as F

_flash_attn_4_func = None
_tried_import = False


def _repeat_kv(x, n_rep):
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)


def _get_fa4():
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


def flash_attention(q, k, v, dropout_p=0.0, is_causal=True):
    """Fused attention 快路径。

    Args:
        q: (batch, seqlen, nheads, head_dim),支持 GQA(nheads >= kv_heads)
        k, v: (batch, seqlen, kv_heads, head_dim)
        dropout_p: attention dropout 概率(训练时)
        is_causal: 是否因果掩码

    Returns:
        (batch, seqlen, nheads, head_dim)
    """
    fa4 = _get_fa4()
    if (
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
                    q = F.pad(q, (0, 0, 0, pad))
                    k = F.pad(k, (0, 0, 0, pad))
                    v = F.pad(v, (0, 0, 0, pad))
                out = fa4(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
                if isinstance(out, tuple):
                    out = out[0]
                return out[:, :seqlen] if pad else out
            except Exception:
                pass  # FA4 kernel 不支持当前 shape/dtype,回退 SDPA
    # 回退: PyTorch SDPA(torch 2.8 在 sm_120 上自动命中 flash backend),需手动 repeat_kv
    n_rep = q.size(2) // k.size(2)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), _repeat_kv(k, n_rep).transpose(1, 2), _repeat_kv(v, n_rep).transpose(1, 2),
        dropout_p=dropout_p, is_causal=is_causal,
    ).transpose(1, 2)
