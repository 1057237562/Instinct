"""KV Cache 精度量化助手。

支持的 kv_cache_dtype:
- fp32 / bf16 / fp16: 直接 cast(无 scale)
- fp8_e4m3 / fp8_e5m2: 按 (batch, kv_head) 分片量化,每片独立 scale

缓存格式约定(由 Attention 层维护):
- 非 fp8: (k, v)
- fp8:    (qk, qv, k_scale, v_scale),scale 形状 (bs, kv_heads, 1, 1)

fp8 收益: KV 缓存每元素 1 字节(vs bf16 的 2 字节),decode 时 KV 带宽减半;
反量化在注意力计算前完成,数学精度仍为 bf16。
"""
from typing import Dict, Optional, Tuple

import torch

_FP8_TYPES: Dict[str, torch.dtype] = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}

_CAST_TYPES: Dict[str, torch.dtype] = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def is_fp8(dtype_str: str) -> bool:
    """判断 ``dtype_str`` 是否为 fp8 量化类型(``fp8_e4m3`` / ``fp8_e5m2``)。"""
    return dtype_str in _FP8_TYPES


def quantize_kv(x: torch.Tensor, dtype_str: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """对 (bs, seq, kv_heads, hd) 的 k 或 v 量化。

    Returns:
        (q, scale): q 为量化张量; fp8 时 scale 形状 (bs, kv_heads, 1, 1),
        非 fp8 时 scale 为 None。
    """
    if dtype_str in _CAST_TYPES:
        return x.to(_CAST_TYPES[dtype_str]), None
    if dtype_str in _FP8_TYPES:
        dt = _FP8_TYPES[dtype_str]
        max_pos = torch.finfo(dt).max
        x = x.to(torch.bfloat16)
        amax = x.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)  # (bs, kv, 1, 1)
        scale = (max_pos / amax).to(torch.float32)
        q = (x.to(torch.float32) * scale).to(dt)
        return q, scale
    raise ValueError(f"未知 kv_cache_dtype: {dtype_str}, 可选 {list(_CAST_TYPES) + list(_FP8_TYPES)}")


def dequantize_kv(q: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
    """反量化缓存。非 fp8 时 scale 为 None,直接返回原张量。"""
    if scale is None:
        return q
    return (q.to(torch.float32) / scale).to(torch.bfloat16)


def parse_cache(cache: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """读取缓存:统一返回 bf16 的 (k, v),兼容 (k,v) 与 (qk,qv,k_scale,v_scale) 两种格式。"""
    if len(cache) == 4:
        return dequantize_kv(cache[0], cache[2]), dequantize_kv(cache[1], cache[3])
    return cache[0], cache[1]


def make_cache(k: torch.Tensor, v: torch.Tensor, dtype_str: str) -> Tuple[torch.Tensor, ...]:
    """写入缓存:按 dtype_str 量化。返回 (k, v) 或 (qk, qv, k_scale, v_scale)。"""
    qk, ks = quantize_kv(k, dtype_str)
    qv, vs = quantize_kv(v, dtype_str)
    if is_fp8(dtype_str):
        return (qk, qv, ks, vs)
    return (qk, qv)
