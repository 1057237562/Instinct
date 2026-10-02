"""Replay packed SDPA across layers, comparing bias allocation and step time.

Run with training stopped: python experiments/bench_shared_attention_bias.py --compile
This isolates attention; it is not an end-to-end model throughput benchmark.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL import order
import trainer.compile_cache  # noqa: F401 -- before torch
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model.attention_mask import prepare_sdpa_attention_bias
from model.flash_attn_4 import flash_attention, _get_fa4
from model.sequence_packing import merge_packed_attention_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--length', type=int, default=2928)
    parser.add_argument('--layers', type=int, default=32)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if min(args.length, args.layers, args.batch) < 1:
        parser.error('length, layers and batch must be positive')
    torch.set_num_threads(1)
    _get_fa4()  # resolve optional imports outside compilation
    attention = (torch.compile(flash_attention, fullgraph=True) if args.compile
                 else flash_attention)
    segments = (torch.arange(args.length, device='cuda') // max(1, args.length // 4))
    segments = segments[None].expand(args.batch, -1)
    mask = merge_packed_attention_mask(segments, None)
    # All layers share these inputs so memory deltas isolate saved attention
    # buffers rather than model parameters, activations or optimizer state.
    q = torch.randn(args.batch, args.length, 16, 32, device='cuda', dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn(args.batch, args.length, 4, 32, device='cuda', dtype=torch.bfloat16,
                    requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    rows = []
    for shared in (False, True):
        warm_mask = prepare_sdpa_attention_bias(mask, q, query_length=args.length) if shared else mask
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attention(q, k, v, attention_mask=warm_mask).float().sum().backward()
        q.grad = k.grad = v.grad = None
        del warm_mask
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        prepared = prepare_sdpa_attention_bias(mask, q, query_length=args.length) if shared else mask
        saved_biases = {}

        def save(tensor):
            if tensor.dtype == q.dtype and tensor.ndim == 4 and tensor.shape[-2:] == (args.length, args.length):
                storage = tensor.untyped_storage()
                saved_biases[storage.data_ptr()] = storage.nbytes()
            return tensor

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            with torch.autograd.graph.saved_tensors_hooks(save, lambda t: t):
                loss = sum(attention(q, k, v, attention_mask=prepared).float().sum()
                           for _ in range(args.layers))
            loss.backward()
        torch.cuda.synchronize()
        rows.append(dict(
            shared_bias=shared, compiled=args.compile,
            attention_forward_backward_ms=1000 * (time.perf_counter() - start),
            peak_above_inputs_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20,
            unique_saved_biases=len(saved_biases),
            saved_bias_storage_mib=sum(saved_biases.values()) / 2**20,
        ))
        q.grad = k.grad = v.grad = None
        del loss, prepared
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
