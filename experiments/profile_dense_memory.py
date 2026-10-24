"""Fresh-process CUDA/PyTorch/Instinct memory diagnostics (including Windows).

Run each variant as a NEW process. Device-wide usage includes the desktop and
other processes; it is not a measurement of this process's CUDA-context cost.
The cuda-only mode imports neither PyTorch nor Triton.
"""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cuda', 'train'], default='train')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=768)
    parser.add_argument('--use_grad_checkpoint', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--use_compile', type=int, choices=[0, 1], default=0)
    parser.add_argument('--compile_mode', default='default', choices=[
        'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--json_out', type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.max_seq_len, args.hidden_size,
           args.num_hidden_layers, args.warmup, args.steps) <= 0:
        parser.error('shapes, warmup, and steps must be positive')
    if args.device < 0:
        parser.error('device must be nonnegative')
    return args


def cuda_context_probe(device_index):
    """Small Driver API baseline; no framework/library context is initialized."""
    lib = ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else ctypes.CDLL('libcuda.so.1')
    def bind(name, types):
        function = getattr(lib, name)
        function.argtypes, function.restype = types, ctypes.c_int
        def call(*args):
            result = function(*args)
            if result:
                raise RuntimeError(f'{name} failed with CUDA driver error {result}')
        return call
    integer, pointer = ctypes.c_int, ctypes.c_void_p
    init = bind('cuInit', [ctypes.c_uint])
    get_device = bind('cuDeviceGet', [ctypes.POINTER(integer), integer])
    retain = bind('cuDevicePrimaryCtxRetain', [ctypes.POINTER(pointer), integer])
    release_name = 'cuDevicePrimaryCtxRelease_v2' if hasattr(lib, 'cuDevicePrimaryCtxRelease_v2') else 'cuDevicePrimaryCtxRelease'
    release = bind(release_name, [integer])
    get_current = bind('cuCtxGetCurrent', [ctypes.POINTER(pointer)])
    set_current = bind('cuCtxSetCurrent', [pointer])
    memory = bind('cuMemGetInfo_v2', [ctypes.POINTER(ctypes.c_size_t)] * 2)
    init(0)
    dev, context, prior = integer(), pointer(), pointer()
    get_device(ctypes.byref(dev), device_index)
    get_current(ctypes.byref(prior))
    retain(ctypes.byref(context), dev)
    try:
        set_current(context)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        memory(ctypes.byref(free), ctypes.byref(total))
        return {'mode': 'cuda', 'device_ordinal_kind': 'CUDA Driver API physical ordinal',
                'device_index': device_index, 'torch_imported': 'torch' in sys.modules,
                'device_free_bytes': free.value, 'device_total_bytes': total.value,
                'device_used_bytes': total.value - free.value}
    finally:
        set_current(prior)
        release(dev)


def train_probe(args):
    # Required Windows import order from AGENTS.md.
    import datasets  # noqa: F401
    import torch
    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no GPU measurements were performed')
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    device = torch.device('cuda', args.device)
    samples = []
    def sample(stage):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        samples.append({'stage': stage, 'allocated_bytes': allocated,
                        'reserved_bytes': reserved, 'unused_cached_bytes': reserved - allocated,
                        'device_used_bytes': total - free, 'device_free_bytes': free})

    sample('cuda_initialized')
    # Use the same native Dense shape defaults as the production model.
    if args.hidden_size % 16:
        raise ValueError('hidden_size must be divisible by 16 (8 heads with even rotary width)')
    cfg = InstinctConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                         use_grad_checkpoint=args.use_grad_checkpoint)
    torch.manual_seed(42)
    model = InstinctForCausalLM(cfg).to(device).train()
    sample('model_allocated')
    args.dtype = 'bfloat16'
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if args.use_compile:
        model = torch.compile(model, mode=args.compile_mode)
    ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.max_seq_len), device=device)
    labels = ids.clone()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            result = model(ids, labels=labels)
            loss = result.loss + result.aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        loss = step()
    sample('after_warmup_and_optimizer_state')
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    sample('after_measured_steps')
    result = {'mode': 'train', 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(device),
              'config': cfg.to_dict(), 'samples': samples, 'seconds': seconds,
              'steps': args.steps, 'warmup': args.warmup,
              'batch_size': args.batch_size, 'max_seq_len': args.max_seq_len,
              'use_compile': bool(args.use_compile), 'compile_mode': args.compile_mode,
              'slot_tokens_per_second': args.steps * ids.numel() / seconds,
              'final_loss': loss.item(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    optimizer.zero_grad(set_to_none=True)
    del loss
    gc.collect()
    torch.cuda.empty_cache()  # Diagnostic only, outside the timed training loop.
    sample('idle_after_empty_cache_model_optimizer_still_live')
    return result


def main():
    args = parse_args()
    try:
        result = cuda_context_probe(args.device) if args.mode == 'cuda' else train_probe(args)
        result['status'] = 'ok'
        code = 0
    except Exception as exc:
        result = {'status': 'error', 'mode': args.mode,
                  'error': f'{type(exc).__name__}: {exc}'}
        code = 1
    result['measurement_note'] = (
        'Device usage includes other processes and Windows display/driver memory. '
        'Compare fresh processes on an idle GPU. reserved-allocated is allocator cache, '
        'not inherently lost training capacity. Driver and PyTorch device ordinals may '
        'differ when CUDA_VISIBLE_DEVICES is set; select the same physical GPU.')
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
