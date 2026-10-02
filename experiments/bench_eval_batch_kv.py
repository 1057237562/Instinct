"""A/B the existing FP8 and eval-only model-dtype KV caches for batched MoE.

Run without other GPU jobs. No benchmark result files or model weights are
written. The batch size and length mirror a HumanEval generation batch.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401,E402 -- configure Inductor before torch
import datasets  # noqa: F401,E402 -- Windows import order
import torch

from eval_batch import _eval_kv_cache_precision
from model.inference_runtime import optimize_inference, select_inference_dtype
from model.model_instinct import InstinctConfig, InstinctForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='checkpoints/pretrain_20260925_121129_512_moe.json')
    parser.add_argument('--weight', default='out/full_sft_20260928_220514_512_moe.pth')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--prompt-width', type=int, default=200)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    with open(ROOT / args.config, encoding='utf-8') as stream:
        config = InstinctConfig(**json.load(stream))
    model = InstinctForCausalLM(config)
    weights = torch.load(ROOT / args.weight, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=False)
    del weights
    model = model.to(dtype=select_inference_dtype('cuda')).eval().cuda()
    optimize_inference(model, 'auto')
    ids = torch.randint(10, config.vocab_size, (args.batch_size, args.prompt_width), device='cuda')
    mask = torch.ones_like(ids)

    def measure(use_fast):
        context = _eval_kv_cache_precision(
            model, args.batch_size, args.prompt_width, args.max_new_tokens)
        with context as dtype:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate(inputs=ids, attention_mask=mask,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False, eos_token_id=None)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        return dtype, elapsed, output

    # Alternate order to reduce clock/temperature drift. Skip initial kernel
    # compilation for both variants before measuring.
    original_ok = model._static_cache_ok
    for fast in (False, True):
        if not fast:
            model._static_cache_ok = False
        measure(fast)
        model._static_cache_ok = original_ok
    samples = {False: [], True: []}
    first_output = None
    for _ in range(args.rounds):
        for fast in (False, True):
            model._static_cache_ok = original_ok if fast else False
            dtype, elapsed, output = measure(fast)
            samples[fast].append(elapsed)
            if fast is False and first_output is None:
                first_output = output
            if fast is True and first_output is not None:
                matches = (first_output == output).float().mean().item()
        model._static_cache_ok = original_ok
    total = args.batch_size * args.max_new_tokens
    for fast in (False, True):
        elapsed = statistics.median(samples[fast])
        print(f'{"adaptive" if fast else "configured"}: '
              f'{total / elapsed:.1f} aggregate tokens/s, {elapsed:.2f}s')
    print(f'last matched-token fraction across cache precisions: {matches:.3f}')


if __name__ == '__main__':
    main()
