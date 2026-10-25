"""Profile the V2 recipe at fixed weights with resident optimizer/gradient state.

No optimizer update is performed. Use profile_checkpoint_recipe.py for final
accumulated-update throughput and verified current-state checkpointing.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401
import datasets  # noqa: F401 -- Windows DLL ordering
import torch

from experiments.profile_checkpoint_policy import reserve_training_state
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--layers', type=int, default=32)
    parser.add_argument('--flex-tma', action='store_true')
    parser.add_argument('--expand-kv', action='store_true')
    args = parser.parse_args()
    if args.flex_tma or args.expand_kv:
        import model.flash_attn_4 as attention_module
        from torch.nn.attention.flex_attention import flex_attention
        options = {'BACKEND': 'TRITON'}
        if args.flex_tma:
            options['USE_TMA'] = True

        def candidate_attention(q, k, v, mask):
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            if args.expand_kv:
                k, v = [t.repeat_interleave(q.shape[1] // t.shape[1], 1) for t in (k, v)]
            return flex_attention(q, k, v, block_mask=mask.block_mask,
                enable_gqa=not args.expand_kv, kernel_options=options).transpose(1, 2)

        attention_module.packed_flex_attention = candidate_attention
    os.environ.update(INSTINCT_MUON_MANTISSA='1', INSTINCT_MUON_MOMENTUM_DTYPE='bf16',
                      INSTINCT_MUON_BATCH_SIZE='16', INSTINCT_MUON_WORKSPACE_MB='512')
    torch.set_num_threads(1)
    torch.manual_seed(307)
    torch.cuda.set_per_process_memory_fraction(min(1., 14.37 * 2**30 /
        torch.cuda.get_device_properties(0).total_memory))
    config = InstinctConfig(**dict(json.loads((ROOT / 'trainer/config_instinct_v2.json').read_text()),
        use_grad_checkpoint=4, num_hidden_layers=args.layers))
    raw = cast_training_parameters(InstinctForCausalLM(config), 'bf16').cuda().train()
    optimizer = reserve_training_state(raw)
    apply_torchao_fp8_training(raw, SimpleNamespace(fp8_training='tensorwise', fp8_filter='auto',
        dtype='bfloat16', device='cuda', use_compile=1))
    model = torch.compile(raw, mode='max-autotune-no-cudagraphs', dynamic=False)
    batches = []
    for b, s, segments in [(4, 3072, 3), (3, 4096, 1)]:
        tokens = torch.randint(3, config.vocab_size, (b, s), device='cuda')
        labels = tokens.clone()
        ids = None
        if segments > 1:
            ids = (torch.arange(s, device='cuda') * segments // s).expand(b, -1).contiguous()
            labels[:, 1:][ids[:, 1:] != ids[:, :-1]] = -100
        batches.append((tokens, labels, ids))

    def step(index):
        tokens, labels, ids = batches[index]
        model.zero_grad(set_to_none=False)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(tokens, labels=labels, sequence_ids=ids)
            loss = out.loss
            del out
        loss.backward()
        return loss.detach()

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', args=vars(args), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, scope='fixed weights; real resident optimizer and gradients; no updates',
        environment={k:v for k,v in os.environ.items() if k.startswith(('INSTINCT_', 'TORCHINDUCTOR_'))})
    target.write_text(json.dumps(report, indent=2))
    for _ in range(args.warmup):
        for index in range(2):
            loss = step(index)
            torch.cuda.synchronize()
            print(f'warmup bucket={index} loss={float(loss):.6f}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    timings = [[], []]
    for _ in range(args.steps):
        for index in range(2):
            start = time.perf_counter()
            loss = step(index)
            torch.cuda.synchronize()
            timings[index].append((time.perf_counter() - start) * 1000)
    report.update(bucket_ms=timings, median_ms=[statistics.median(x) for x in timings],
        allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    print(json.dumps(report), flush=True)
    if args.profile:
        for index in range(2):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False) as prof:
                step(index)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(target.with_suffix(f'.bucket{index}.trace.json')))
            target.with_suffix(f'.bucket{index}.txt').write_text(
                prof.key_averages().table(sort_by='self_device_time_total', row_limit=70), encoding='utf-8')
    report['status'] = 'ok'
    target.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
