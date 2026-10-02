"""Model-level training step throughput, for A/B against a stashed tree.

Times forward + backward over a fixed synthetic batch, so it depends only on
``model/`` (not the data pipeline). Deliberately touches no inference fast-path
attribute, so the same file runs against a pre-change model tree:

    git worktree add ../instinct-base HEAD
    cp experiments/measure_train_step.py ../instinct-base/experiments/
    cd ../instinct-base && python experiments/measure_train_step.py --config <abs path>
    python experiments/measure_train_step.py --config checkpoints/....json

Deliberate choices, so the numbers mean something on a shared GPU:

* Forward + backward only. The optimizer step does not touch any code this
  benchmark is meant to compare, and AdamW's two fp32 moments on a 678M-parameter
  MoE (10+ GiB) do not fit beside whatever else owns the card; overflow makes the
  driver thrash and the timings swing by 3-5x. Free VRAM is printed for that
  reason: a full-size MoE step measured 1.4-7.9 s/step with only ~10 GiB free and
  ~0.3 s/step with ~13 GiB free, on identical code.
* ``--layers`` shrinks the MoE, and ``--block`` times a single MOEFeedForward
  forward+backward with no model at all: the tightest, drift-free look at the
  expert dispatch.
* Alternate trees run by run. This box drifts ~1.2x over minutes, which is the
  same size as the effect being measured; a single block per tree is not evidence.
* ``--dump-grads`` prints loss and gradient checksums for one seeded step: the way
  to show a change left training maths alone (identical to 10 decimals across
  trees, for both the dense and MoE configs).
* ``--count-syncs`` lists host-blocking CUDA ops per step. The only training-side
  difference these changes introduced is one such op disappearing (the RoPE-cache
  check); ``--inject-sync`` re-adds one to show it is not measurable in step time.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM, MOEFeedForward

REPO = Path(__file__).resolve().parents[1]
MOE_CONFIG = REPO / 'checkpoints' / 'pretrain_20260919_174858_512_moe.json'
DENSE_CONFIG = REPO / 'checkpoints' / 'pretrain_20260912_180004_768.json'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--layers', type=int, default=8, help='override num_hidden_layers')
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--seq', type=int, default=256)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--blocks', type=int, default=3)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--config', default=str(MOE_CONFIG))
parser.add_argument('--block', action='store_true',
                    help='time a single MOEFeedForward forward+backward instead')
parser.add_argument('--count-syncs', action='store_true',
                    help='report host-blocking CUDA ops per step instead of timing')
parser.add_argument('--inject-sync', action='store_true',
                    help='force one host sync per step, to attribute the base tree\'s extra sync')
parser.add_argument('--dump-grads', action='store_true',
                    help='print loss and gradient checksums for one step (A/B the maths)')
args = parser.parse_args()


def time_steps(run, warmup, steps, blocks):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(blocks):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) / steps)
    return statistics.median(samples) * 1e3, [round(s * 1e3, 1) for s in samples]


with open(args.config, encoding='utf-8') as handle:
    raw = json.load(handle)
raw['num_hidden_layers'] = args.layers
config = InstinctConfig(**raw)
torch.manual_seed(0)

if args.block:
    model = InstinctForCausalLM(config).cuda().train()
    block = model.model.layers[0].mlp
    assert isinstance(block, MOEFeedForward), 'config must use MoE for --block'
    hidden = config.hidden_size
    x = torch.randn(args.batch, args.seq, hidden, device='cuda', dtype=torch.float32,
                    requires_grad=True)

    def run():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = block(x)
        output.float().square().mean().backward()
        x.grad = None

    label = f'MoE block  (batch {args.batch}x{args.seq})'
else:
    model = InstinctForCausalLM(config).cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    data = torch.randint(3, config.vocab_size - 2, (args.batch, args.seq), device='cuda')

    def run():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(input_ids=data, labels=data)
        output.loss.backward()
        model.zero_grad(set_to_none=True)

    label = f'{Path(args.config).stem} ({args.layers}L, batch {args.batch}x{args.seq})'

if args.inject_sync:
    inner = run

    def run():
        torch.cuda.synchronize()  # stands in for the pre-change RoPE-cache check
        inner()

torch.cuda.reset_peak_memory_stats()
if args.dump_grads:
    # Same seeds, same batch, one fwd+bwd: identical numbers mean the training
    # graph is untouched (the checks are dtype/order sensitive on purpose).
    torch.manual_seed(0)
    model = InstinctForCausalLM(config).cuda().train()
    data = torch.randint(3, config.vocab_size - 2, (args.batch, args.seq), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids=data, labels=data)
    output.loss.backward()
    total, squares = 0.0, 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.double().abs().sum().item()
            squares += parameter.grad.double().square().sum().item()
    first_mlp = model.model.layers[0].mlp
    probe = (first_mlp.experts[0].gate_proj if hasattr(first_mlp, 'experts') else first_mlp.gate_proj)
    print(f'[grads] loss={output.loss.item():.10f} '
          f'grad_abs_sum={total:.6f} grad_sq_sum={squares:.6f} '
          f'layer0_gate_sum={probe.weight.grad.double().abs().sum().item():.6f}')
    sys.exit(0)

if args.count_syncs:
    import warnings
    from pathlib import Path as _Path
    for _ in range(args.warmup):
        run()
    counts = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.cuda.set_sync_debug_mode('warn')
        try:
            run()
        finally:
            torch.cuda.set_sync_debug_mode('default')
    for item in caught:
        site = f'{_Path(item.filename).name}:{item.lineno}'
        counts[site] = counts.get(site, 0) + 1
    print(f'[sync] {label}: {sum(counts.values())} host-blocking CUDA op(s) per step')
    for site, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'[sync]     {count:3d}  {site}')
    sys.exit(0)

ms, samples = time_steps(run, args.warmup, args.steps, args.blocks)
print(f'[train] {label:44s} {ms:8.2f} ms/step  peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB '
      f'blocks={samples}')
print(f'[train] free VRAM at start: {torch.cuda.mem_get_info()[0] / 2 ** 30:.2f} GiB')
