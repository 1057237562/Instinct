"""Prepare/replay one Dense block using cuBLASLt + Triton, without Torch.

Run from the repository root. This is a kernel/scheduler smoke test, not a trainer.
The native shim must first be built with CMake; see docs/dense_cublas.md.
"""
import argparse
from dataclasses import asdict
import json
import sys

from experiments.smoke_dense_triton import RejectTorch


def run(args):
    if any(n in sys.modules for n in ('torch', 'torchao')):
        raise RuntimeError('run in a fresh process without torch')
    sys.meta_path.insert(0, RejectTorch())
    import numpy as np
    from experiments.compile_dense_triton import weight_shapes
    from model.dense_cublas.backend import CublasBackend, ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps, ScheduledDenseBlock
    from model.dense_triton.ops import BlockConfig

    schedule = ScheduleConfig(workspace_mib=args.workspace_mib, candidates=args.candidates,
                              tune=args.tune, overlap_dw=args.overlap_dw,
                              dw_sm_target=args.dw_sm_target)
    c = BlockConfig()
    rng = np.random.default_rng(42)
    with CublasBackend(args.device, schedule, args.library) as backend:
        baseline = backend.memory()
        op = CublasDenseOps(backend, fp8=bool(args.fp8))
        weights = {}
        for name, shape in weight_shapes(c).items():
            value = np.ones(shape, dtype=np.float32) if len(shape) == 1 else rng.normal(0, 0.02, shape).astype(np.float32)
            weights[name] = backend.from_numpy(value)
        x = op.cast(backend.from_numpy(rng.standard_normal((args.batch_size, args.seq_len, c.hidden_size)).astype(np.float32)), args.dtype)
        dy = op.cast(backend.from_numpy(rng.standard_normal(x.shape).astype(np.float32)), args.dtype)
        angles = np.arange(args.seq_len, dtype=np.float32)[:, None] / (1e6 ** (np.arange(0, c.head_dim, 2, dtype=np.float32) / c.head_dim))
        cos, sin = [backend.from_numpy(np.concatenate((f(angles), f(angles)), -1)) for f in (np.cos, np.sin)]
        block = ScheduledDenseBlock(op, weights, c)

        def step():
            out, saved = block.forward(x, cos, sin)
            dx, gradients = block.backward(saved, dy)
            return (out, dx, gradients)

        plan = backend.build(step)
        out, dx, grads = plan.outputs
        tensors = (out, dx, *grads.values())
        reference = [backend.to_numpy(t) for t in tensors]
        if not all(np.isfinite(a).all() for a in reference):
            raise RuntimeError('nonfinite Dense output/gradient during preparation')
        memory = backend.memory()
        allocations = len(backend._allocations)
        if args.cuda_graph:
            plan.capture()
        for _ in range(args.steps):
            plan.replay()
        backend.synchronize()
        if len(backend._allocations) != allocations:
            raise RuntimeError('replay allocated new arena buffers')
        for tensor, expected in zip(tensors, reference):
            np.testing.assert_allclose(backend.to_numpy(tensor), expected, atol=2e-5, rtol=2e-5)
        return {'status': 'ok', 'arch': backend.arch, 'fp8': bool(args.fp8), 'shape': x.shape,
                'schedule': asdict(schedule), 'cuda_graph': args.cuda_graph, 'replays': args.steps,
                'torch_imported': 'torch' in sys.modules, 'torchao_imported': 'torchao' in sys.modules,
                'gradient_parameters': sorted(grads), 'baseline_memory': baseline,
                'prepared_memory': memory, 'after_replay_memory': backend.memory(),
                'commands': plan.describe(), 'algorithms': backend.tuning_report(),
                'note': 'Fixed-buffer block smoke test; no optimizer, trainer, or MFU result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--library', help='path to the built cuBLASLt shim')
    parser.add_argument('--seq_len', type=int, default=17)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    parser.add_argument('--fp8', type=int, choices=[0, 1], default=1)
    parser.add_argument('--workspace_mib', type=int, default=32)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--overlap_dw', action='store_true')
    parser.add_argument('--dw_sm_target', type=int, default=0)
    parser.add_argument('--cuda_graph', action='store_true')
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.seq_len <= 32768 or args.batch_size < 1 or args.steps < 1:
        parser.error('invalid device, sequence length, batch size or replay count')
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
