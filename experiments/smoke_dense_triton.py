"""Run one Dense block forward/backward using CUDA + Triton, without torch.

Not a trainer or a performance benchmark. Allocations last until process exit.
Run from the repository root: python -m experiments.smoke_dense_triton --fp8 1
"""
import argparse
import importlib.abc
import json
import sys


class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchao'):
            raise ImportError(f'framework-free smoke test forbids importing {fullname}')


def run(args):
    if any(n in sys.modules for n in ('torch', 'torchao')):
        raise RuntimeError('run this smoke test in a fresh process without torch')
    sys.meta_path.insert(0, RejectTorch())
    import numpy as np
    from model.dense_triton.runtime import CudaBackend
    from model.dense_triton.ops import BlockConfig, DenseBlock, DenseOps
    from experiments.compile_dense_triton import weight_shapes
    c = BlockConfig()
    rng = np.random.default_rng(42)
    with CudaBackend(args.device) as backend:
        baseline = backend.memory()
        op = DenseOps(backend, fp8=bool(args.fp8))
        weights = {}
        for name, shape in weight_shapes(c).items():
            value = np.ones(shape, dtype=np.float32) if len(shape) == 1 else rng.normal(0, 0.02, shape).astype(np.float32)
            weights[name] = backend.from_numpy(value)
        x = op.cast(backend.from_numpy(rng.standard_normal((1, args.seq_len, c.hidden_size)).astype(np.float32)), 'bf16')
        dy = op.cast(backend.from_numpy(rng.standard_normal(x.shape).astype(np.float32)), 'bf16')
        angles = np.arange(args.seq_len, dtype=np.float32)[:, None] / (1e6 ** (np.arange(0, c.head_dim, 2, dtype=np.float32) / c.head_dim))
        cos, sin = [backend.from_numpy(np.concatenate((f(angles), f(angles)), -1)) for f in (np.cos, np.sin)]
        block = DenseBlock(op, weights, c)
        out, saved = block.forward(x, cos, sin)
        dx, gradients = block.backward(saved, dy)
        output_arrays = [backend.to_numpy(t) for t in (out, dx, *gradients.values())]
        if not all(np.isfinite(t).all() for t in output_arrays):
            raise RuntimeError('nonfinite Dense output/gradient')
        return {'status': 'ok', 'arch': backend.arch, 'fp8': bool(args.fp8),
                'shape': x.shape, 'gradient_parameters': sorted(gradients),
                'torch_imported': 'torch' in sys.modules, 'torchao_imported': 'torchao' in sys.modules,
                'output_rms': float(np.sqrt(np.mean(output_arrays[0] ** 2))),
                'baseline_memory': baseline, 'after_backward_memory': backend.memory(),
                'note': 'One block correctness smoke test, not full LLM training or MFU measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=17)
    parser.add_argument('--fp8', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.seq_len <= 32768:
        parser.error('device must be nonnegative; seq_len must be in 1..32768')
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
