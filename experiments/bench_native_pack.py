"""Torch-free FP8 transpose validation and device timing; no weight updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/results/cublas/pack_tiles.json')
    args = parser.parse_args()
    sys.meta_path.insert(0, RejectTorch())
    import triton
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas import kernels
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(608)
    with TrainingBackend(allocator_backend='async') as backend:
        ops = CublasDenseOps(backend)
        for shape in ((17, 33), (1536, 384), (5504, 1536), (12288, 5504)):
            for dtype in ('fp8e4nv', 'fp8e5'):
                mark = backend.mark()
                source = backend.from_numpy(rng.standard_normal(shape, dtype=np.float32))
                source, scale = ops.quantize(source, dtype)
                nr, nc = shape[1], shape[0]
                pr, pc = triton.cdiv(nr, 16) * 16, triton.cdiv(nc, 16) * 16
                expected = np.zeros((pr, pc), dtype=np.uint8)
                expected[:nr, :nc] = backend.to_numpy(source).T
                options = dict(ROWS=nr, COLS=nc, PAD_ROWS=pr, PAD_COLS=pc,
                               STRIDE_ROW=1, STRIDE_COL=shape[1])
                for tile in (0, 32, 64, 128):
                    out = backend.empty((pr, pc), dtype)
                    fn = kernels.pack_matrix_tiled if tile else kernels.pack_matrix
                    grid = (triton.cdiv(pr, tile), triton.cdiv(pc, tile)) if tile else (triton.cdiv(pr*pc, 256),)
                    kwargs = options | ({'TILE': tile} if tile else {})
                    call = lambda: backend.launch(fn, grid, (source, out), **kwargs)
                    call()
                    np.testing.assert_array_equal(backend.to_numpy(out), expected)
                    for _ in range(2): call()
                    samples = [backend.time_call(call, backend.main_stream, repeats=10) for _ in range(3)]
                    row = dict(shape=shape, dtype=dtype, tile=tile,
                               median_ms=float(np.median(samples)), samples_ms=samples, bitwise_equal=True)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    target.write_text(json.dumps(dict(status='running', rows=rows), indent=2))
                backend.release_since(mark)
    target.write_text(json.dumps(dict(status='ok', rows=rows, torch_imported='torch' in sys.modules), indent=2))


if __name__ == '__main__': main()
