"""Offline-compile a default-width Dense block forward/backward without CUDA or torch.

python -m experiments.compile_dense_triton --arch 89 --fp8 1
This verifies compilation, not GPU execution, memory safety or throughput.
"""
import argparse
import json
import sys

from model.dense_triton.ops import BlockConfig, DenseBlock, DenseOps
from model.dense_triton.runtime import Tensor, compile_kernel, constants_for
from model.dense_triton import elementwise as ew


def weight_shapes(c):
    d, f, q, k = c.hidden_size, c.intermediate_size, c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    return {'input_layernorm.weight': (d,), 'post_attention_layernorm.weight': (d,),
            'self_attn.q_norm.weight': (c.head_dim,), 'self_attn.k_norm.weight': (c.head_dim,),
            'self_attn.q_proj.weight': (q, d), 'self_attn.k_proj.weight': (k, d),
            'self_attn.v_proj.weight': (k, d), 'self_attn.o_proj.weight': (d, q),
            'mlp.gate_proj.weight': (f, d), 'mlp.up_proj.weight': (f, d), 'mlp.down_proj.weight': (d, f)}


class CompileBackend:
    def __init__(self, arch):
        self.arch, self.kernels = arch, {}

    def empty(self, shape, dtype, zero=False):
        return Tensor(tuple(shape), dtype, None, self)

    def launch(self, fn, grid, args, **constants):
        constants = constants_for(fn, constants)
        types = tuple(t.dtype for t in args)
        key = (fn.__name__, types, tuple(sorted(constants.items())))
        if key not in self.kernels:
            kernel = compile_kernel(fn, types, constants, self.arch)
            meta = kernel.metadata
            assert meta.global_scratch_size == 0 and meta.num_ctas == 1
            assert meta.cluster_dims == (1, 1, 1)
            self.kernels[key] = {'name': meta.name, 'shared_bytes': meta.shared,
                                 'cubin_bytes': len(kernel.asm['cubin'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', type=int, choices=[80, 86, 89, 90, 100, 120], default=89)
    parser.add_argument('--fp8', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    if args.fp8 and args.arch < 89:
        parser.error('native FP8 requires SM89+')
    backend = CompileBackend(args.arch)
    c = BlockConfig()
    weights = {n: backend.empty(s, 'fp32') for n, s in weight_shapes(c).items()}
    op = DenseOps(backend, fp8=bool(args.fp8))
    block = DenseBlock(op, weights, c)
    # Nonmultiples of the attention/FP8 tiles check masked tails in both passes.
    x = backend.empty((1, 35, c.hidden_size), 'bf16')
    cos, sin = [backend.empty((35, c.head_dim), 'fp32') for _ in range(2)]
    mask = backend.empty((1, 35), 'i32')
    out, saved = block.forward(x, cos, sin, mask)
    block.backward(saved, out)
    ids, table = backend.empty((35,), 'i64'), backend.empty((6400, 768), 'fp32')
    for fn, output in [(ew.embedding, x), (ew.embedding_backward, table)]:
        buffers = (ids, table, output) if fn is ew.embedding else (ids, x, output)
        backend.launch(fn, (35, 3), buffers, TOKENS=35, DIM=768, VOCAB=6400)
    logits = backend.empty((35, 6400), 'bf16')
    backend.launch(ew.cross_entropy, (35,), (logits, ids, backend.empty((35,), 'fp32'),
                    backend.empty((35,), 'i32'), logits), VOCAB=6400, BLOCK=8192)
    assert 'torch' not in sys.modules and 'torchao' not in sys.modules
    print(json.dumps({'arch': args.arch, 'fp8': bool(args.fp8), 'torch_imported': False,
                      'specializations': len(backend.kernels),
                      'kernels': sorted({v['name'] for v in backend.kernels.values()}),
                      'max_shared_bytes': max(v['shared_bytes'] for v in backend.kernels.values())}, indent=2))


if __name__ == '__main__':
    main()
