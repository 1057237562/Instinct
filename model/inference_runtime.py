"""Inference-only fused kernels shared across native model layers.

Compile stable numerical kernels, leaving streaming, cache growth and Python
control flow outside Dynamo. Parameters/state_dict and training remain intact.
"""
import sys
import torch
import torch.nn.functional as F


def rms_kernel(x, weight, eps):
    value = x.float()
    return (weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def swiglu_kernel(gate, up):
    return F.silu(gate) * up


class FusedKernel:
    def __init__(self, function, compiler=None):
        self.function = function
        self.compiler = compiler or torch.compile
        self.compiled = None
        self.failed = False

    def __call__(self, *args):
        if not self.failed:
            try:
                if self.compiled is None:
                    self.compiled = self.compiler(self.function, fullgraph=True, dynamic=True)
                return self.compiled(*args)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                self.failed = True
                self.compiled = None
                print(f'[Inference] {self.function.__name__} compile unavailable; using eager: {type(exc).__name__}: {str(exc)[:200]}', flush=True)
        return self.function(*args)


def optimize_inference(model, mode='auto'):
    if mode == 'off' or next(model.parameters()).device.type != 'cuda':
        return model
    if not type(model).__module__.startswith('model.model_instinct'):
        return model
    if sys.platform == 'win32' and not sys.flags.utf8_mode:
        print('[Inference] Windows compilation requires PYTHONUTF8=1 before launch; using eager.', flush=True)
        return model
    rms, gate = FusedKernel(rms_kernel), FusedKernel(swiglu_kernel)
    norms = gates = 0
    for layer in model.modules():
        if type(layer).__name__ == 'RMSNorm' and type(layer).__module__.startswith('model.model_instinct'):
            layer._inference_norm = rms
            norms += 1
        if type(layer).__name__ == 'FeedForward' and getattr(model.config, 'hidden_act', '') == 'silu':
            layer._inference_gate = gate
            gates += 1
    print(f'[Inference] Compiled kernels enabled: {norms} RMSNorm, {gates} SwiGLU; first use compiles/caches kernels.', flush=True)
    return model
