"""Shape-batched Muon with native PyTorch checkpoint layout.

Only temporary updates are batched; model parameters and momentum buffers stay
independent. This preserves parameter IDs/order in existing resume checkpoints.
"""

from __future__ import annotations

import math

import torch


_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _batched_zeropower(update, *, ns_coefficients, ns_steps, eps):
    """Apply the native quintic iteration independently to each matrix."""
    x = update.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    # Never normalize over the batch dimension: experts must remain independent.
    x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=eps)
    a, b, c = ns_coefficients
    for _ in range(ns_steps):
        gram = torch.bmm(x, x.transpose(-2, -1))
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, polynomial, x, beta=a)
    return x.transpose(-2, -1) if transposed else x


class BatchedMuon(torch.optim.Optimizer):
    """Muon using foreach momentum/weight updates and batched BF16 matmuls.

    ``workspace_mb`` bounds a conservative estimate of temporary tensor storage
    per chunk, not total CUDA allocation (parameters, momentum and library
    workspaces are additional). One matrix is the minimum chunk size.
    Runtime chunk limits are deliberately not checkpointed: native Muon state
    can be loaded without losing the limits chosen for the current machine.
    """

    def __init__(
        self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
        nesterov=True, ns_coefficients=_NS_COEFFICIENTS, ns_steps=5,
        eps=1e-7, adjust_lr_fn=None, *, batch_size=16, workspace_mb=64,
    ):
        if not 0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= momentum:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not isinstance(ns_steps, int) or not 0 <= ns_steps < 100:
            raise ValueError("ns_steps must be an integer in [0, 100)")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must contain three values")
        if not eps > 0:
            raise ValueError("eps must be positive")
        if adjust_lr_fn not in (None, 'original', 'match_rms_adamw'):
            raise ValueError(f"Invalid adjust_lr_fn: {adjust_lr_fn}")
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(workspace_mb) or workspace_mb <= 0:
            raise ValueError("workspace_mb must be finite and positive")
        self.batch_size = int(batch_size)
        self.workspace_bytes = int(workspace_mb * 2**20)
        super().__init__(params, dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_coefficients=tuple(ns_coefficients),
            ns_steps=ns_steps, eps=eps, adjust_lr_fn=adjust_lr_fn,
        ))
        for group in self.param_groups:
            for p in group['params']:
                self._validate_parameter(p)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            # The older in-repo fallback did not serialize this native field.
            group.setdefault('ns_coefficients', _NS_COEFFICIENTS)

    @staticmethod
    def _validate_parameter(p):
        if p.ndim != 2 or not p.is_floating_point() or p.layout != torch.strided:
            raise ValueError("Muon requires real floating-point, strided 2D parameters")

    def _chunk_size(self, p):
        # Nesterov list + stacked update + cast, with room for foreach temporaries
        # and the BF16 Gram/polynomial matrices used by Newton-Schulz.
        elements = p.numel()
        short = min(p.shape)
        per_matrix = (3 * p.element_size() + 4) * elements + 8 * short * short
        return max(1, min(self.batch_size, self.workspace_bytes // max(1, per_matrix)))

    def _step_chunk(self, params, group):
        grads = [p.grad for p in params]
        buffers = []
        for p in params:
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p.grad)
            buffers.append(state['momentum_buffer'])
        momentum = group['momentum']
        torch._foreach_lerp_(buffers, grads, 1 - momentum)
        if group['nesterov']:
            updates = torch._foreach_lerp(grads, buffers, momentum)
            stacked = torch.stack(updates).to(torch.bfloat16)
            del updates
        else:
            stacked = torch.stack(buffers).to(torch.bfloat16)
        update = _batched_zeropower(
            stacked,
            ns_coefficients=group.get('ns_coefficients', _NS_COEFFICIENTS),
            ns_steps=group['ns_steps'], eps=group['eps'],
        )
        del stacked
        lr = group['lr']
        rows, cols = params[0].shape
        if group['adjust_lr_fn'] == 'match_rms_adamw':
            adjusted_lr = lr * 0.2 * math.sqrt(max(rows, cols))
        else:
            adjusted_lr = lr * math.sqrt(max(1.0, rows / cols))
        torch._foreach_mul_(params, 1 - lr * group['weight_decay'])
        # Match dtypes so foreach can use its fused CUDA path, not a per-tensor
        # fallback for mixed FP32 parameters / BF16 updates.
        # Tall matrices were transposed back above; materialize row-major slices
        # during the cast so standard Linear parameters also match foreach's
        # stride requirements.
        update = update.to(params[0].dtype, memory_format=torch.contiguous_format)
        torch._foreach_add_(params, list(update.unbind(0)), alpha=-adjusted_lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = {}
            for p in group['params']:
                if p.grad is None:
                    continue
                self._validate_parameter(p)
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                key = (p.device, p.dtype, tuple(p.shape))
                buckets.setdefault(key, []).append(p)
            for params in buckets.values():
                size = self._chunk_size(params[0])
                # Muon's iteration precision must not depend on the caller's
                # activation autocast context.
                with torch.autocast(device_type=params[0].device.type, enabled=False):
                    for start in range(0, len(params), size):
                        self._step_chunk(params[start:start + size], group)
        return loss
