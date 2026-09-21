"""Shape-batched Muon with native PyTorch checkpoint layout.

Only temporary updates are batched; model parameters and momentum buffers stay
independent. This preserves parameter IDs/order in existing resume checkpoints.
"""

from __future__ import annotations

import math
import os

import torch


_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

# Gram-form NS restarts after this many iterations to repair half-precision
# drift (a rounding-level no-op in fp64); ns_steps=5 therefore restarts once.
_GRAM_RESTART_AFTER = 3
# The Gram form only pays off once the short side is large enough to amortize
# its extra m×m products; smaller and square matrices stay on the classic path.
_GRAM_MIN_SHORT_SIDE = 256


def gram_newton_schulz_enabled() -> bool:
    """Whether the Gram-form NS kernel is active (``INSTINCT_MUON_GRAM=0`` restores the classic iteration)."""
    return os.environ.get("INSTINCT_MUON_GRAM", "1").strip().lower() not in ("0", "false", "off", "no")


def _standard_zeropower(update, *, ns_coefficients, ns_steps, eps):
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


def _gram_zeropower(update, *, ns_coefficients, ns_steps, eps):
    """Gram-form quintic Newton-Schulz; identical map to the classic iteration.

    docs/GRAM_NEWTON_SCHULZ.md: the odd polynomial p(x) = x·h(x²) lets the
    Gram matrix A = XXᵀ evolve on its own (A ← Z·A·Z with Z = aI + bA + cA²)
    while Q accumulates the Z factors, so only two rectangular m×n matmuls
    remain (A₀ = XXᵀ and the final Q·X). Equivalence validated to ~1e-15 in
    fp64 and by short-training A/B. A restart after
    ``_GRAM_RESTART_AFTER`` iterations repairs half-precision drift and is
    exact in fp64.

    Iterates in fp16 on CUDA (measured faster and more accurate than bf16),
    bf16 elsewhere. Normalization happens in the input dtype before the cast,
    so entries are ≤ 1 and cannot overflow the fp16 range.
    """
    x = update
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    # Never normalize over the batch dimension: experts must remain independent.
    x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=eps)
    compute_dtype = torch.float16 if x.is_cuda else torch.bfloat16
    x = x.to(compute_dtype)
    a, b, c = ns_coefficients
    batch, m, _ = x.shape
    eye = torch.eye(m, dtype=compute_dtype, device=x.device)
    gram = torch.bmm(x, x.transpose(-2, -1))
    Q = eye.expand(batch, m, m).clone()
    done = 0
    while done < ns_steps:
        run = _GRAM_RESTART_AFTER if ns_steps - done > _GRAM_RESTART_AFTER else ns_steps - done
        for _ in range(run):
            Z = torch.baddbmm(gram, gram, gram, beta=b, alpha=c) + a * eye
            Q = torch.bmm(Q, Z)
            gram = torch.bmm(torch.bmm(Z, gram), Z)
            done += 1
        if done < ns_steps:  # stabilization restart: exact in fp64
            x = torch.bmm(Q, x)
            gram = torch.bmm(x, x.transpose(-2, -1))
            Q = eye.expand(batch, m, m).clone()
    x = torch.bmm(Q, x)
    return x.transpose(-2, -1) if transposed else x


def _batched_zeropower(update, *, ns_coefficients, ns_steps, eps):
    """Orthogonalize each update matrix, dispatching between NS kernels.

    Rectangular matrices with a short side ≥ ``_GRAM_MIN_SHORT_SIDE`` use the
    Gram-form kernel by default (``INSTINCT_MUON_GRAM=0`` restores the classic
    iteration everywhere). Square matrices, tiny routers and CPU fallbacks
    stay on the classic path — measured faster there.
    """
    if (
        gram_newton_schulz_enabled()
        and update.size(-2) != update.size(-1)
        and min(update.shape[-2:]) >= _GRAM_MIN_SHORT_SIDE
    ):
        return _gram_zeropower(
            update, ns_coefficients=ns_coefficients, ns_steps=ns_steps, eps=eps,
        )
    return _standard_zeropower(
        update, ns_coefficients=ns_coefficients, ns_steps=ns_steps, eps=eps,
    )


class BatchedMuon(torch.optim.Optimizer):
    """Muon using foreach momentum/weight updates and batched matmuls.

    Large rectangular updates orthogonalize through the Gram-form kernel
    (fp16 on CUDA, bf16 elsewhere; see docs/GRAM_NEWTON_SCHULZ.md); other
    shapes keep the classic bf16 quintic iteration. ``INSTINCT_MUON_GRAM=0``
    restores the classic iteration everywhere.

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
