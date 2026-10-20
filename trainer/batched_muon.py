"""Shape-batched Muon with native PyTorch checkpoint layout.

Only temporary updates are batched; model parameters and momentum buffers stay
independent. This preserves parameter IDs/order in existing resume checkpoints.

Mantissa mode (``use_mantissa=True``, env ``INSTINCT_MUON_MANTISSA=1`` through
``build_optimizer``) lets bfloat16 parameters receive full fp32 updates: a bf16
parameter holds the high 16 bits of its fp32 value and a parallel uint16 buffer
holds the low 16 bits. The split is a truncating bit split, so split-then-
reconstruct is a bitwise identity, and every optimizer update accumulates on the
reconstructed fp32 value before being split back. This costs four fp32
momentum bytes (or two with ``momentum_dtype='bf16'``) plus two uint16 low bytes per element (the persistent-byte
estimator is kept in sync). Mantissa mode is opt-in. Checkpoints record the bit
layout and reject a resume that would discard it. The forward weight is truncated
toward zero, unlike the round-to-nearest weight produced by BF16 autocast; exact
optimizer arithmetic with identical gradients does not imply identical training.
"""

from __future__ import annotations

import math
import os

import torch


_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_V2_SHAPES = {(1536, 1536), (384, 1536), (5504, 1536), (1536, 5504)}
_VALIDATED_SHAPES = {(512, 512), (128, 512), (1664, 512), (512, 1664)} | _V2_SHAPES
_MANTISSA_FORMAT = 'bf16_uint16_truncate_v1'
_MANTISSA_BF16_MOMENTUM_FORMAT = 'bf16_uint16_truncate_bf16_momentum_v1'


def has_mantissa_state(state_dict):
    """Recognize both versioned and original, unversioned mantissa checkpoints."""
    return (state_dict.get('muon_precision') is not None
            or any(isinstance(entry, dict) and 'mantissa' in entry
                   for entry in state_dict.get('state', {}).values())
            or any(has_mantissa_state(sd) for sd in state_dict.get('optimizers', [])))


def reject_mantissa_state(optimizer, state_dict):
    """Load pre-hook for optimizers that cannot interpret split weight storage."""
    if has_mantissa_state(state_dict):
        raise ValueError(
            'Muon mantissa checkpoint requires BatchedMuon(use_mantissa=True), '
            'INSTINCT_MUON_BACKEND=batched and INSTINCT_MUON_MANTISSA=1; '
            'refusing to discard the saved uint16 low bits.'
        )


def _ns_backend(update):
    backend = os.environ.get('INSTINCT_MUON_NS_BACKEND', 'auto')
    if backend not in ('torch', 'triton', 'triton_graph', 'auto'):
        raise ValueError('INSTINCT_MUON_NS_BACKEND must be torch, triton, triton_graph or auto')
    if backend == 'auto':
        from model.cuda_fastpath import blackwell_training_kernels
        shape = tuple(update.shape[-2:])
        backend = 'torch'
        if shape in _VALIDATED_SHAPES and blackwell_training_kernels(update.device):
            # V2's large matrices already amortize launches; retaining separate
            # graph pools for their chunks costs activation/checkpoint headroom.
            backend = 'triton' if shape in _V2_SHAPES else 'triton_graph'
    return backend


# Shift/bitwise kernels are not implemented for unsigned 32-bit tensors on every
# torch build, so the split works on the signed bitcast of the same layout.
# Truncating the shifted result back to 16 bits keeps the raw bit pattern, so the
# split/reconstruct pair is bit-exact (not a lossy cast).
def _reconstruct_precise(param, mantissa):
    """Recombine bfloat16 param + uint16 mantissa into the exact float32 value."""
    bits = param.contiguous().view(torch.int16).to(torch.int32)
    bits = torch.bitwise_left_shift(bits, 16) | mantissa.to(torch.int32)
    return bits.view(torch.float32)


def _split_precise(precise):
    """Truncating split of float32 into (bfloat16 high 16 bits, uint16 low 16 bits)."""
    bits = precise.contiguous().view(torch.int32)
    high = torch.bitwise_right_shift(bits, 16).to(torch.int16).view(torch.bfloat16)
    low = torch.bitwise_and(bits, 0xFFFF).to(torch.uint16)
    return high, low


def _batched_zeropower(update, *, ns_coefficients, ns_steps, eps):
    backend = _ns_backend(update)
    if (backend != 'torch' and update.is_cuda and update.ndim == 3
            and min(update.shape[-2:]) >= 128):
        from trainer.muon_kernels import zeropower
        return zeropower(update, ns_coefficients=ns_coefficients, ns_steps=ns_steps, eps=eps)
    return _batched_zeropower_reference(update, ns_coefficients=ns_coefficients,
                                       ns_steps=ns_steps, eps=eps)


def _batched_zeropower_reference(update, *, ns_coefficients, ns_steps, eps):
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
        use_mantissa=False, momentum_dtype='fp32',
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
        self.use_mantissa = bool(use_mantissa)
        if momentum_dtype not in ('fp32', 'bf16'):
            raise ValueError('momentum_dtype must be fp32 or bf16')
        if momentum_dtype == 'bf16' and not self.use_mantissa:
            raise ValueError('Explicit BF16 momentum requires use_mantissa=True')
        self.momentum_dtype = torch.bfloat16 if momentum_dtype == 'bf16' else torch.float32
        super().__init__(params, dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_coefficients=tuple(ns_coefficients),
            ns_steps=ns_steps, eps=eps, adjust_lr_fn=adjust_lr_fn,
        ))
        for group in self.param_groups:
            for p in group['params']:
                self._validate_parameter(p)
                # The mantissa bit layout is bf16-specific (fp32's high 16 bits).
                if self.use_mantissa and p.dtype != torch.bfloat16:
                    raise ValueError(
                        "BatchedMuon(use_mantissa=True) requires bfloat16 parameters"
                    )

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('momentum_dtype', torch.float32)
        # The pure NS cache contains no optimizer state. Drop device allocations
        # on load and rebuild lazily for the restored runtime/device.
        self.__dict__.pop('_ns_graph_cache', None)
        for group in self.param_groups:
            # The older in-repo fallback did not serialize this native field.
            group.setdefault('ns_coefficients', _NS_COEFFICIENTS)

    @torch.no_grad()
    def reserve_momentum(self):
        """Allocate optimizer state before activation buffers fragment the pool.

        Does not take a step, create gradients, apply decay, or replace restored
        state. Mantissa mode reserves the selected momentum dtype and uint16 low bits.
        Parameters with grad=None still skip every update in step().
        """
        allocated = 0
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad or not p.is_contiguous():
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    dtype = self.momentum_dtype if self.use_mantissa else p.dtype
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=dtype)
                    allocated += p.numel() * state['momentum_buffer'].element_size()
                elif self.use_mantissa and state['momentum_buffer'].dtype != self.momentum_dtype:
                    state['momentum_buffer'] = state['momentum_buffer'].to(self.momentum_dtype)
                    allocated += p.numel() * state['momentum_buffer'].element_size()
                if self.use_mantissa and 'mantissa' not in state:
                    state['mantissa'] = torch.zeros_like(p, dtype=torch.uint16)
                    allocated += p.numel() * state['mantissa'].element_size()
        return allocated

    def state_dict(self):
        state = super().state_dict()
        if self.use_mantissa:
            state['muon_precision'] = (_MANTISSA_BF16_MOMENTUM_FORMAT
                                      if self.momentum_dtype == torch.bfloat16 else _MANTISSA_FORMAT)
        return state

    def load_state_dict(self, state_dict):
        # The base loader casts every floating state tensor to the bf16 param
        # dtype. Keep the selected momentum precision and exact uint16 low bits.
        if not self.use_mantissa:
            reject_mantissa_state(self, state_dict)
        precision = state_dict.get('muon_precision')
        if precision is not None and precision not in (_MANTISSA_FORMAT, _MANTISSA_BF16_MOMENTUM_FORMAT):
            raise ValueError(f'Unsupported Muon mantissa checkpoint format: {precision!r}')
        saved_momentum_dtype = (torch.bfloat16 if precision == _MANTISSA_BF16_MOMENTUM_FORMAT
                                else torch.float32)
        if precision is not None and saved_momentum_dtype != self.momentum_dtype:
            raise ValueError('Muon momentum dtype changed on resume; select the saved momentum_dtype '
                             '(INSTINCT_MUON_MOMENTUM_DTYPE), or start a new stage from weights.')
        saved_ids = [pid for group in state_dict['param_groups'] for pid in group['params']]
        current = [p for group in self.param_groups for p in group['params']]
        index_map = dict(zip(saved_ids, current))
        saved = {}
        cleaned = dict(state_dict)
        cleaned['state'] = dict(state_dict.get('state', {}))
        split_weights = has_mantissa_state(state_dict)
        if self.use_mantissa:
            for index, entry in state_dict.get('state', {}).items():
                if not isinstance(entry, dict):
                    continue
                param = index_map.get(index)
                if param is None:
                    raise ValueError('Muon checkpoint contains state for an unknown parameter')
                if split_weights and entry and 'mantissa' not in entry:
                    raise ValueError('Muon mantissa checkpoint is missing uint16 low bits')
                picked = {
                    key: entry[key]
                    for key in ('momentum_buffer', 'mantissa')
                    if key in entry
                }
                for key, tensor in picked.items():
                    if not isinstance(tensor, torch.Tensor) or tensor.shape != param.shape:
                        raise ValueError(f'Invalid Muon {key} tensor shape')
                    if key == 'mantissa' and tensor.dtype != torch.uint16:
                        raise ValueError('Muon mantissa must be uint16; refusing corrupted low bits')
                    if key == 'momentum_buffer' and tensor.dtype not in (torch.float32, torch.bfloat16):
                        raise ValueError('Muon mantissa momentum must be float32 or legacy bfloat16')
                    if key == 'momentum_buffer':
                        if precision is not None and tensor.dtype != saved_momentum_dtype:
                            raise ValueError('Muon checkpoint momentum dtype disagrees with its precision format')
                        if self.momentum_dtype == torch.bfloat16 and tensor.dtype != torch.bfloat16:
                            raise ValueError('Muon momentum dtype changed on resume; refusing implicit BF16 quantization')
                        if precision is None and split_weights and tensor.dtype != self.momentum_dtype:
                            raise ValueError('Muon momentum dtype changed on resume from unversioned split state')
                if picked:
                    saved[index] = picked
                    # Avoid a temporary, lossy BF16 copy of both saved tensors.
                    cleaned['state'][index] = {k: v for k, v in entry.items() if k not in picked}
        super().load_state_dict(cleaned)
        if not saved:
            return
        for index, entry in saved.items():
            param = index_map.get(index)
            if param is None or param not in self.state:
                continue
            state = self.state[param]
            for key, tensor in entry.items():
                # Keep saved FP32/BF16 momentum and raw uint16 low bits.
                state[key] = tensor.to(device=param.device)

    @staticmethod
    def _validate_parameter(p):
        if p.ndim != 2 or not p.is_floating_point() or p.layout != torch.strided:
            raise ValueError("Muon requires real floating-point, strided 2D parameters")

    def _chunk_size(self, p):
        # Nesterov list + stacked update + cast, with room for foreach temporaries
        # and the BF16 Gram/polynomial matrices used by Newton-Schulz.
        elements = p.numel()
        short = min(p.shape)
        # Mantissa mode adds fp32 promoted grads + reconstructed precises per matrix.
        extra = 12 if self.use_mantissa else 4
        per_matrix = (3 * p.element_size() + extra) * elements + 8 * short * short
        return max(1, min(self.batch_size, self.workspace_bytes // max(1, per_matrix)))

    def _step_chunk(self, params, group):
        grads = [p.grad for p in params]
        buffers = []
        mantissas = []
        for i, p in enumerate(params):
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(
                    grads[i], dtype=self.momentum_dtype if self.use_mantissa else grads[i].dtype)
            if self.use_mantissa and state['momentum_buffer'].dtype != self.momentum_dtype:
                # A legacy (non-mantissa) checkpoint stores bf16 momentum; promote
                # it once so the fp32 lerp path stays uniform with promoted grads.
                state['momentum_buffer'] = state['momentum_buffer'].to(self.momentum_dtype)
            buffers.append(state['momentum_buffer'])
            if self.use_mantissa:
                if 'mantissa' not in state:
                    # Zero low bits == the legacy precise value on a fresh/old checkpoint.
                    state['mantissa'] = torch.zeros_like(p, dtype=torch.uint16)
                mantissas.append(state['mantissa'])
        momentum = group['momentum']
        fused_setting = os.environ.get('INSTINCT_MUON_FUSED_UPDATES', 'auto')
        if fused_setting not in ('0', '1', 'auto'):
            raise ValueError('INSTINCT_MUON_FUSED_UPDATES must be 0, 1 or auto')
        from model.cuda_fastpath import blackwell_training_kernels
        fused = ((fused_setting == '1' or (fused_setting == 'auto'
                  and tuple(params[0].shape) in _VALIDATED_SHAPES
                  and blackwell_training_kernels(params[0].device)))
                 and params[0].is_cuda
                 and (params[0].dtype == torch.float32 or self.use_mantissa)
                 and all(t.is_contiguous() for t in (*params, *grads, *buffers, *mantissas)))
        if self.use_mantissa and not fused:
            # The reference foreach path needs actual FP32 tensors. Fused CUDA
            # promotes in registers, avoiding one full-size copy per gradient.
            grads = [g.to(torch.float32) for g in grads]
        stored_buffers = buffers
        if self.use_mantissa and self.momentum_dtype == torch.bfloat16 and not fused:
            # Reference path: compute the EMA/Nesterov in FP32, then round only
            # the persistent EMA. Fused CUDA performs this promotion in registers.
            buffers = [b.float() for b in buffers]
        if fused:
            from trainer.muon_kernels import prepare_update
            stacked = prepare_update(grads, buffers, momentum=momentum, nesterov=group['nesterov'])
        elif group['nesterov']:
            torch._foreach_lerp_(buffers, grads, 1 - momentum)
            updates = torch._foreach_lerp(grads, buffers, momentum)
            stacked = torch.stack(updates).to(torch.bfloat16)
            del updates
        else:
            torch._foreach_lerp_(buffers, grads, 1 - momentum)
            stacked = torch.stack(buffers).to(torch.bfloat16)
        if buffers is not stored_buffers:
            for stored, precise in zip(stored_buffers, buffers):
                stored.copy_(precise)
            del buffers, precise
        ns_function = _batched_zeropower
        if (_ns_backend(stacked) == 'triton_graph'
                and stacked.is_cuda and min(stacked.shape[-2:]) >= 128):
            from trainer.muon_kernels import ZeropowerGraphCache
            if not hasattr(self, '_ns_graph_cache'):
                self._ns_graph_cache = ZeropowerGraphCache()
            ns_function = self._ns_graph_cache
        update = ns_function(
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
        if self.use_mantissa:
            if fused:
                from trainer.muon_kernels import apply_mantissa_update
                apply_mantissa_update(params, mantissas, update,
                                      decay=1 - lr * group['weight_decay'], alpha=-adjusted_lr)
                return
            # Reconstruct the exact fp32 value, apply decay/update there, then split
            # back. p.copy_ preserves the Parameter identity for the model.
            precises = [_reconstruct_precise(p, m) for p, m in zip(params, mantissas)]
            torch._foreach_mul_(precises, 1 - lr * group['weight_decay'])
            update = update.to(torch.float32, memory_format=torch.contiguous_format)
            torch._foreach_add_(precises, list(update.unbind(0)), alpha=-adjusted_lr)
            for p, m, precise in zip(params, mantissas, precises):
                high, low = _split_precise(precise)
                p.copy_(high)
                m.copy_(low)
            return
        if fused:
            from trainer.muon_kernels import apply_update
            apply_update(params, update, decay=1 - lr * group['weight_decay'], alpha=-adjusted_lr)
            return
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
                if self.use_mantissa and p.dtype != torch.bfloat16:
                    raise ValueError(
                        "BatchedMuon(use_mantissa=True) requires bfloat16 parameters"
                    )
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
