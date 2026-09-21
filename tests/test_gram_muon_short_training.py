"""Short-training A/B: Gram Newton-Schulz vs the shipped Muon iteration.

Final gate for docs/GRAM_NEWTON_SCHULZ.md: the math equivalence proven in
``test_gram_newton_schulz.py`` must survive the *full production optimizer
machinery* — ``build_optimizer``'s Muon/AdamW split, momentum + nesterov,
foreach updates, shape bucketing, chunking, autocast-off — over a multi-step
trajectory, not just one call.

The NS kernel is swapped by patching ``trainer.batched_muon._batched_zeropower``
(the symbol ``_step_chunk`` calls); no production file is modified. Everything
runs on CPU with a fixed seed and ``dropout=0`` math attention, so identical
configurations reproduce bit-exactly — that determinism is itself asserted as
the harness control.

Data is periodic (period-8 repeated random bases): next-token prediction has
real signal, so "both configurations learn" is checkable (random tokens sit on
the ln(vocab) entropy floor with nothing to learn).

Comparison ladder (mirrors the equivalence ladder of the unit tests):

1. Control: two production runs are bit-identical (validates the harness).
2. fp64 kernels: standard vs Gram trajectories agree to rounding level —
   equivalence holds across training dynamics, not just per call.
3. Production dtypes: Gram bf16 / recommended Gram fp16 (single restart after
   iteration 3) track the shipped bf16 path within training noise on the loss.
   Parameters need a calibrated criterion: top-1 routing is discrete, so ANY
   rounding difference — including running the standard algorithm in fp64 —
   flips token→expert assignments and diverges parameters chaotically
   (measured ≈0.5–0.75 after these steps; unordered — chaos saturates). Gram's
   divergence must stay within the band of a dtype-only change to the
   standard algorithm.
"""

from functools import lru_cache

import pytest
import torch

import trainer.batched_muon as batched_muon_module
from model.model_instinct import InstinctForCausalLM
from tests.helpers import make_tiny_config
from tests.test_gram_newton_schulz import gram_zeropower, standard_zeropower
from trainer.trainer_utils import build_optimizer

SEED = 20260922
STEPS = 25
LR = 0.02
BATCH, SEQ, PERIOD = 8, 96, 8

KERNELS = {"production": None, "standard": standard_zeropower, "gram": gram_zeropower}


def _make_batches(seed, steps):
    """Learnable synthetic data: each sample repeats a random period-8 base."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(steps):
        base = torch.randint(1, 256, (BATCH, PERIOD), generator=generator)
        batches.append(base.repeat(1, (SEQ + PERIOD - 1) // PERIOD)[:, :SEQ])
    return batches


def _resolve_kernel(name, dtype, restart_after):
    """Return an NS function with the production signature for this config."""
    if name == "production":
        return batched_muon_module._batched_zeropower
    kernel = KERNELS[name]

    def ns(update, *, ns_coefficients, ns_steps, eps):
        kwargs = {} if restart_after is None else {"restart_after": restart_after}
        return kernel(update, ns_coefficients=ns_coefficients, ns_steps=ns_steps,
                      eps=eps, dtype=dtype, **kwargs)

    return ns


@lru_cache(maxsize=None)
def _run_short_training(name, dtype, restart_after, *, seed=SEED, steps=STEPS, lr=LR):
    """Train a tiny MoE model on fixed synthetic data; return (losses, params).

    All configs share weight init, data, optimizer construction and the loss
    path ``res.loss + res.aux_loss`` used by ``train_pretrain.py``. Results are
    cached per configuration: every test below reuses the same trajectories.
    """
    torch.manual_seed(seed)
    config = make_tiny_config(use_moe=True)
    model = InstinctForCausalLM(config).train()
    batches = _make_batches(seed + 1, steps)
    optimizer = build_optimizer(model.named_parameters(), lr=lr, optimizer="muon")

    ns_fn = _resolve_kernel(name, dtype, restart_after)
    original = batched_muon_module._batched_zeropower
    batched_muon_module._batched_zeropower = ns_fn
    try:
        losses = []
        for tokens in batches:
            res = model(tokens, labels=tokens)
            loss = res.loss + res.aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    finally:
        batched_muon_module._batched_zeropower = original

    params = torch.cat([
        p.detach().flatten() for _, p in sorted(model.named_parameters())
    ]).clone()
    return losses, params


def _max_rel_loss_diff(a, b):
    a, b = torch.tensor(a), torch.tensor(b)
    return ((a - b).abs() / a.abs().clamp(min=1e-3)).max().item()


def _param_rel_distance(p, q):
    return ((p - q).norm() / q.norm().clamp(min=1e-12)).item()


def test_harness_is_deterministic():
    """Identical configurations must reproduce bit-exactly, else every A/B
    difference below could be harness noise."""
    losses_a, params_a = _run_short_training("production", None, None)
    # Same config through the cache-miss path: clear and rerun.
    _run_short_training.cache_clear()
    losses_b, params_b = _run_short_training("production", None, None)

    assert losses_a == losses_b
    assert torch.equal(params_a, params_b)
    _run_short_training.cache_clear()  # repopulate shared cache for later tests


def test_fp64_trajectories_match():
    """Same map in exact arithmetic: training with a fp64 standard kernel vs a
    fp64 Gram kernel (single restart, a rounding-level no-op in fp64) must stay
    at rounding-level agreement across the whole trajectory."""
    losses_std, params_std = _run_short_training("standard", torch.float64, None)
    losses_gram, params_gram = _run_short_training("gram", torch.float64, 3)

    assert _max_rel_loss_diff(losses_gram, losses_std) < 1e-6
    assert _param_rel_distance(params_gram, params_std) < 1e-6


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"],
)
def test_production_dtype_trajectories_within_noise(dtype):
    """The shipped bf16 path vs Gram in a working dtype with the recommended
    single restart after iteration 3.

    Loss agreement and learning progress use fixed thresholds. Parameter
    trajectories use a calibrated criterion (see module docstring): swapping
    the algorithm must perturb the trajectory no more than swapping dtype
    inside the standard algorithm does."""
    losses_std, params_std = _run_short_training("production", None, None)
    losses_gram, params_gram = _run_short_training("gram", dtype, 3)
    _, params_std_fp16 = _run_short_training("standard", torch.float16, None)

    assert all(torch.isfinite(torch.tensor(losses_gram)))
    assert _max_rel_loss_diff(losses_gram, losses_std) < 0.05
    assert _param_rel_distance(params_gram, params_std) <= 1.25 * _param_rel_distance(
        params_std_fp16, params_std)

    # Both configurations must learn the periodic structure, not just agree.
    for losses in (losses_std, losses_gram):
        assert min(losses[-5:]) < losses[0] - 0.25


def test_parameter_divergence_is_routing_chaos_not_algorithm():
    """Pin the chaos finding as a guard: the fp64 *standard* algorithm —
    strictly more accurate than production — already diverges from production
    parameters by O(0.1–1) after 25 steps (discrete top-1 routing amplifies
    any rounding difference), and Gram sits inside that natural band. If this
    ever fails, the divergence regime changed and the calibrated thresholds
    above need re-deriving."""
    _, params_prod = _run_short_training("production", None, None)
    _, params_exact = _run_short_training("standard", torch.float64, None)
    _, params_std_fp16 = _run_short_training("standard", torch.float16, None)
    _, params_gram = _run_short_training("gram", torch.float16, 3)

    exact_divergence = _param_rel_distance(params_exact, params_prod)
    dtype_divergence = _param_rel_distance(params_std_fp16, params_prod)
    gram_divergence = _param_rel_distance(params_gram, params_prod)

    assert 0.1 < exact_divergence < 2.0            # chaos regime is present
    natural_band = max(exact_divergence, dtype_divergence)
    assert gram_divergence < 1.25 * natural_band
