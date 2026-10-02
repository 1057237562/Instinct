"""Equivalence tests for the Gram Newton-Schulz rewrite (docs/GRAM_NEWTON_SCHULZ.md).

These tests validate the math BEFORE the production change: the Gram-form
candidate below must reproduce the quintic Newton-Schulz iteration currently
shipped in ``trainer/batched_muon.py::_batched_zeropower``.

Equivalence ladder, strongest first:

1. fp64: Gram iteration == standard iteration (identical map, different
   association order) — relative error at rounding level.
2. bf16: the dtype-flexible standard mirror is bitwise-identical to the
   production function, so conclusions transfer to the shipped code.
3. bf16/fp16: Gram's deviation from the fp64 reference is the same order as
   the standard iteration's own low-precision deviation (no precision
   regression); the iteration-3 restart keeps fp16 within budget.

When the rewrite lands in ``trainer/batched_muon.py`` these tests keep running
unchanged against the production symbol; only the import of ``gram_zeropower``
moves.
"""

import math

import pytest
import torch

from trainer.batched_muon import _NS_COEFFICIENTS, _batched_zeropower


def standard_zeropower(update, *, ns_coefficients=_NS_COEFFICIENTS, ns_steps=5,
                       eps=1e-7, dtype=torch.float64):
    """Dtype-flexible mirror of ``_batched_zeropower`` with identical op order.

    Accepts 2D or 3D input; single matrices are processed as a batch of one so
    the bmm/baddbmm call sequence matches production exactly.
    """
    squeeze = update.dim() == 2
    x = update.unsqueeze(0) if squeeze else update
    x = x.to(dtype)
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
    x = x.transpose(-2, -1) if transposed else x
    return x.squeeze(0) if squeeze else x


def gram_zeropower(update, *, ns_coefficients=_NS_COEFFICIENTS, ns_steps=5,
                   eps=1e-7, dtype=torch.float64, restart_after=None):
    """Gram-form candidate: iterate on A = X Xᵀ (m×m) with accumulated factor Q.

    Exact-arithmetic identity with ``standard_zeropower``: the quintic
    polynomial is odd, p(x) = x·h(x²), so X_{t+1} = h(A_t)·X_t implies
    A_{t+1} = h(A_t)·A_t·h(A_t) — the Gram matrix evolves alone while
    Q_t = Q_{t-1}·h(A_t) accumulates the orthogonal factor. Only two
    rectangular m×n matmuls remain (A₀ = XXᵀ and the final Q·X); all NS
    iterates are polynomials in A₀, hence mutually commuting.

    ``restart_after=k`` re-materializes X ← Q·X after k iterations, recomputes
    A = XXᵀ and resets Q = I. Exact math is unchanged; it only repairs
    half-precision drift (docs/GRAM_NEWTON_SCHULZ.md, stabilization).
    """
    squeeze = update.dim() == 2
    x = update.unsqueeze(0) if squeeze else update
    x = x.to(dtype)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=eps)
    a, b, c = ns_coefficients
    batch, m, _ = x.shape
    eye = torch.eye(m, dtype=x.dtype, device=x.device)

    def h_gram(A):
        # b·A + c·A² via the same baddbmm the standard iteration uses.
        polynomial = torch.baddbmm(A, A, A, beta=b, alpha=c)
        return polynomial + a * eye

    gram = torch.bmm(x, x.transpose(-2, -1))
    Q = eye.expand(batch, m, m).clone()
    done = 0
    while done < ns_steps:
        steps_left = ns_steps - done
        if restart_after is not None and steps_left > restart_after:
            run = restart_after
        else:
            run = steps_left
        for _ in range(run):
            Z = h_gram(gram)
            Q = torch.bmm(Q, Z)
            gram = torch.bmm(torch.bmm(Z, gram), Z)
            done += 1
        if done < ns_steps:  # stabilization restart
            x = torch.bmm(Q, x)
            gram = torch.bmm(x, x.transpose(-2, -1))
            Q = eye.expand(batch, m, m).clone()
    x = torch.bmm(Q, x)
    x = x.transpose(-2, -1) if transposed else x
    return x.squeeze(0) if squeeze else x


def _rel_err(actual, reference):
    diff = torch.linalg.vector_norm((actual - reference).to(torch.float64))
    denom = torch.linalg.vector_norm(reference.to(torch.float64))
    return (diff / denom.clamp(min=1e-30)).item()


# Shapes mirror the production aspect ratios (experts, qkv/o projections,
# router) at test-friendly sizes; the identity is shape-independent.
FP64_SHAPES = [
    (1, 8, 24),      # single wide matrix
    (3, 12, 4),      # tall, batched
    (2, 10, 10),     # square (alpha = 1)
    (4, 16, 40),     # expert-like aspect ratio
    (2, 3, 64),      # router-like extreme aspect ratio
]


@pytest.mark.parametrize("shape", FP64_SHAPES)
@pytest.mark.parametrize("ns_steps", [0, 1, 2, 5, 8])
def test_gram_matches_standard_fp64(shape, ns_steps):
    """The core equivalence proof: same map, different association order."""
    torch.manual_seed(0)
    update = torch.randn(*shape, dtype=torch.float64)

    standard = standard_zeropower(update, ns_steps=ns_steps)
    gram = gram_zeropower(update, ns_steps=ns_steps)

    assert _rel_err(gram, standard) < 1e-9, _rel_err(gram, standard)


@pytest.mark.parametrize("shape", FP64_SHAPES)
def test_gram_restart_preserves_fp64_identity(shape):
    """Restart only repairs rounding drift; in fp64 it must stay a no-op."""
    torch.manual_seed(1)
    update = torch.randn(*shape, dtype=torch.float64)

    standard = standard_zeropower(update, ns_steps=5)
    restarted = gram_zeropower(update, ns_steps=5, restart_after=2)

    assert _rel_err(restarted, standard) < 1e-9


@pytest.mark.parametrize(
    "shape", [(3, 16, 40), (2, 10, 10), (4, 12, 4)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gram_low_precision_tracks_fp64_reference(shape, dtype):
    """Gram in half precision deviates from the fp64 iteration no more than a
    small multiple of the standard iteration's own low-precision deviation."""
    torch.manual_seed(2)
    update = torch.randn(*shape, dtype=torch.float64)
    reference = standard_zeropower(update, ns_steps=5)

    standard_lp = standard_zeropower(update, ns_steps=5, dtype=dtype)
    gram_lp = gram_zeropower(update, ns_steps=5, dtype=dtype, restart_after=2)

    standard_err = _rel_err(standard_lp, reference)
    gram_err = _rel_err(gram_lp, reference)
    assert gram_err < max(3.0 * standard_err, 5e-3), (standard_err, gram_err)


def test_standard_mirror_is_bitwise_production():
    """The dtype-flexible mirror reproduces the shipped bf16 function exactly,
    so every equivalence result above transfers to production code."""
    torch.manual_seed(3)
    update = torch.randn(6, 32, 80, dtype=torch.float32)

    production = _batched_zeropower(
        update, ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7,
    )
    mirror = standard_zeropower(update, ns_steps=5, dtype=torch.bfloat16)

    torch.testing.assert_close(mirror, production, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape", [(24, 512, 1664), (32, 512, 512), (32, 1664, 512), (32, 8, 512)],
)
@pytest.mark.gpu
def test_gram_matches_production_on_gpu_shapes(shape):
    """End-to-end check on the real bucket shapes: Gram (computed in fp64 on
    GPU) vs the shipped bf16 production iteration — Gram must sit inside the
    bf16 rounding budget of the exact same iteration."""
    torch.manual_seed(4)
    update = torch.randn(*shape, dtype=torch.float32, device="cuda")

    production = _batched_zeropower(
        update, ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7,
    ).to(torch.float64)
    exact = standard_zeropower(update, ns_steps=5).to("cuda")

    production_err = _rel_err(production, exact)
    gram_err = _rel_err(gram_zeropower(update, ns_steps=5, dtype=torch.float64),
                        exact)
    assert gram_err < 1e-9
    assert production_err < 0.05, production_err


def test_batch_elements_are_independent():
    """Batched Gram must equal per-matrix Gram: experts never mix (the norm is
    per batch element and Q/gram are block-diagonal per element)."""
    torch.manual_seed(5)
    updates = torch.randn(4, 12, 30, dtype=torch.float64)

    joint = gram_zeropower(updates, ns_steps=5)
    solo = torch.stack([
        gram_zeropower(updates[i], ns_steps=5) for i in range(updates.shape[0])
    ])

    torch.testing.assert_close(joint, solo, rtol=0, atol=0)


def test_tall_input_matches_wide_input_transposed():
    """polar(Xᵀ) = polar(X)ᵀ and the odd iteration commutes with transpose;
    the internal tall→wide transpose handling must respect that."""
    torch.manual_seed(6)
    wide = torch.randn(2, 6, 20, dtype=torch.float64)

    from_wide = gram_zeropower(wide, ns_steps=5)
    from_tall = gram_zeropower(wide.transpose(-2, -1).contiguous(), ns_steps=5)

    torch.testing.assert_close(from_tall, from_wide.transpose(-2, -1),
                               rtol=1e-10, atol=1e-12)


def test_zero_update_stays_zero():
    """A zero gradient must stay zero without NaNs (norm clamp → eps path)."""
    zero = torch.zeros(3, 8, 16, dtype=torch.float64)

    assert torch.count_nonzero(gram_zeropower(zero, ns_steps=5)) == 0
    assert torch.count_nonzero(
        gram_zeropower(zero, ns_steps=5, dtype=torch.float16, restart_after=2)
    ) == 0


def test_fp64_agreement_against_svd_polar_distance():
    """Independent cross-check: standard and Gram sit equally far from the true
    polar factor U·Vᵀ — they approximate the same object, not just each other."""
    torch.manual_seed(7)
    update = torch.randn(2, 12, 30, dtype=torch.float64)

    standard = standard_zeropower(update, ns_steps=5)
    gram = gram_zeropower(update, ns_steps=5)

    def polar_distance(matrix):
        u, _, vh = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
        return _rel_err(matrix, u @ vh)

    assert abs(polar_distance(standard) - polar_distance(gram)) < 1e-9
