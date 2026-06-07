"""Tests for the fair sample-based CRPS — issue 0092 / GitHub #92.

One generic proper scoring rule for the T-draw predictive (PRD 0002 §22):
the fair (unbiased) sample estimator

    CRPS = (1/M) Σᵢ |xᵢ − y|  −  1/(2M(M−1)) Σᵢⱼ |xᵢ − xⱼ|,

computed sort-based (O(M log M)), accumulated in float64, family-agnostic
(samples in, scores out).

Reference-test archetypes (CLAUDE.md):
  - Closed-form: sort-based result equals the naive O(M²) pairwise double-sum
                 (the defining formula, coded independently below) to float64
                 round-off; estimator matches the analytic Gaussian CRPS
                 closed form on Normal fixtures to MC tolerance.
  - Shape:       (M, n) samples + (n,) y → (n,) per-observation scores.
"""

import math

import pytest
import torch

from dune_bayes.metrics import crps

# ── independent reference: the defining O(M²) formula, coded naively ──────────


def _crps_naive_pairwise(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fair CRPS straight from the definition — O(M²) pairwise, float64.

    The implementation under test must never materialize the pairwise matrix;
    this reference deliberately does, so the two share no code path.
    """
    x = samples.to(torch.float64)
    y64 = y.to(torch.float64)
    m = x.shape[0]
    term1 = (x - y64).abs().mean(dim=0)  # (n,)
    pairwise = (x.unsqueeze(0) - x.unsqueeze(1)).abs()  # (M, M, n)
    term2 = pairwise.sum(dim=(0, 1)) / (2 * m * (m - 1))  # (n,)
    return term1 - term2


def _crps_gaussian_analytic(
    mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Analytic CRPS of N(μ, σ) at y (Gneiting & Raftery 2007, eq. 21).

    CRPS = σ [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],  z = (y − μ)/σ.

    Φ and φ come from torch's own Normal — an independent code path from the
    sample estimator under test.
    """
    std_normal = torch.distributions.Normal(0.0, 1.0, validate_args=True)
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * std_normal.cdf(z) - 1.0)
        + 2.0 * std_normal.log_prob(z).exp()
        - 1.0 / math.sqrt(math.pi)
    )


# ── 1: Tracer — sort-based identity equals the defining formula ───────────────


def test_crps_matches_naive_pairwise_reference():
    """Sort-based CRPS equals the naive O(M²) fair estimator exactly.

    Both sides evaluate the SAME deterministic formula on the SAME samples in
    float64 — no MC noise enters, so the tolerance is float64 summation
    round-off only.
    """
    g = torch.Generator().manual_seed(0)
    m, n = 64, 7
    samples = torch.randn(m, n, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)

    result = crps(samples, y)

    assert result.shape == (n,)
    torch.testing.assert_close(
        result, _crps_naive_pairwise(samples, y), rtol=1e-12, atol=1e-12
    )


# ── 2: Analytic Gaussian reference (acceptance criterion 1) ───────────────────


def test_crps_matches_analytic_gaussian():
    """Fair estimator on N(μ, σ) draws matches the closed form to MC tolerance.

    The fair estimator is UNBIASED for the population CRPS, so the only error
    is MC noise: per-observation standard error ≈ c·σ/√M with c = O(1)
    (term1 is a mean of |x − y| whose std is ≲ σ). At M = 100_000 and σ ≤ 2
    that is ≲ 2/√1e5 ≈ 0.006; atol = 0.03 gives ≈ 5 standard errors of
    headroom under the fixed seed.
    """
    g = torch.Generator().manual_seed(42)
    m = 100_000
    mu = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    sigma = torch.tensor([0.5, 1.0, 2.0, 0.1])
    y = torch.tensor([-1.5, 0.0, 3.0, 2.05])  # in-, at-, and off-center cases

    samples = mu + sigma * torch.randn(m, 4, generator=g, dtype=torch.float64)
    result = crps(samples, y)

    expected = _crps_gaussian_analytic(
        mu.to(torch.float64), sigma.to(torch.float64), y.to(torch.float64)
    )
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.03)


# ── 3: MC-convergence — estimate stabilizes as M grows (criterion 2) ──────────


def test_crps_converges_as_m_grows():
    """The estimate approaches the analytic value inside a shrinking 1/√M band.

    The fair estimator is unbiased with MC standard error ∝ 1/√M, so the
    tolerance must TIGHTEN with M — a fixed tolerance would not demonstrate
    stabilization. Envelope 3/√M: per-observation SE for standard-normal draws
    is ≈ 0.55/√M, and the worst deviation across 8 probe seeds was ≈ 1.4/√M,
    so 3/√M leaves > 2× headroom while still shrinking 30× over the sweep
    (0.3 at M = 100 → 0.0095 at M = 100_000).
    """
    std_normal = torch.distributions.Normal(0.0, 1.0, validate_args=True)
    y = torch.tensor([0.7], dtype=torch.float64)
    expected = float(
        (2.0 * std_normal.cdf(y) - 1.0) * y
        + 2.0 * std_normal.log_prob(y).exp()
        - 1.0 / math.sqrt(math.pi)
    )

    for m in [100, 1_000, 10_000, 100_000]:
        g = torch.Generator().manual_seed(0)
        samples = torch.randn(m, 1, generator=g, dtype=torch.float64)
        estimate = float(crps(samples, y))
        tol = 3.0 / math.sqrt(m)
        assert estimate == pytest.approx(expected, abs=tol), (
            f"M={m}: |{estimate:.5f} − {expected:.5f}| > {tol:.5f}"
        )


# ── 4: Discrete predictive — integer samples, heavy ties (criterion 3) ────────


def test_crps_works_on_discrete_integer_samples():
    """Integer (count) samples work unchanged — the sort identity survives ties.

    A discrete predictive with small support produces MANY tied draws; the
    sort-based pairwise identity must still equal the defining double sum
    (ties contribute |xᵢ − xⱼ| = 0 either way). Poisson(3) draws as int64
    exercise both the tie structure and the integer input dtype. Same
    no-MC-noise reference as the tracer → float64 round-off tolerance.
    """
    g = torch.Generator().manual_seed(7)
    m, n = 256, 5
    rate = torch.full((m, n), 3.0)
    samples = torch.poisson(rate, generator=g).to(torch.int64)  # counts
    y = torch.tensor([0, 2, 3, 5, 9], dtype=torch.int64)

    result = crps(samples, y)

    assert result.shape == (n,)
    torch.testing.assert_close(
        result, _crps_naive_pairwise(samples, y), rtol=1e-12, atol=1e-12
    )


# ── 5: float64 accumulation + mean reducer (criterion 4 + issue scope) ────────


def test_crps_accumulates_in_float64_from_float32_input():
    """float32 samples (the forward-path dtype) → float64 scores (dtype rule).

    Predictive draws arrive float32 (CLAUDE.md: forward pass stays float32);
    the metric must promote BEFORE accumulating, so the result is float64.
    """
    g = torch.Generator().manual_seed(3)
    samples = torch.randn(128, 4, generator=g)  # float32, like real draws
    y = torch.randn(4, generator=g)  # float32

    result = crps(samples, y)

    assert result.dtype == torch.float64


def test_crps_mean_reducer():
    """reduce="mean" returns the scalar mean of the per-observation scores."""
    g = torch.Generator().manual_seed(4)
    samples = torch.randn(128, 6, generator=g, dtype=torch.float64)
    y = torch.randn(6, generator=g, dtype=torch.float64)

    reduced = crps(samples, y, reduce="mean")

    assert reduced.shape == ()
    assert reduced.dtype == torch.float64
    torch.testing.assert_close(reduced, crps(samples, y).mean())


def test_crps_rejects_unknown_reduce():
    """An unknown reduce value raises ValueError, not silent no-reduction."""
    samples = torch.randn(8, 2, dtype=torch.float64)
    y = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="reduce"):
        crps(samples, y, reduce="sum")
