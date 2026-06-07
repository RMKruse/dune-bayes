"""Tests for the scipy-backed family CDFs — issue 0093 / GitHub #93.

``BaseFamily.cdf`` feeds the PIT calibration metric; torch's StudentT has no
``.cdf``, so families map their linked parameters to scipy distributions
(eval-time only, no gradient path).

Reference-test archetype (CLAUDE.md): MC-convergence — F(y) is checked against
the empirical P(X ≤ y) from torch's OWN sampler for the same linked
distribution. torch sampling and scipy CDF evaluation share no code, so
agreement cross-validates the parameter mapping (loc/scale/df/rate
conventions), which is exactly where a scipy-mapping bug would live.
"""

import torch

from dune_bayes.families import (
    BetaFamily,
    GammaFamily,
    NegativeBinomialFamily,
    StudentTFamily,
)

# MC tolerance: the empirical CDF at y is a binomial proportion with
# SE = sqrt(p(1-p)/N) ≤ 0.5/sqrt(N); at N = 200_000 that is ≤ 0.0011, so
# atol = 0.006 gives > 5 standard errors of headroom under the fixed seed.
_N_MC = 200_000
_ATOL_MC = 0.006


def _mc_cdf_reference(
    dist: torch.distributions.Distribution, y: torch.Tensor
) -> torch.Tensor:
    """Empirical P(X ≤ y) from torch's sampler — independent of scipy."""
    g = torch.manual_seed(1234)  # global seed: .sample() takes no generator
    draws = dist.sample((_N_MC,))  # (N, n)
    del g
    return (draws <= y).to(torch.float64).mean(dim=0)


def test_student_t_cdf_matches_torch_sampler():
    """StudentT cdf agrees with the empirical CDF of torch's own draws.

    Covers light and heavy tails (df 30 / 3) and off-center y so the df /
    loc / scale → scipy.stats.t(df, loc, scale) mapping is each exercised.
    """
    family = StudentTFamily(validate_args=True)
    # Raw params: loc identity; scale/df softplus-linked — raw 2.0 → ≈ 2.13,
    # raw 5.0 → ≈ 5.01 (exact link values are irrelevant: the reference
    # builds the SAME linked dist via family(params)).
    params = torch.tensor(
        [
            [0.0, 0.5, 5.0],  # ≈ t(df 6.0, loc 0, scale 0.97)
            [-2.0, 2.0, 30.0],  # ≈ t(df 31, loc −2, scale 2.13)
            [1.0, 0.0, 2.0],  # ≈ t(df 3.1, loc 1, scale 0.69)
        ]
    )
    y = torch.tensor([0.8, -3.5, 1.0])

    result = family.cdf(params, y)

    expected = _mc_cdf_reference(family(params), y)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, rtol=0.0, atol=_ATOL_MC)


def test_negative_binomial_cdf_matches_torch_sampler():
    """NegBin cdf agrees with the empirical CDF of torch's own draws.

    The mapping hotspot (issue #95): scipy's nbinom success probability is
    1 − torch's p (= sigmoid(−logits)) — a forgotten complement passes any
    p ≈ 0.5 fixture, so the linked (μ, σ) here put p well away from 0.5 on
    both sides (torch p = σμ/(1+σμ) ≈ 0.22, 0.56, 0.73). On integer support
    the empirical P(X ≤ y) sits flat between jumps, so the MC reference
    carries no boundary subtlety at integer y.
    """
    family = NegativeBinomialFamily(validate_args=True)
    # Raw params softplus-linked: mean / dispersion strictly positive.
    params = torch.tensor(
        [
            [2.0, -2.0],  # ≈ NBI(μ 2.13, σ 0.13) — near-Poisson, torch p ≈ 0.22
            [0.5, 1.0],  # ≈ NBI(μ 0.97, σ 1.31) — torch p ≈ 0.56
            [4.0, 0.0],  # ≈ NBI(μ 4.02, σ 0.69) — overdispersed, torch p ≈ 0.73
        ]
    )
    y = torch.tensor([2.0, 0.0, 3.0])

    result = family.cdf(params, y)

    expected = _mc_cdf_reference(family(params), y)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, rtol=0.0, atol=_ATOL_MC)


def test_beta_cdf_matches_torch_sampler():
    """Beta cdf agrees with the empirical CDF of torch's own draws.

    The mapping hotspot (issue #96): a swapped μ ↔ (1−μ) (i.e. scipy a ↔ b)
    passes any μ = 0.5 fixture, so the linked means here sit well away from
    1/2 on both sides (μ ≈ 0.12, 0.73) with a symmetric control in between.
    """
    family = BetaFamily(validate_args=True)
    # Raw params: mean sigmoid-linked, precision softplus-linked.
    params = torch.tensor(
        [
            [-2.0, 1.0],  # ≈ Beta(a 0.16, b 1.15) — μ ≈ 0.12, diffuse
            [1.0, 3.0],  # ≈ Beta(a 2.23, b 0.82) — μ ≈ 0.73
            [0.0, 5.0],  # ≈ Beta(a 2.51, b 2.51) — symmetric control
        ]
    )
    y = torch.tensor([0.1, 0.6, 0.5])

    result = family.cdf(params, y)

    expected = _mc_cdf_reference(family(params), y)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, rtol=0.0, atol=_ATOL_MC)


def test_gamma_cdf_matches_torch_sampler():
    """Gamma cdf agrees with the empirical CDF of torch's own draws.

    The mapping hotspot: torch parameterizes by RATE, scipy.stats.gamma by
    SCALE (= 1/rate) — a swapped convention passes any rate ≈ 1 fixture, so
    the rates here sit well away from 1 on both sides (≈ 0.13 and ≈ 4.0).
    """
    family = GammaFamily(validate_args=True)
    # Raw params softplus-linked: concentration / rate strictly positive.
    params = torch.tensor(
        [
            [2.0, -2.0],  # ≈ Gamma(conc 2.13, rate 0.13) — mean ≈ 17
            [0.5, 4.0],  # ≈ Gamma(conc 0.97, rate 4.02) — mean ≈ 0.24
            [5.0, 1.0],  # ≈ Gamma(conc 5.01, rate 1.31) — mean ≈ 3.8
        ]
    )
    y = torch.tensor([10.0, 0.1, 4.0])

    result = family.cdf(params, y)

    expected = _mc_cdf_reference(family(params), y)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, rtol=0.0, atol=_ATOL_MC)
