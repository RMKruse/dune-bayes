"""Tests for NegativeBinomialFamily — issue 0095 / GitHub #95.

The count family pins the GAMLSS NBI parameterization: network heads are
(μ, σ) = (mean, dispersion) with Var(y) = μ + σμ², mapped onto torch's
NegativeBinomial via total_count = 1/σ, logits = log(μ) + log(σ).

Reference-test archetype (CLAUDE.md): closed-form — log_prob against
scipy.stats.nbinom.logpmf, with the parameterization translation
(scipy's success probability p = 1 − torch's p = 1/(1 + σμ)) stated and
tested explicitly (AC1).
"""

import math

import pytest
import torch
from scipy import stats

from dune_bayes.families import NegativeBinomialFamily
from dune_bayes.metrics import pit


def _raw(value: float) -> float:
    """Invert the softplus link: softplus(log(expm1(v))) == v."""
    return math.log(math.expm1(value))


# ── 1: Tracer — torch log_prob == scipy nbinom.logpmf under the translation ──


def test_log_prob_matches_scipy_nbinom_at_pinned_params():
    """log_prob at (μ=3, σ=0.5) equals nbinom.logpmf(k, n=1/σ, p=1/(1+σμ)).

    The translation under test (AC1): GAMLSS NBI (μ, σ) → torch
    (total_count=1/σ, logits=log(μσ)) → scipy (n=1/σ, p=1/(1+σμ)) — scipy's
    p is the per-trial probability of the OTHER outcome, 1 − torch's p.
    Tolerance: the links are softplus+EPS, shifting (μ, σ) by 1e-6 with
    O(1) logpmf sensitivity; atol=1e-4 covers that plus float32 lgamma
    round-off with ample headroom while still pinning the value.
    """
    family = NegativeBinomialFamily(validate_args=True)
    assert family.param_count == 2

    mu, sigma = 3.0, 0.5
    y = torch.arange(0.0, 12.0)  # counts 0..11 span the bulk and right tail
    params = torch.tensor([[_raw(mu), _raw(sigma)]]).expand(y.shape[0], 2)

    dist = family(params)
    assert isinstance(dist, torch.distributions.NegativeBinomial)
    assert dist.batch_shape == (y.shape[0],)

    result = family.log_prob(params, y)

    expected = stats.nbinom.logpmf(y.numpy(), n=1.0 / sigma, p=1.0 / (1.0 + sigma * mu))
    torch.testing.assert_close(
        result.to(torch.float64),
        torch.from_numpy(expected),
        rtol=0.0,
        atol=1e-4,
    )


# ── 1b: the GAMLSS NBI claim in moment form ───────────────────────────────────


def test_moments_pin_the_gamlss_nbi_convention():
    """dist.mean == μ and dist.variance == μ + σμ² on the natural scale.

    torch computes both moments from (total_count, logits) through its own
    code path, so this independently pins the translation: a swapped head or
    a 1/σ-vs-σ slip shifts the variance by orders of magnitude. Tolerance:
    rtol=1e-4 covers the EPS link floor (relative shift ≤ 1e-5 at μ = 0.1)
    plus float32 exp/sigmoid round-off; no MC noise enters.
    """
    family = NegativeBinomialFamily(validate_args=True)
    mu = torch.tensor([0.1, 1.0, 5.0, 50.0])
    sigma = torch.tensor([2.0, 0.5, 0.02, 10.0])
    params = torch.stack(
        [torch.log(torch.expm1(mu)), torch.log(torch.expm1(sigma))], dim=-1
    )

    dist = family(params)

    torch.testing.assert_close(dist.mean, mu, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        dist.variance, mu + sigma * mu * mu, rtol=1e-4, atol=1e-5
    )


# ── 2: log_prob across the (μ, σ) parameter range (AC1) ──────────────────────


def test_log_prob_matches_scipy_across_parameter_range():
    """log_prob == nbinom.logpmf over near-Poisson to heavily-overdispersed.

    Sweeps μ across three orders of magnitude and σ from 0.02 (total_count
    50, near-Poisson) to 10 (total_count 0.1, extreme overdispersion); each
    cell is scored at counts spanning zero, the bulk, and the +3·sd tail.
    Tolerance: torch evaluates in float32 — the dominant error is the lgamma
    difference at the largest arguments (lgamma(μ + 1/σ) ≈ 3e3 at μ = 500,
    relative float32 error ~1e-7 → ~5e-4 absolute), so atol = 5e-3 gives
    10× headroom while still catching any translation slip (a swapped p
    convention is off by whole units of log-probability).
    """
    family = NegativeBinomialFamily(validate_args=True)

    for mu in [0.1, 1.0, 5.0, 50.0, 500.0]:
        for sigma in [0.02, 0.5, 2.0, 10.0]:
            sd = math.sqrt(mu + sigma * mu * mu)
            counts = sorted({0.0, 1.0, round(mu), round(mu + 3.0 * sd)})
            y = torch.tensor(counts)
            params = torch.tensor([[_raw(mu), _raw(sigma)]]).expand(y.shape[0], 2)

            result = family.log_prob(params, y)

            expected = stats.nbinom.logpmf(
                y.numpy(), n=1.0 / sigma, p=1.0 / (1.0 + sigma * mu)
            )
            torch.testing.assert_close(
                result.to(torch.float64),
                torch.from_numpy(expected),
                rtol=0.0,
                atol=5e-3,
                msg=f"log_prob mismatch at mu={mu}, sigma={sigma}",
            )


# ── 3: randomized PIT on a well-specified NegBin simulation (AC4) ─────────────


def test_randomized_pit_uniform_under_well_specified_negbin():
    """Randomized PIT on data drawn from the model's own NegBin is uniform.

    The count-response calibration integration (issue #95): heterogeneous
    true (μ_i, σ_i); each y_i is drawn from NBI(μ_i, σ_i) via torch's own
    sampler, and the predictive is T identical draws AT the truth (perfectly
    specified, zero-epistemic). u·F(y) + (1−u)·F(y−1) then restores exact
    uniformity on the integer support (issue #93). Same deterministic KS
    regression gate as tests/metrics/test_pit.py: p > 0.01 at the fixed
    seed (probe: KS ≈ 0.023, p ≈ 0.25 — 25× above the threshold; plain PIT
    on the same fixture gives p ≈ 8e-33, so the gate has real teeth).
    """
    g = torch.Generator().manual_seed(95)
    n, t = 2_000, 4
    mu = 1.0 + 4.0 * torch.rand(n, generator=g)  # U(1, 5): small counts
    sigma = 0.2 + 0.8 * torch.rand(n, generator=g)  # U(0.2, 1): overdispersed
    params = torch.stack(
        [torch.log(torch.expm1(mu)), torch.log(torch.expm1(sigma))], dim=-1
    )
    family = NegativeBinomialFamily(validate_args=True)
    torch.manual_seed(950)  # NegativeBinomial.sample takes no generator
    y = family(params).sample()
    samples = params.unsqueeze(0).expand(t, n, 2)  # T = 4 draws at truth

    result = pit(family, samples, y, randomized=True, seed=9)

    ks = stats.kstest(result.numpy(), "uniform")
    assert ks.pvalue > 0.01, f"randomized PIT not uniform: KS={ks.statistic:.4f}"


# ── 4: parameter recovery on simulated count data (AC5) ───────────────────────


def test_mle_recovers_mean_and_dispersion():
    """MLE through family.log_prob recovers (μ, σ) from simulated counts.

    Simulate n = 4000 NBI(μ=4, σ=0.6) draws, fit the two raw pre-link values
    by Adam on the mean negative log-likelihood — the exact gradient path
    BayesianNAMLSS trains through (lgamma + logits, no rsample needed: the
    counts are data, not draws). Tolerances are MLE sampling error at
    n = 4000 plus residual optimizer error: SE(μ̂) = sd/√n ≈ 0.06, and the
    dispersion's SE is a few times wider — abs 0.2 / 0.15 leave > 2×
    headroom (recovery test, not a precision benchmark; JohnsonSU pattern).
    """
    truth = {"mu": 4.0, "sigma": 0.6}
    family = NegativeBinomialFamily()
    true_params = torch.tensor([[_raw(truth["mu"]), _raw(truth["sigma"])]])
    torch.manual_seed(951)  # NegativeBinomial.sample takes no generator
    y = family(true_params).sample((4000,)).squeeze(-1)

    raw = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(600):
        optimizer.zero_grad()
        nll = -family.log_prob(raw.expand(y.shape[0], 2), y).mean()
        nll.backward()
        optimizer.step()

    fitted = family(raw.detach().unsqueeze(0))
    recovered_mu = fitted.mean.item()
    # Invert Var = μ + σμ²: σ = (Var − μ)/μ² recovers the dispersion head.
    recovered_sigma = (fitted.variance.item() - recovered_mu) / recovered_mu**2

    assert recovered_mu == pytest.approx(truth["mu"], abs=0.2), (
        f"mu: recovered {recovered_mu:.3f}, true {truth['mu']}"
    )
    assert recovered_sigma == pytest.approx(truth["sigma"], abs=0.15), (
        f"sigma: recovered {recovered_sigma:.3f}, true {truth['sigma']}"
    )
