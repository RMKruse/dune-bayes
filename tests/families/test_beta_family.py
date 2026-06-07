"""Tests for BetaFamily — issue 0096 / GitHub #96.

The bounded-response family pins the beta-regression mean/precision
parameterization (Ferrari & Cribari-Neto 2004; the GAMLSS-adjacent
convention): network heads are (μ, φ) with mean μ ∈ (0, 1) and precision
φ > 0, mapped onto torch's Beta via concentration1 = μφ,
concentration0 = (1−μ)φ, so Var(y) = μ(1−μ)/(1+φ).

Reference-test archetype (CLAUDE.md): closed-form — log_prob against
scipy.stats.beta.logpdf, with the parameterization translation
(scipy a = μφ, b = (1−μ)φ) stated and tested explicitly (AC1).
"""

import math

import pytest
import torch
from scipy import stats

from dune_bayes.families import BetaFamily
from dune_bayes.metrics import pit


def _raw_phi(value: float) -> float:
    """Invert the softplus link: softplus(log(expm1(v))) == v."""
    return math.log(math.expm1(value))


def _raw_mu(value: float) -> float:
    """Invert the sigmoid link (up to the EPS floor): sigmoid(logit(v)) == v."""
    return math.log(value / (1.0 - value))


# ── 1: Tracer — torch log_prob == scipy beta.logpdf under the translation ────


def test_log_prob_matches_scipy_beta_at_pinned_params():
    """log_prob at (μ=0.3, φ=10) equals beta.logpdf(y, a=μφ, b=(1−μ)φ).

    The translation under test (AC1): mean/precision (μ, φ) → torch
    (concentration1 = μφ, concentration0 = (1−μ)φ) — scipy's (a, b) are the
    same concentrations, so a mixed-up head or a swapped (1−μ) shows up as
    whole units of log-density. Tolerance: the links shift (μ, φ) by O(EPS)
    with O(1)–O(10) logpdf sensitivity (digamma/log terms ⇒ error ≈ 1.5e-5),
    plus float32 lgamma round-off ~2e-6; atol = 1e-4 gives ~6× headroom
    while still pinning the value.
    """
    family = BetaFamily(validate_args=True)
    assert family.param_count == 2

    mu, phi = 0.3, 10.0
    y = torch.tensor([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    params = torch.tensor([[_raw_mu(mu), _raw_phi(phi)]]).expand(y.shape[0], 2)

    dist = family(params)
    assert isinstance(dist, torch.distributions.Beta)
    assert dist.batch_shape == (y.shape[0],)

    result = family.log_prob(params, y)

    expected = stats.beta.logpdf(y.numpy(), a=mu * phi, b=(1.0 - mu) * phi)
    torch.testing.assert_close(
        result.to(torch.float64),
        torch.from_numpy(expected),
        rtol=0.0,
        atol=1e-4,
    )


# ── 1b: the mean/precision claim in moment form ───────────────────────────────


def test_moments_pin_the_mean_precision_convention():
    """dist.mean == μ and dist.variance == μ(1−μ)/(1+φ) on the natural scale.

    torch computes both moments from (concentration1, concentration0) through
    its own code path, so this independently pins the translation: swapped
    heads or a missing (1−μ) move the mean to the wrong side of 1/2; a
    φ-vs-1/φ slip shifts the variance by orders of magnitude. Tolerance:
    rtol=1e-4 covers the EPS link floors (relative shift ≤ 1e-5 at μ = 0.05)
    plus float32 sigmoid/exp round-off; no MC noise enters.
    """
    family = BetaFamily(validate_args=True)
    mu = torch.tensor([0.05, 0.3, 0.5, 0.95])
    phi = torch.tensor([0.5, 2.0, 10.0, 200.0])
    # Invert the links in float64: expm1(200) overflows float32 (the linked
    # forward pass never computes expm1 — only the test-side inversion does).
    params = torch.stack(
        [
            torch.log(mu.double() / (1.0 - mu.double())),
            torch.log(torch.expm1(phi.double())),
        ],
        dim=-1,
    ).float()

    dist = family(params)

    torch.testing.assert_close(dist.mean, mu, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        dist.variance, mu * (1.0 - mu) / (1.0 + phi), rtol=1e-4, atol=1e-6
    )


# ── 2: log_prob across the (μ, φ) range incl. support boundaries (AC1) ───────


def test_log_prob_matches_scipy_near_support_boundaries():
    """log_prob == beta.logpdf over the (μ, φ) range, y down to 1e-6 / 1−1e-6.

    Sweeps the mean from near-0 to near-1 and the precision from diffuse
    (φ = 0.5, both concentrations < 1: density diverges at BOTH boundaries)
    to sharply concentrated (φ = 200); every cell is scored at y → 0⁺ and
    y → 1⁻ as well as the bulk — exactly where a bare log(y)/log(1−y)
    would blow up (AC1). Tolerance: the dominant error is the EPS floor on
    μ amplified by the precision (Δ(μφ) ≈ EPS·φ = 2e-4 at φ = 200) times
    the |log y − digamma| sensitivity ~10 (probed max over the grid:
    3.4e-3); atol = 1e-2 keeps ~3× headroom while a swapped head or a
    missing (1−μ) is off by whole units of log-density.
    """
    family = BetaFamily(validate_args=True)
    y = torch.tensor([1e-6, 1e-4, 0.05, 0.5, 0.95, 1.0 - 1e-4, 1.0 - 1e-6])

    for mu in [0.02, 0.3, 0.5, 0.7, 0.98]:
        for phi in [0.5, 2.0, 10.0, 200.0]:
            params = torch.tensor(
                [[_raw_mu(mu), _raw_phi(phi)]], dtype=torch.float64
            ).float()
            params = params.expand(y.shape[0], 2)

            result = family.log_prob(params, y)

            # scipy reference evaluated at the float32-rounded y values.
            expected = stats.beta.logpdf(
                y.to(torch.float64).numpy(), a=mu * phi, b=(1.0 - mu) * phi
            )
            torch.testing.assert_close(
                result.to(torch.float64),
                torch.from_numpy(expected),
                rtol=0.0,
                atol=1e-2,
                msg=f"log_prob mismatch at mu={mu}, phi={phi}",
            )


# ── 3: PIT on a well-specified Beta simulation ────────────────────────────────


def test_pit_uniform_under_well_specified_beta():
    """PIT on data drawn from the model's own Beta is uniform.

    The bounded-response calibration integration (issue #96): heterogeneous
    true (μ_i, φ_i); each y_i is drawn from Beta(μ_iφ_i, (1−μ_i)φ_i) via
    torch's own sampler, and the predictive is T identical draws AT the
    truth (perfectly specified, zero-epistemic) — continuous support, so
    plain (non-randomized) PIT is exactly uniform. Same deterministic KS
    regression gate as tests/metrics/test_pit.py: p > 0.01 at the fixed
    seed (probe: KS ≈ 0.018, p ≈ 0.53 — 50× above the threshold).
    """
    g = torch.Generator().manual_seed(96)
    n, t = 2_000, 4
    mu = 0.2 + 0.6 * torch.rand(n, generator=g)  # U(0.2, 0.8)
    phi = 2.0 + 18.0 * torch.rand(n, generator=g)  # U(2, 20)
    params = torch.stack(
        [torch.log(mu / (1.0 - mu)), torch.log(torch.expm1(phi))], dim=-1
    )
    family = BetaFamily(validate_args=True)
    torch.manual_seed(960)  # Beta.sample takes no generator
    y = family(params).sample()
    samples = params.unsqueeze(0).expand(t, n, 2)  # T = 4 draws at truth

    result = pit(family, samples, y)

    ks = stats.kstest(result.numpy(), "uniform")
    assert ks.pvalue > 0.01, f"PIT not uniform: KS={ks.statistic:.4f}"


# ── 4: parameter recovery on simulated (0, 1) data (AC3) ─────────────────────


def test_mle_recovers_mean_and_precision():
    """MLE through family.log_prob recovers (μ, φ) from simulated proportions.

    Simulate n = 4000 Beta(μ=0.7, φ=10) draws, fit the two raw pre-link
    values by Adam on the mean negative log-likelihood — the exact gradient
    path BayesianNAMLSS trains through (lgamma + the sigmoid/softplus links;
    no rsample needed: the proportions are data, not draws). 2000 steps:
    at the fixed seed Adam then matches scipy's exact MLE
    (``beta.fit(floc=0, fscale=1)`` → μ̂ = 0.6995, φ̂ = 10.149) to 3
    decimals, so the residual is pure sampling error: SE(μ̂) = sd/√n ≈
    0.002, SE(φ̂) ≈ φ√(2/n) ≈ 0.25 — abs 0.02 / 1.0 leave ≈ 4–10×
    headroom (recovery test, not a precision benchmark; NegBin #95 pattern).
    """
    truth = {"mu": 0.7, "phi": 10.0}
    family = BetaFamily()
    true_params = torch.tensor([[_raw_mu(truth["mu"]), _raw_phi(truth["phi"])]])
    torch.manual_seed(961)  # Beta.sample takes no generator
    y = family(true_params).sample((4000,)).squeeze(-1)

    raw = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(2000):
        optimizer.zero_grad()
        nll = -family.log_prob(raw.expand(y.shape[0], 2), y).mean()
        nll.backward()
        optimizer.step()

    fitted = family(raw.detach().unsqueeze(0))
    recovered_mu = fitted.mean.item()
    # Invert Var = μ(1−μ)/(1+φ): φ = μ(1−μ)/Var − 1 recovers the precision.
    recovered_phi = recovered_mu * (1.0 - recovered_mu) / fitted.variance.item() - 1.0

    assert recovered_mu == pytest.approx(truth["mu"], abs=0.02), (
        f"mu: recovered {recovered_mu:.4f}, true {truth['mu']}"
    )
    assert recovered_phi == pytest.approx(truth["phi"], abs=1.0), (
        f"phi: recovered {recovered_phi:.3f}, true {truth['phi']}"
    )
