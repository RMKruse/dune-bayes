"""Tests for the probability integral transform — issue 0093 / GitHub #93.

PIT_i = (1/T) Σ_t F(y_i | θ_t): the predictive (mixture) CDF at the observed
response, averaged over posterior draws (PRD 0002 §23). Under a well-specified
model the PIT values are uniform on [0, 1]; calibration claims are read off
that uniformity. CDFs come from scipy (eval-time only, no gradient path) —
torch's StudentT has no ``.cdf``.

Reference-test archetypes (CLAUDE.md):
  - Closed-form: with T = 1 Normal draws the predictive IS one Gaussian, so
                 PIT must equal Φ((y − μ)/σ) exactly (torch's own Normal.cdf
                 is the independent reference — a different code path from
                 the scipy-backed implementation).
"""

import math

import torch
import torch.nn.functional as F
from scipy import stats

from dune_bayes.families import NormalFamily
from dune_bayes.families.base import BaseFamily
from dune_bayes.metrics import pit
from dune_bayes.utils import EPS


class _PoissonFamily(BaseFamily):
    """Minimal discrete fixture family — what the BaseFamily.cdf seam is for.

    Test-local on purpose: no discrete family ships in v1, but randomized
    PIT must be exercised against genuine integer support (issue #93).
    """

    param_count: int = 1

    def __call__(self, params: torch.Tensor) -> torch.distributions.Poisson:
        rate = F.softplus(params[..., 0]) + EPS  # numerical rule 1
        return torch.distributions.Poisson(rate=rate, validate_args=True)

    def cdf(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dist = self(params)
            value = stats.poisson.cdf(
                y.detach().cpu().numpy(), mu=dist.rate.cpu().numpy()
            )
        return torch.from_numpy(value).to(torch.float64)


# Invert the softplus scale link: raw = log(expm1(scale)) gives
# softplus(raw) == scale, so fixtures can state σ on the natural scale.


def _raw_scale(scale: float) -> float:
    return math.log(math.expm1(scale))


# ── 1: Tracer — T = 1 Normal predictive has the Gaussian CDF as its PIT ───────


def test_pit_single_normal_draw_equals_gaussian_cdf():
    """With one posterior draw the mixture collapses to N(μ, σ): PIT = Φ(z).

    No MC noise enters — both sides evaluate one Gaussian CDF at the same
    points. Tolerance: the family link is softplus(raw) + EPS, so the
    effective scale is σ + 1e-6 evaluated through float32 params; the induced
    CDF shift is |∂Φ/∂σ|·EPS = φ(z)|z|/σ · 1e-6 ≲ 5e-7 here (σ ≥ 0.5).
    atol = 1e-5 leaves > 10× headroom while still pinning the value.
    """
    family = NormalFamily(validate_args=True)
    mu = torch.tensor([-1.0, 0.0, 2.0])
    sigma = torch.tensor([0.5, 1.0, 2.0])
    y = torch.tensor([-1.5, 0.3, 4.0])

    # (T=1, n=3, param_count=2) summed-predictor draws, links inverted.
    raw_sigma = torch.tensor([_raw_scale(s) for s in sigma.tolist()])
    samples = torch.stack([mu, raw_sigma], dim=-1).unsqueeze(0)

    result = pit(family, samples, y)

    expected = torch.distributions.Normal(
        mu.to(torch.float64), sigma.to(torch.float64), validate_args=True
    ).cdf(y.to(torch.float64))
    assert result.shape == (3,)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-5)


# ── 2: Well-specified Normal simulation → uniform PIT (criterion 1) ───────────


def test_pit_uniform_under_well_specified_normal():
    """PIT on data drawn from the model's own predictive is uniform (KS check).

    Construction: heterogeneous true (μ_i, σ_i); each y_i is drawn from
    N(μ_i, σ_i), and the predictive is T identical draws AT the truth (a
    perfectly-specified, zero-epistemic model). Then PIT_i = Φ((y_i − μ_i)/σ_i)
    is exactly U(0, 1) in distribution — any KS failure indicts the metric,
    not the model. Threshold: under H0 the KS p-value is U(0, 1); requiring
    p > 0.01 would reject a CORRECT implementation 1% of the time on a random
    seed, but the seed is fixed, so this is a deterministic regression gate
    (probe: KS ≈ 0.024, p ≈ 0.18 at seed 0 — 18× above the threshold).
    """
    g = torch.Generator().manual_seed(0)
    n, t = 2_000, 5
    mu = 2.0 * torch.rand(n, generator=g) - 1.0  # U(−1, 1)
    sigma = 0.5 + torch.rand(n, generator=g)  # U(0.5, 1.5)
    y = mu + sigma * torch.randn(n, generator=g)

    raw_sigma = torch.log(torch.expm1(sigma))  # invert the softplus link
    samples = torch.stack([mu, raw_sigma], dim=-1).expand(t, n, 2)

    result = pit(NormalFamily(validate_args=True), samples, y)

    ks = stats.kstest(result.numpy(), "uniform")
    assert ks.pvalue > 0.01, f"PIT not uniform: KS={ks.statistic:.4f}"


# ── 3: Discrete support — randomized PIT uniform, plain PIT not (criterion 2) ─


def _poisson_perfect_model(
    n: int, seed: int
) -> tuple[_PoissonFamily, torch.Tensor, torch.Tensor]:
    """Well-specified Poisson fixture: y drawn from the model's own rates."""
    g = torch.Generator().manual_seed(seed)
    rate = 1.0 + 4.0 * torch.rand(n, generator=g)  # U(1, 5): small counts
    y = torch.poisson(rate.unsqueeze(0), generator=g).squeeze(0)
    raw_rate = torch.log(torch.expm1(rate))  # invert the softplus link
    samples = raw_rate.reshape(1, n, 1).expand(4, n, 1)  # T = 4 draws at truth
    return _PoissonFamily(), samples, y


def test_randomized_pit_uniform_on_discrete_support():
    """u·F(y) + (1−u)·F(y−1) with u ~ U(0,1) is uniform under a perfect model.

    The randomization spreads each F-jump's probability mass evenly across
    [F(y−1), F(y)], which restores exact uniformity in distribution — same
    deterministic KS gate as the Normal test (p > 0.01 at the fixed seed).
    """
    family, samples, y = _poisson_perfect_model(n=2_000, seed=11)

    result = pit(family, samples, y, randomized=True, seed=7)

    ks = stats.kstest(result.numpy(), "uniform")
    assert ks.pvalue > 0.01, f"randomized PIT not uniform: KS={ks.statistic:.4f}"


def test_plain_pit_not_uniform_on_discrete_support():
    """Plain PIT on the SAME discrete fixture demonstrably fails uniformity.

    F(y) on integer support only takes the jump-top values, so the PIT
    histogram is lumpy even under a perfect model — the reason randomized
    PIT exists. With n = 2_000 the KS test rejects overwhelmingly; requiring
    p < 1e-6 shows the failure is structural, not borderline.
    """
    family, samples, y = _poisson_perfect_model(n=2_000, seed=11)

    result = pit(family, samples, y)

    ks = stats.kstest(result.numpy(), "uniform")
    assert ks.pvalue < 1e-6, f"plain PIT looked uniform: KS={ks.statistic:.4f}"


def test_randomized_pit_is_seeded():
    """Same seed → identical values; different seed → different values.

    The u-draw must come from an owned generator, not the global stream —
    exact equality (not approx) is the contract, and the global RNG state
    must be untouched so callers' downstream draws don't shift.
    """
    family, samples, y = _poisson_perfect_model(n=200, seed=3)

    torch.manual_seed(999)
    state_before = torch.get_rng_state()
    first = pit(family, samples, y, randomized=True, seed=42)
    second = pit(family, samples, y, randomized=True, seed=42)
    other = pit(family, samples, y, randomized=True, seed=43)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.equal(torch.get_rng_state(), state_before)
