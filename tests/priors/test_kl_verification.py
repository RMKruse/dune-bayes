"""Prior-tier KL verification — analytic claims vs independent references (#86).

PRD 0002 (#84) hardening slice: one verification test per prior tier
(ADR-0002).  No behavior change expected — any RED here is a real bug in the
KL math, not a test problem.

  1. Fixed tier:        analytic Gaussian–Gaussian weight-KL vs many-sample MC.
  2. Empirical-Bayes:   same with the learned scale + gradient reaches rho.
  3. Hierarchical IG:   closed-form scale-KL vs MC (KL bijection-invariance).
  4. Hierarchical HC:   single-sample MC estimator unbiased vs scipy quadrature.

All references are independent of package code: torch.distributions log_prob
for the MC estimates (never our hand-coded densities), scipy for quadrature.
All four run in the core (unskippable) suite — numerical correctness tests
are never skippable (CLAUDE.md).
"""

import math

import pytest
import torch
import torch.nn.functional as F

# ── independent MC reference ──────────────────────────────────────────────────


def _mc_gaussian_kl(
    loc: torch.Tensor,
    scale: torch.Tensor,
    prior_scale: float,
    n: int = 200_000,
    seed: int = 0,
) -> float:
    """MC estimate of KL[N(loc, scale²) ‖ N(0, prior_scale²)] summed over params.

    Uses torch.distributions.Normal.log_prob for both densities — independent
    of the package's hand-coded gaussian_kl formula (numerical rule 7).
    n=200_000 → std_err ≈ 0.005 for layer-sized parameter counts.
    """
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn((n, *loc.shape), generator=g)
    w = loc + scale * eps  # reparameterized draws from q
    q = torch.distributions.Normal(loc, scale, validate_args=True)
    p = torch.distributions.Normal(0.0, prior_scale, validate_args=True)
    per_draw = (q.log_prob(w) - p.log_prob(w)).sum(dim=tuple(range(1, w.dim())))
    return float(per_draw.mean())


# ── 1. fixed tier ─────────────────────────────────────────────────────────────


def test_fixed_tier_weight_kl_matches_mc():
    """Fixed tier: analytic weight-KL stashed by forward() agrees with MC.

    A VariationalDense carrying a fixed PriorScale handle stashes the
    closed-form Gaussian–Gaussian KL on forward(); a 200k-draw MC estimate of
    E_q[log q(W) − log p(W)] over the same posterior must agree.

    Tolerance: MC noise only (the analytic side is exact float32 arithmetic).
    Per-draw std over the 8-parameter layer is O(1) → std_err ≈ 0.005 at
    n=200k; abs=0.05 gives ~10× headroom.
    """
    from dune_bayes.layers import VariationalDense
    from dune_bayes.priors import PriorScale

    torch.manual_seed(0)
    handle = PriorScale(mode="fixed", scale=2.0)
    # Small layer (3×2 kernel + 2 bias = 8 params) keeps MC variance tight.
    layer = VariationalDense(
        in_features=3, units=2, prior_scale_handle=handle, validate_args=True
    )
    layer(torch.randn(4, 3))  # forward stashes the analytic KL
    analytic = float(layer.kl.detach())

    prior = 2.0
    kernel_scale = F.softplus(layer.kernel_rho).detach()
    bias_scale = F.softplus(layer.bias_rho).detach()
    mc = _mc_gaussian_kl(layer.kernel_loc.detach(), kernel_scale, prior, seed=1)
    mc += _mc_gaussian_kl(layer.bias_loc.detach(), bias_scale, prior, seed=2)

    assert analytic == pytest.approx(mc, abs=0.05), (
        f"analytic={analytic:.4f}, mc={mc:.4f}"
    )


# ── 2. empirical-Bayes tier ───────────────────────────────────────────────────


def test_eb_tier_weight_kl_matches_mc_and_grad_reaches_scale():
    """EB tier: analytic weight-KL agrees with MC at the *learned* prior scale,
    and the KL gradient reaches the scale parameter (rho).

    The MC reference evaluates p at the handle's current softplus(rho) value,
    so the agreement verifies the analytic KL tracks the learned scale.  The
    backward pass then confirms the ELBO can actually move that scale — the
    REML-analog claim of the tier (ADR-0002).

    Tolerance: same MC-noise budget as the fixed-tier test (n=200k,
    std_err ≈ 0.005, abs=0.05 → ~10× headroom).
    """
    from dune_bayes.layers import VariationalDense
    from dune_bayes.priors import PriorScale

    torch.manual_seed(3)
    handle = PriorScale(mode="empirical_bayes", scale=0.7)
    layer = VariationalDense(
        in_features=3, units=2, prior_scale_handle=handle, validate_args=True
    )
    layer(torch.randn(4, 3))  # forward stashes KL at the live learned scale
    analytic = float(layer.kl.detach())

    prior = float(handle().detach())  # the learned scale the KL must track
    kernel_scale = F.softplus(layer.kernel_rho).detach()
    bias_scale = F.softplus(layer.bias_rho).detach()
    mc = _mc_gaussian_kl(layer.kernel_loc.detach(), kernel_scale, prior, seed=4)
    mc += _mc_gaussian_kl(layer.bias_loc.detach(), bias_scale, prior, seed=5)

    assert analytic == pytest.approx(mc, abs=0.05), (
        f"analytic={analytic:.4f}, mc={mc:.4f}"
    )

    # Gradient path: KL must be able to drive the learned scale (rho).
    layer.kl.backward()
    assert handle.rho.grad is not None
    assert float(handle.rho.grad) != 0.0


# ── 3. hierarchical inverse-gamma tier ────────────────────────────────────────


def test_hier_ig_closed_form_kl_matches_mc():
    """Hierarchical IG: the closed-form scale-KL agrees with an MC estimate.

    Independent reference via KL's invariance under bijections: with Y = s²,
    KL[LogNormal(μ, σ²) ‖ sqrt-InvGamma(α₀, β₀)] in s-space equals
    KL[LogNormal(2μ, (2σ)²) ‖ InvGamma(α₀, β₀)] in Y-space.  Both Y-space
    densities come from torch.distributions, so the reference shares zero
    code with the package's moment-based derivation (unlike the existing
    closed-form test in test_prior_scale.py, which re-derives the same
    moments).

    Tolerance: MC noise only — per-draw std of log q(Y) − log p(Y) is O(1)
    at these parameters → std_err ≈ 0.003 at n=200k; abs=0.05 → >10× headroom.
    """
    from dune_bayes.priors import PriorScale

    mu, sigma = 0.3, 0.4
    alpha0, beta0 = 2.0, 1.5
    ps = PriorScale(
        mode="hierarchical", hyperprior="inverse_gamma", alpha0=alpha0, beta0=beta0
    )
    with torch.no_grad():
        ps.loc_s.fill_(mu)
        ps.rho_s.fill_(math.log(math.expm1(sigma)))  # softplus_inv(sigma)

    closed_form = float(ps.hyperprior_kl().detach())

    # MC in Y = s² space: q_Y = LogNormal(2μ, 2σ), p_Y = InvGamma(α₀, β₀).
    g = torch.Generator().manual_seed(6)
    n = 200_000
    log_y = 2.0 * mu + 2.0 * sigma * torch.randn(n, generator=g)
    y = torch.exp(log_y)
    q_y = torch.distributions.LogNormal(2.0 * mu, 2.0 * sigma, validate_args=True)
    p_y = torch.distributions.InverseGamma(alpha0, beta0, validate_args=True)
    mc = float((q_y.log_prob(y) - p_y.log_prob(y)).mean())

    assert closed_form == pytest.approx(mc, abs=0.05), (
        f"closed_form={closed_form:.4f}, mc={mc:.4f}"
    )


# ── 4. hierarchical half-Cauchy tier ──────────────────────────────────────────


def test_hier_hc_single_sample_estimator_is_unbiased_vs_quadrature():
    """Hierarchical half-Cauchy: the single-sample MC KL estimator is unbiased.

    There is no analytic KL[LogNormal ‖ HalfCauchy]; the training loop uses
    one reparameterized sample per call (accepted decision, ADR-0002/#86).
    Unbiasedness check: the mean of T independent single-sample estimates
    must agree with the true KL computed by 1-D numerical quadrature of
    q(s)·(log q(s) − log p(s)) over (0, ∞), with scipy.stats densities —
    fully independent of package code (acceptance criterion).

    Tolerance: quadrature error is ~3e-9 (negligible); MC error dominates.
    Per-sample std of the estimator at these parameters ≈ 0.67 →
    std_err ≈ 0.011 at T=4000; abs=0.06 ≈ 5.5σ under the fixed seed.
    """
    from scipy import integrate, stats

    from dune_bayes.priors import PriorScale

    mu, sigma, tau = 0.5, 0.3, 1.0
    ps = PriorScale(mode="hierarchical", hyperprior="half_cauchy", tau=tau)
    with torch.no_grad():
        ps.loc_s.fill_(mu)
        ps.rho_s.fill_(math.log(math.expm1(sigma)))  # softplus_inv(sigma)

    # True KL by quadrature — scipy densities only, no package code.
    q = stats.lognorm(s=sigma, scale=math.exp(mu))
    p = stats.halfcauchy(scale=tau)
    true_kl, quad_err = integrate.quad(
        lambda s: q.pdf(s) * (q.logpdf(s) - p.logpdf(s)), 0.0, math.inf
    )
    assert quad_err < 1e-6  # quadrature converged; MC error dominates below

    torch.manual_seed(8)
    T = 4000
    mc_mean = sum(float(ps.hyperprior_kl().detach()) for _ in range(T)) / T

    assert mc_mean == pytest.approx(true_kl, abs=0.06), (
        f"mc_mean={mc_mean:.4f}, quadrature={true_kl:.4f}"
    )
