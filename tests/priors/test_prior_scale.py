"""Tests for PriorScale — three-tier prior variance handle (ADR-0002, issue 0011 / #12).

Four reference-test archetypes (CLAUDE.md):
  - Closed-form:      IG KL against analytic Gaussian–Gaussian/LogNormal–IG reference.
  - Round-trip:       state_dict + from_config with max|Δw| == 0.
  - Shape:            forward() returns a positive scalar tensor.
  - MC-convergence:   half-Cauchy KL averaged over T draws converges to reference.
"""

import math

import pytest
import torch

# ── tracer bullet ─────────────────────────────────────────────────────────────


def test_import():
    """PriorScale is importable from neural_bamlss.priors."""
    from neural_bamlss.priors import PriorScale  # noqa: F401


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fixed():
    return pytest.importorskip("neural_bamlss.priors").PriorScale(
        mode="fixed", scale=2.0
    )


@pytest.fixture
def eb():
    return pytest.importorskip("neural_bamlss.priors").PriorScale(
        mode="empirical_bayes", scale=1.0
    )


@pytest.fixture
def hier_ig():
    return pytest.importorskip("neural_bamlss.priors").PriorScale(
        mode="hierarchical",
        hyperprior="inverse_gamma",
        scale=1.0,
        alpha0=2.0,
        beta0=1.0,
    )


@pytest.fixture
def hier_hc():
    return pytest.importorskip("neural_bamlss.priors").PriorScale(
        mode="hierarchical", hyperprior="half_cauchy", scale=1.0, tau=1.0
    )


# ── 1. fixed tier ─────────────────────────────────────────────────────────────


def test_fixed_forward_returns_configured_scale(fixed):
    """Fixed tier: forward() returns the exact configured scale as a scalar tensor."""
    out = fixed()
    assert out.shape == ()
    assert float(out) == pytest.approx(2.0)


def test_fixed_kl_is_zero(fixed):
    """Fixed tier: hyperprior_kl() returns scalar zero — no hyperprior contribution."""
    kl = fixed.hyperprior_kl()
    assert kl.shape == ()
    assert float(kl) == pytest.approx(0.0, abs=1e-7)


def test_fixed_no_parameters(fixed):
    """Fixed tier has no trainable parameters — the scale is a buffer constant."""
    assert len(list(fixed.parameters())) == 0


def test_fixed_get_config_roundtrip():
    """Fixed tier: from_config(get_config()) preserves mode and scale."""
    from neural_bamlss.priors import PriorScale

    original = PriorScale(mode="fixed", scale=3.14)
    cfg = original.get_config()
    rebuilt = PriorScale.from_config(cfg)

    assert rebuilt.mode == original.mode
    assert float(rebuilt()) == pytest.approx(3.14)


# ── 2. empirical-Bayes tier ───────────────────────────────────────────────────


def test_eb_forward_starts_near_initial_scale(eb):
    """EB tier: forward() starts close to initial_scale at construction."""
    s = float(eb().detach())
    # softplus(softplus_inv(1.0)) == 1.0 — round-trip check
    assert s == pytest.approx(1.0, rel=1e-4)


def test_eb_forward_is_positive(eb):
    """EB tier: forward() always returns a positive scale (softplus guarantee)."""
    assert float(eb().detach()) > 0.0


def test_eb_gradients_flow_through_scale(eb):
    """EB tier: the scale is a Parameter so gradients flow for ELBO optimisation."""
    s = eb()
    loss = s * 2.0
    loss.backward()
    assert eb.rho.grad is not None
    assert float(eb.rho.grad) != 0.0


def test_eb_kl_is_zero(eb):
    """EB tier: no hyperprior → hyperprior_kl() returns zero."""
    kl = eb.hyperprior_kl()
    assert kl.shape == ()
    assert float(kl) == pytest.approx(0.0, abs=1e-7)


def test_eb_state_dict_round_trip(tmp_path):
    """EB tier: config + state_dict save/load reconstructs identical rho weight."""
    from neural_bamlss.priors import PriorScale

    layer = PriorScale(mode="empirical_bayes", scale=0.5)
    bundle_path = tmp_path / "eb.pt"
    torch.save(
        {"config": layer.get_config(), "state_dict": layer.state_dict()}, bundle_path
    )

    bundle = torch.load(bundle_path, weights_only=True)
    loaded = PriorScale.from_config(bundle["config"])
    loaded.load_state_dict(bundle["state_dict"])

    sa, sb = layer.state_dict(), loaded.state_dict()
    assert sa.keys() == sb.keys()
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, f"max|Δw| = {max_delta:.2e}"


# ── 3. hierarchical IG tier ───────────────────────────────────────────────────


def _kl_ig_reference(mu: float, sigma: float, alpha0: float, beta0: float) -> float:
    """Reference: KL[LogNormal(mu, sigma²) || sqrt(InvGamma(alpha0, beta0))].

    Computed via E_q[log q(s)] and E_q[log p(s)] using log-normal moments:
      E_q[log q(s)] = -mu - 1/2 - 1/2 log(2π sigma²)   [entropy of LogNormal]
      E_q[log p(s)]: uses E[log s] = mu and E[1/s²] = exp(-2mu + 2sigma²).
    """
    # E[log q(s)] = -H[q], H[LogNormal(mu,sigma²)] = mu + 1/2 + 1/2 log(2πsigma²)
    e_log_q = -mu - 0.5 - 0.5 * math.log(2.0 * math.pi * sigma**2)

    # log p(s) = log2 + alpha0*log(beta0) - lgamma(alpha0)
    #            + (-2*alpha0-1)*log(s) - beta0/s²
    # E_q uses E[log s] = mu and E[1/s²] = exp(-2mu + 2sigma²)
    e_log_p = (
        math.log(2.0)
        + alpha0 * math.log(beta0)
        - math.lgamma(alpha0)
        + (-2.0 * alpha0 - 1.0) * mu
        - beta0 * math.exp(-2.0 * mu + 2.0 * sigma**2)
    )

    return e_log_q - e_log_p


def test_hier_ig_forward_is_positive(hier_ig):
    """Hierarchical IG: forward() returns a positive scalar (reparameterized sample)."""
    torch.manual_seed(0)
    s = hier_ig()
    assert s.shape == ()
    assert float(s.detach()) > 0.0


def test_hier_ig_forward_is_stochastic(hier_ig):
    """Hierarchical IG: consecutive forward() calls differ (reparameterization)."""
    s1 = float(hier_ig().detach())
    s2 = float(hier_ig().detach())
    assert s1 != s2


def test_hier_ig_kl_matches_closed_form_reference():
    """Hierarchical IG: kl() equals the analytic KL at known variational params.

    We set loc_s and rho_s to known values and compare kl() against the
    independent reference _kl_ig_reference().

    Tolerance: float32 arithmetic only (no MC noise) — rel=1e-4 is conservative.
    """
    from neural_bamlss.priors import PriorScale

    mu_val = 0.3
    sigma_val = 0.4  # softplus(rho) == 0.4 → need to solve for rho_s

    alpha0, beta0 = 2.0, 1.5
    ps = PriorScale(
        mode="hierarchical", hyperprior="inverse_gamma", alpha0=alpha0, beta0=beta0
    )

    # Override variational parameters to known values for the reference comparison.
    # softplus_inv(0.4) so that softplus(rho_s) == 0.4
    rho_val = math.log(math.expm1(sigma_val))
    with torch.no_grad():
        ps.loc_s.fill_(mu_val)
        ps.rho_s.fill_(rho_val)

    kl_computed = float(ps.hyperprior_kl().detach())
    kl_reference = _kl_ig_reference(mu_val, sigma_val, alpha0, beta0)

    # Pure arithmetic (no randomness in IG KL) — rel=1e-4 is conservative for float32.
    assert kl_computed == pytest.approx(kl_reference, rel=1e-4), (
        f"computed={kl_computed:.6f}, reference={kl_reference:.6f}"
    )


def test_hier_ig_kl_is_positive(hier_ig):
    """Hierarchical IG: hyperprior_kl() is non-negative (KL divergence property)."""
    # Closed-form KL — deterministic, not MC. Always >= 0.
    torch.manual_seed(5)
    kl = float(hier_ig.hyperprior_kl().detach())
    assert kl >= 0.0


# ── 4. hierarchical half-Cauchy tier ─────────────────────────────────────────


def _kl_half_cauchy_mc_reference(
    mu: float, sigma: float, tau: float, n: int = 200_000, seed: int = 42
) -> float:
    """MC reference for KL[LogNormal(mu, sigma²) || HalfCauchy(0, tau)].

    Uses n importance-weighted samples from LogNormal(mu, sigma) to compute
    E_q[log q(s) - log p(s)] numerically. n=200_000 gives std_err < 0.01.
    """
    g = torch.Generator().manual_seed(seed)
    # Draw log_s ~ N(mu, sigma²) via reparameterization
    log_s = torch.randn(n, generator=g) * sigma + mu
    s = torch.exp(log_s)

    # log q(s): LogNormal(mu, sigma²) density in s-space
    log_q = (
        -0.5 * math.log(2.0 * math.pi * sigma**2)
        - (log_s - mu).pow(2) / (2.0 * sigma**2)
        - log_s
    )

    # log p(s): HalfCauchy(0, tau) density
    log_p = math.log(2.0 / math.pi) - math.log(tau) - torch.log(1.0 + (s / tau).pow(2))

    return float((log_q - log_p).mean())


def test_hier_hc_forward_is_positive(hier_hc):
    """Hierarchical half-Cauchy: forward() returns a positive scalar."""
    torch.manual_seed(1)
    s = hier_hc()
    assert s.shape == ()
    assert float(s.detach()) > 0.0


def test_hier_hc_forward_is_stochastic(hier_hc):
    """Hierarchical half-Cauchy: consecutive forward() calls differ."""
    s1 = float(hier_hc().detach())
    s2 = float(hier_hc().detach())
    assert s1 != s2


def test_hier_hc_kl_mc_converges_to_reference():
    """Hierarchical half-Cauchy: MC-averaged kl() converges to numerical reference.

    kl() is a single-sample reparameterized estimator (no closed form for
    KL[LogNormal || HalfCauchy]). Averaged over T=2000 calls under a fixed seed,
    it should agree with the high-T reference within MC noise tolerance.

    Tolerance: std_err of single-sample KL estimator ≈ σ_kl / √T.
    abs=0.15 gives ample headroom for the noise level at T=2000.
    """
    from neural_bamlss.priors import PriorScale

    mu_val, sigma_val, tau_val = 0.5, 0.3, 1.0
    rho_val = math.log(math.expm1(sigma_val))

    ps = PriorScale(mode="hierarchical", hyperprior="half_cauchy", tau=tau_val)
    with torch.no_grad():
        ps.loc_s.fill_(mu_val)
        ps.rho_s.fill_(rho_val)

    torch.manual_seed(7)
    T = 2000
    kl_samples = [float(ps.hyperprior_kl().detach()) for _ in range(T)]
    kl_mean = sum(kl_samples) / T

    reference = _kl_half_cauchy_mc_reference(mu_val, sigma_val, tau_val)

    # MC error: std_err ≈ std(kl_samples)/√T; abs=0.15 gives ample headroom.
    assert kl_mean == pytest.approx(reference, abs=0.15), (
        f"MC mean kl={kl_mean:.4f}, reference={reference:.4f}"
    )


def test_hier_hc_kl_is_positive_on_average():
    """Hierarchical half-Cauchy: MC-averaged kl() > 0 (KL divergence property)."""
    from neural_bamlss.priors import PriorScale

    torch.manual_seed(99)
    ps = PriorScale(mode="hierarchical", hyperprior="half_cauchy", tau=1.0)
    kl_mean = sum(float(ps.hyperprior_kl().detach()) for _ in range(500)) / 500
    assert kl_mean > 0.0


# ── 5. hierarchical state_dict round-trip ─────────────────────────────────────


def test_hierarchical_state_dict_round_trip(tmp_path):
    """Hierarchical tier: config + state_dict save/load reconstructs identical params.

    max|Δw| == 0 on deterministic parameter tensors (loc_s, rho_s).
    """
    from neural_bamlss.priors import PriorScale

    torch.manual_seed(2)
    layer = PriorScale(
        mode="hierarchical", hyperprior="half_cauchy", tau=2.0, scale=0.5
    )
    bundle_path = tmp_path / "hier.pt"
    torch.save(
        {"config": layer.get_config(), "state_dict": layer.state_dict()}, bundle_path
    )

    bundle = torch.load(bundle_path, weights_only=True)
    loaded = PriorScale.from_config(bundle["config"])
    loaded.load_state_dict(bundle["state_dict"])

    sa, sb = layer.state_dict(), loaded.state_dict()
    assert sa.keys() == sb.keys()
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, f"max|Δw| = {max_delta:.2e}"


# ── 6. VariationalLayer contract (issue #73) ─────────────────────────────────
#
# PriorScale stashes its own hyperprior KL on forward(), so collect_kl()
# reaches it through the module walk — and counts a shared handle exactly
# once, because nn.Module.modules() deduplicates shared submodules.


def test_hierarchical_forward_stashes_hyperprior_kl_for_collect_kl():
    """After forward(), .kl holds β·hyperprior_kl/kl_divisor and collect_kl sees it.

    IG hyperprior: closed-form KL, deterministic — exact comparison possible.
    Tolerance rel=1e-5: pure float32 arithmetic, no MC noise.
    """
    from neural_bamlss.layers import collect_kl
    from neural_bamlss.priors import PriorScale

    mu_val, sigma_val, alpha0, beta0 = 0.3, 0.4, 2.0, 1.5
    ps = PriorScale(
        mode="hierarchical",
        hyperprior="inverse_gamma",
        alpha0=alpha0,
        beta0=beta0,
        kl_divisor=4.0,
    )
    rho_val = math.log(math.expm1(sigma_val))
    with torch.no_grad():
        ps.loc_s.fill_(mu_val)
        ps.rho_s.fill_(rho_val)

    ps()  # forward stashes the hyperprior KL

    expected = _kl_ig_reference(mu_val, sigma_val, alpha0, beta0) / 4.0
    assert float(ps.kl.detach()) == pytest.approx(expected, rel=1e-5)
    assert float(collect_kl(ps).detach()) == pytest.approx(expected, rel=1e-5)


def test_fixed_and_eb_forward_stash_zero_kl(fixed, eb):
    """Fixed / EB tiers stash a structural zero on forward() — no hyperprior."""
    fixed()
    eb()
    assert float(fixed.kl.detach()) == pytest.approx(0.0, abs=1e-7)
    assert float(eb.kl.detach()) == pytest.approx(0.0, abs=1e-7)


def test_set_kl_beta_scales_stashed_hyperprior_kl():
    """set_kl_beta() reaches PriorScale — warm-up anneals the hyperprior KL too."""
    from neural_bamlss.layers import set_kl_beta
    from neural_bamlss.priors import PriorScale

    ps = PriorScale(mode="hierarchical", hyperprior="inverse_gamma")
    set_kl_beta(ps, 0.0)
    ps()
    assert float(ps.kl.detach()) == pytest.approx(0.0, abs=1e-7)


def test_kl_divisor_round_trips_in_config():
    """kl_divisor is part of the closure-free config round-trip."""
    from neural_bamlss.priors import PriorScale

    ps = PriorScale(mode="empirical_bayes", scale=0.5, kl_divisor=128.0)
    rebuilt = PriorScale.from_config(ps.get_config())
    assert rebuilt.kl_divisor == 128.0


# ── 7. validation ─────────────────────────────────────────────────────────────


def test_invalid_mode_raises():
    """Unsupported mode string raises ValueError immediately at construction."""
    from neural_bamlss.priors import PriorScale

    with pytest.raises(ValueError, match="unknown mode"):
        PriorScale(mode="banana")


def test_invalid_hyperprior_raises():
    """Unsupported hyperprior string raises ValueError at construction."""
    from neural_bamlss.priors import PriorScale

    with pytest.raises(ValueError, match="unknown hyperprior"):
        PriorScale(mode="hierarchical", hyperprior="gamma")


# ── 8. get_config is closure-free ────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "fixed", "scale": 1.5},
        {"mode": "empirical_bayes", "scale": 0.3},
        {
            "mode": "hierarchical",
            "hyperprior": "half_cauchy",
            "tau": 2.0,
            "scale": 0.5,
        },
        {
            "mode": "hierarchical",
            "hyperprior": "inverse_gamma",
            "alpha0": 1.5,
            "beta0": 2.0,
        },
    ],
)
def test_get_config_is_closure_free(kwargs):
    """get_config() contains only primitive types across all tiers."""
    from neural_bamlss.priors import PriorScale

    ps = PriorScale(**kwargs)
    cfg = ps.get_config()
    assert all(not callable(v) for v in cfg.values()), (
        f"callable values in config: {[k for k, v in cfg.items() if callable(v)]}"
    )
    # Round-trip: from_config(get_config()) must not raise
    PriorScale.from_config(cfg)
