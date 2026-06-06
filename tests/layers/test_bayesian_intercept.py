"""Tests for BayesianIntercept — variational intercept layer (issue 0010 / GitHub #11).

Four reference-test archetypes (CLAUDE.md):
  - Shape:       forward output (units,).
  - Closed-form: KL against hand-computed Gaussian–Gaussian reference.
  - Round-trip:  state_dict + from_config with max|Δw| == 0.
  - MC-convergence: variational mean converges to loc over T draws.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dune_bayes.layers import (
    BayesianIntercept,
    VariationalDense,
    collect_kl,
    set_kl_beta,
)

UNITS = 3


@pytest.fixture
def intercept():
    torch.manual_seed(0)
    return BayesianIntercept(units=UNITS, prior_scale=10.0, validate_args=True)


# ── 1. shape ──────────────────────────────────────────────────────────────────


def test_forward_output_shape(intercept):
    out = intercept()
    assert out.shape == (UNITS,)


# ── 2. KL closed-form ─────────────────────────────────────────────────────────


def _reference_kl(loc: torch.Tensor, rho: torch.Tensor, prior_scale: float) -> float:
    scale = F.softplus(rho)
    kl = torch.sum(
        math.log(prior_scale)
        - torch.log(scale)
        + (scale.pow(2) + loc.pow(2)) / (2.0 * prior_scale**2)
        - 0.5
    )
    return float(kl.detach())


def test_kl_matches_closed_form_reference(intercept):
    """After forward(), stashed .kl equals analytic Gaussian–Gaussian KL."""
    intercept()
    expected = _reference_kl(intercept.loc, intercept.rho, intercept.prior_scale)
    # float32, no MC noise — rel=1e-5 is conservative.
    assert float(intercept.kl.detach()) == pytest.approx(expected, rel=1e-5)


# ── 3. point mode ─────────────────────────────────────────────────────────────


def test_point_mode_returns_loc_exactly():
    """In point mode forward() returns self.loc without stochastic noise."""
    layer = BayesianIntercept(units=2, mode="point")
    with torch.no_grad():
        out = layer()
    assert torch.equal(out, layer.loc)


def test_point_mode_kl_is_zero():
    """In point mode no KL is contributed — deterministic parameter, no posterior."""
    layer = BayesianIntercept(units=2, mode="point")
    layer()
    assert float(layer.kl.detach()) == pytest.approx(0.0, abs=1e-7)


# ── 4. collect_kl aggregates intercept KL ────────────────────────────────────


class _ModelWithIntercept(nn.Module):
    """Minimal stand-in for BayesianNAMLSS: one feature net + one intercept."""

    def __init__(self) -> None:
        super().__init__()
        self.net = VariationalDense(2, 4)
        self.intercept = BayesianIntercept(units=4, prior_scale=10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.intercept()


def test_collect_kl_includes_bayesian_intercept():
    """collect_kl() must reach BayesianIntercept KL alongside VariationalDense KL."""
    model = _ModelWithIntercept()
    x = torch.randn(8, 2)
    model(x)

    total_kl = float(collect_kl(model).detach())
    net_kl = float(model.net.kl.detach())
    intercept_kl = float(model.intercept.kl.detach())

    assert total_kl == pytest.approx(net_kl + intercept_kl, rel=1e-5), (
        "collect_kl dropped the intercept KL contribution"
    )
    assert intercept_kl > 0.0, "variational intercept KL should be positive"


def test_collect_kl_point_intercept_contributes_zero():
    """A point-mode intercept adds zero to collect_kl."""

    class _ModelPointIntercept(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = VariationalDense(2, 4)
            self.intercept = BayesianIntercept(units=4, mode="point")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) + self.intercept()

    model = _ModelPointIntercept()
    model(torch.randn(8, 2))

    total_kl = float(collect_kl(model).detach())
    net_kl = float(model.net.kl.detach())
    # point intercept contributes nothing, so total == net KL.
    assert total_kl == pytest.approx(net_kl, rel=1e-5)


# ── 5. set_kl_beta propagates to BayesianIntercept ───────────────────────────


def test_set_kl_beta_gates_intercept_kl(intercept):
    """set_kl_beta(0) zeroes out BayesianIntercept KL via collect_kl."""
    set_kl_beta(intercept, 0.0)
    intercept()
    assert float(intercept.kl.detach()) == pytest.approx(0.0, abs=1e-7)


def test_set_kl_beta_propagates_through_model():
    """set_kl_beta() reaches BayesianIntercept inside a composite module."""
    model = _ModelWithIntercept()
    set_kl_beta(model, 0.5)
    assert float(model.intercept.kl_beta) == pytest.approx(0.5)
    assert float(model.net.kl_beta) == pytest.approx(0.5)


# ── 6. get_config / from_config ───────────────────────────────────────────────


def test_get_config_is_closure_free(intercept):
    """get_config() contains only ints, floats, and strings — no callables."""
    cfg = intercept.get_config()
    assert all(not callable(v) for v in cfg.values())


def test_from_config_preserves_hyperparameters():
    """from_config(get_config()) reconstructs an equivalent BayesianIntercept."""
    original = BayesianIntercept(
        units=5, prior_scale=7.0, kl_divisor=50.0, mode="point"
    )
    rebuilt = BayesianIntercept.from_config(original.get_config())

    assert rebuilt.units == original.units
    assert rebuilt.prior_scale == pytest.approx(original.prior_scale)
    assert rebuilt.kl_divisor == pytest.approx(original.kl_divisor)
    assert rebuilt.mode == original.mode


# ── 7. state_dict round-trip ──────────────────────────────────────────────────


def test_state_dict_round_trip_max_delta_zero(tmp_path):
    """config + state_dict save/load reconstructs identical variational weights.

    ADR-0004 load-bearing claim B pattern: max|Δw| == 0 (exact equality on
    deterministic parameter tensors, not stochastic predictions).
    """
    torch.manual_seed(1)
    layer = BayesianIntercept(units=4, prior_scale=5.0, kl_divisor=100.0)
    bundle_path = tmp_path / "intercept.pt"

    torch.save(
        {"config": layer.get_config(), "state_dict": layer.state_dict()}, bundle_path
    )

    bundle = torch.load(bundle_path, weights_only=True)
    loaded = BayesianIntercept.from_config(bundle["config"])
    loaded.load_state_dict(bundle["state_dict"])

    sa, sb = layer.state_dict(), loaded.state_dict()
    assert sa.keys() == sb.keys()
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, (
        f"max|Δw| = {max_delta:.2e} — weights changed across round-trip"
    )


# ── 8. MC convergence — uncertainty shows up in predictive mean ───────────────


def test_variational_samples_are_stochastic():
    """Consecutive forward() calls produce different samples (not a constant)."""
    torch.manual_seed(42)
    layer = BayesianIntercept(units=4, prior_scale=10.0)
    s1 = layer().detach().clone()
    s2 = layer().detach().clone()
    assert not torch.equal(s1, s2), (
        "variational forward produced identical samples — no randomness"
    )


def test_variational_mean_converges_to_loc():
    """Mean of T samples converges to loc (acceptance criterion: intercept uncertainty
    in the response-level predictive mean).

    Tolerance: MC std_err ≈ softplus(-3)/√T ≈ 0.049/√2000 ≈ 0.001 per element;
    abs=0.05 gives 50× headroom while catching any genuine bias.
    """
    torch.manual_seed(3)
    layer = BayesianIntercept(units=4, prior_scale=10.0)

    T = 2000
    samples = torch.stack([layer().detach() for _ in range(T)])
    mean_sample = samples.mean(0)

    assert mean_sample == pytest.approx(layer.loc.detach().numpy(), abs=0.05), (
        "variational intercept mean drifted from loc — posterior mean should equal loc"
    )


# ── sample-dimension draws (issue 0027 / GitHub #80) ─────────────────────────


def test_sample_dim_shape_and_independence(intercept):
    """n_samples=S returns (S, units) with S independent intercept draws.

    The vectorized predict_params sweep needs a fresh intercept per posterior
    draw — a single draw broadcast S ways would understate epistemic spread.
    Two independent float32 draws coinciding exactly has probability ~0.
    """
    S = 4
    with torch.no_grad():
        out = intercept(n_samples=S)
    assert out.shape == (S, UNITS)
    for s in range(1, S):
        assert not torch.equal(out[0], out[s]), (
            f"slice {s} equals slice 0 — intercept draw broadcast across samples"
        )


def test_sample_dim_point_mode_expands_loc():
    """Point mode under n_samples returns loc replicated — deterministic, no KL."""
    point = BayesianIntercept(units=UNITS, mode="point")
    with torch.no_grad():
        out = point(n_samples=4)
    assert out.shape == (4, UNITS)
    assert torch.equal(out, point.loc.detach().expand(4, UNITS))
    assert float(point.kl) == 0.0
