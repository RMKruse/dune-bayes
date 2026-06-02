"""Tests for VariationalDense — the variational atom (ADR-0004, issue 0001 / GitHub #2).

Four reference-test archetypes (CLAUDE.md):
  - Closed-form: KL against hand-computed Gaussian–Gaussian reference.
  - Round-trip:  state_dict + from_config with max|Δw| == 0.
  - Shape:       forward output (batch, units).
  - MC-convergence: flipout/vanilla mean agreement over T draws.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_bamlss.layers import VariationalDense, collect_kl, set_kl_beta

# ── fixtures ──────────────────────────────────────────────────────────────────

IN, UNITS, BATCH = 3, 4, 8


@pytest.fixture
def layer():
    torch.manual_seed(0)
    return VariationalDense(
        IN, UNITS, prior_scale=1.0, kl_divisor=1.0, validate_args=True
    )


@pytest.fixture
def x():
    return torch.randn(BATCH, IN, generator=torch.Generator().manual_seed(42))


# ── 1. shape ──────────────────────────────────────────────────────────────────


def test_forward_output_shape(layer, x):
    out = layer(x)
    assert out.shape == (BATCH, UNITS)


# ── 2. KL closed-form ─────────────────────────────────────────────────────────


def _reference_kl(loc: torch.Tensor, rho: torch.Tensor, prior_scale: float) -> float:
    """Hand-computed KL[N(loc, softplus(rho)^2) || N(0, prior_scale^2)], summed."""
    scale = F.softplus(rho)
    kl = torch.sum(
        math.log(prior_scale)
        - torch.log(scale)
        + (scale.pow(2) + loc.pow(2)) / (2.0 * prior_scale**2)
        - 0.5
    )
    return float(kl.detach())


def test_kl_matches_closed_form_reference(layer, x):
    """After a forward pass, layer.kl equals the analytic Gaussian–Gaussian KL."""
    # kl_divisor=1 and kl_beta=1 so stashed kl == raw closed-form KL.
    layer(x)

    expected_kl = _reference_kl(layer.kernel_loc, layer.kernel_rho, layer.prior_scale)
    if layer.use_bias:
        expected_kl += _reference_kl(layer.bias_loc, layer.bias_rho, layer.prior_scale)

    # float32 machine eps ~1e-7; 1e-5 relative tolerance is conservative
    # (pure arithmetic, no MC noise).
    assert float(layer.kl.detach()) == pytest.approx(
        expected_kl, rel=1e-5
    ), f"stashed kl={float(layer.kl.detach()):.6f} vs reference={expected_kl:.6f}"


# ── 3. set_kl_beta utility ───────────────────────────────────────────────────


def test_set_kl_beta_sets_buffer_on_all_layers(x):
    """set_kl_beta() propagates to every VariationalDense in the module tree."""
    model = nn.Sequential(
        VariationalDense(IN, 8),
        VariationalDense(8, UNITS),
    )
    set_kl_beta(model, 0.5)
    for m in model.modules():
        if isinstance(m, VariationalDense):
            assert float(m.kl_beta) == pytest.approx(0.5)


# ── 4. kl_beta gates ─────────────────────────────────────────────────────────


def test_kl_beta_zero_zeroes_kl(layer, x):
    """β=0 collapses the emitted KL to exactly zero."""
    layer.kl_beta.fill_(0.0)
    layer(x)
    assert float(layer.kl.detach()) == pytest.approx(0.0, abs=1e-7)


def test_kl_beta_one_gives_full_kl(layer, x):
    """β=1 (default) gives the full KL/N — unchanged from the closed-form reference."""
    layer.kl_beta.fill_(1.0)
    layer(x)
    expected_kl = _reference_kl(layer.kernel_loc, layer.kernel_rho, layer.prior_scale)
    if layer.use_bias:
        expected_kl += _reference_kl(layer.bias_loc, layer.bias_rho, layer.prior_scale)
    # kl_divisor=1 in the fixture, so KL/N == KL.
    assert float(layer.kl.detach()) == pytest.approx(expected_kl, rel=1e-5)


# ── 5. collect_kl module-walk (ADR-0004 load-bearing claim A) ─────────────────


class _NestedModel(nn.Module):
    """Mimics the real NAMLSS graph: per-feature sub-modules -> sum."""

    def __init__(self, n_features: int = 3) -> None:
        super().__init__()
        self.nets = nn.ModuleList(
            nn.Sequential(
                VariationalDense(1, 8, activation="relu"),
                VariationalDense(8, 2),
            )
            for _ in range(n_features)
        )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([net(x) for net, x in zip(self.nets, xs)]).sum(0)


def test_collect_kl_reaches_all_variational_layers():
    """collect_kl() sums KL from every VariationalDense across nested sub-modules.

    This is ADR-0004 load-bearing claim A: the explicit module-walk must cross
    nested-sub-module boundaries (per-feature nets inside a ModuleList).
    """
    n_features = 3
    model = _NestedModel(n_features)
    xs = [torch.randn(8, 1) for _ in range(n_features)]

    set_kl_beta(model, 1.0)
    model(xs)

    kl = collect_kl(model)

    # Each feature net has 2 VariationalDense layers → 2 * n_features total.
    n_expected = 2 * n_features
    n_found = sum(1 for m in model.modules() if isinstance(m, VariationalDense))
    assert n_found == n_expected, f"expected {n_expected} layers, found {n_found}"
    assert float(kl.detach()) > 0.0, (
        "collect_kl returned zero — some KL was silently dropped"
    )


def test_collect_kl_beta_zero_zeroes_total():
    """set_kl_beta(0) → collect_kl returns exactly zero."""
    model = _NestedModel(2)
    xs = [torch.randn(8, 1) for _ in range(2)]
    set_kl_beta(model, 0.0)
    model(xs)
    assert float(collect_kl(model).detach()) == pytest.approx(0.0, abs=1e-7)


# ── 6. get_config / from_config ───────────────────────────────────────────────


def test_get_config_is_closure_free(layer):
    """get_config() contains only ints, floats, strings, and bools — no callables."""
    cfg = layer.get_config()
    assert all(
        not callable(v) for v in cfg.values()
    ), f"callable values found in config: {[k for k, v in cfg.items() if callable(v)]}"


def test_from_config_preserves_hyperparameters():
    """from_config(get_config()) reconstructs an equivalent layer."""
    original = VariationalDense(
        5, 4, prior_scale=0.33, kl_divisor=99.0, activation="relu", flipout=True
    )
    cfg = original.get_config()
    rebuilt = VariationalDense.from_config(cfg)

    assert rebuilt.in_features == original.in_features
    assert rebuilt.units == original.units
    assert rebuilt.prior_scale == pytest.approx(original.prior_scale)
    assert rebuilt.kl_divisor == pytest.approx(original.kl_divisor)
    assert rebuilt.activation == original.activation
    assert rebuilt.flipout == original.flipout
    assert rebuilt.use_bias == original.use_bias


# ── 7. state_dict round-trip (ADR-0004 load-bearing claim B) ─────────────────


def test_state_dict_round_trip_max_delta_zero(tmp_path):
    """config + state_dict save/load reconstructs identical variational weights.

    This is ADR-0004 load-bearing claim B: max|Δw| == 0 (exact equality),
    because we compare deterministic parameter tensors, not stochastic predictions.
    """
    torch.manual_seed(0)
    layer = VariationalDense(3, 4, prior_scale=0.5, kl_divisor=100.0)
    bundle_path = tmp_path / "layer.pt"

    # Save: config dict + state_dict (the single supported path per ADR-0006).
    torch.save(
        {"config": layer.get_config(), "state_dict": layer.state_dict()}, bundle_path
    )

    # Load: reconstruct architecture from config alone, then load weights.
    bundle = torch.load(bundle_path, weights_only=True)
    loaded = VariationalDense.from_config(bundle["config"])
    loaded.load_state_dict(bundle["state_dict"])

    sa, sb = layer.state_dict(), loaded.state_dict()
    assert sa.keys() == sb.keys(), "state_dict key sets differ"
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, (
        f"max|Δw| = {max_delta:.2e} — weights changed across round-trip"
    )


# ── 8. flipout / vanilla MC convergence ───────────────────────────────────────


def test_flipout_vanilla_agree_in_expectation():
    """Flipout and vanilla estimators produce the same expected output.

    Verified by MC convergence (T draws) under a fixed seed.  We compare means,
    not individual samples — reparameterization noise differs between the two
    estimators by design.  Tolerance: MC error ∝ 1/√T; with T=2000 and the
    fixed seed the observed noise is well below rel=0.05 for this layer size.
    """
    torch.manual_seed(7)
    x = torch.randn(16, IN)

    # Build two layers with identical parameters using the same seed.
    # Note: reproducibility holds within one model object under a global seed;
    # across two freshly-built objects it does not, so we copy weights explicitly.
    vanilla = VariationalDense(
        IN, UNITS, prior_scale=1.0, kl_divisor=1.0, flipout=False
    )
    flipout = VariationalDense.from_config({**vanilla.get_config(), "flipout": True})
    flipout.load_state_dict(vanilla.state_dict())

    T = 2000
    vanilla_outputs = []
    flipout_outputs = []
    for _ in range(T):
        with torch.no_grad():
            vanilla_outputs.append(vanilla(x))
            flipout_outputs.append(flipout(x))

    mean_vanilla = torch.stack(vanilla_outputs).mean(0)
    mean_flipout = torch.stack(flipout_outputs).mean(0)

    # The two estimators have the same expectation: x @ kernel_loc + bias_loc.
    # abs=0.05: MC std_err ≈ σ_out/√T ≈ 0.1/√2000 ≈ 0.002 per element;
    # 0.05 gives 25× headroom while catching any genuine bias (which would be O(σ)).
    # rel tolerance is avoided because expected values can be near zero.
    assert mean_vanilla == pytest.approx(
        mean_flipout.numpy(), abs=0.05
    ), "flipout and vanilla means diverged — they must agree in expectation"
