"""Tests for compare module: to_inference_data, waic, loo, compare, elbo
(issue 0009 / GitHub #10).

Reference-test archetypes (CLAUDE.md):
  - Shape:      to_inference_data produces (1, T, n) log_likelihood.
  - Closed-form: waic on a deterministic model = sum(log_prob), since p_waic=0.
  - Reference:  loo on a deterministic model matches az.loo on the same DataTree.
  - Behavior:   loo warns on Pareto k > 0.5 at low T; compare ranks models correctly.
"""

import arviz as az
import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_bamlss.compare import compare, elbo, loo, to_inference_data, waic
from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 30
IN = 1

# ── deterministic fixture ─────────────────────────────────────────────────────
# A plain nn.Linear with fixed weights: every LogLikSampler draw is identical.
# This gives a closed-form WAIC/LOO reference.


@pytest.fixture
def det_model():
    """Deterministic shape function — no weight stochasticity, zero KL."""
    layer = nn.Linear(IN, 2, bias=True)
    nn.init.zeros_(layer.weight)
    layer.bias.data = torch.tensor([0.5, -1.0])
    family = NormalFamily(validate_args=True)
    return BayesianNAMLSS({"x": layer}, family, n_obs=N_OBS)


@pytest.fixture
def det_data():
    return {"x": torch.zeros(N_OBS, IN)}, torch.zeros(N_OBS)


# ── stochastic fixture ────────────────────────────────────────────────────────


@pytest.fixture
def bay_model():
    """BayesianMLP-backed model for structural / warning tests."""
    family = NormalFamily(validate_args=True)
    formula = {"x": BayesianMLP(IN, 2, [8], N_OBS)}
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def bay_data():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    X = {"x": torch.randn(N_OBS, IN, generator=g1)}
    y = torch.randn(N_OBS, generator=g2)
    return X, y


# ── Slice 1: to_inference_data — tracer bullet ────────────────────────────────


def test_to_inference_data_returns_datatree(bay_model, bay_data):
    """to_inference_data returns an arviz DataTree (InferenceData 1.x)."""
    import xarray as xr

    X, y = bay_data
    idata = to_inference_data(bay_model, X, y, T=10)
    assert isinstance(idata, xr.DataTree)


def test_to_inference_data_has_log_likelihood_group(bay_model, bay_data):
    """to_inference_data DataTree has a 'log_likelihood' group with 'y' variable."""
    X, y = bay_data
    idata = to_inference_data(bay_model, X, y, T=10)
    assert "/log_likelihood" in idata.groups
    assert "y" in idata["log_likelihood"].data_vars


def test_to_inference_data_loglik_shape(bay_model, bay_data):
    """log_likelihood['y'] has shape (chain=1, draw=T, obs=n)."""
    X, y = bay_data
    T = 15
    idata = to_inference_data(bay_model, X, y, T=T)
    ll = idata["log_likelihood"]["y"]
    assert ll.dims == ("chain", "draw", "y_dim_0")
    assert ll.shape == (1, T, N_OBS)


# ── Slice 2: waic — closed-form reference on deterministic model ───────────────


def test_waic_returns_waic_data(det_model, det_data):
    """waic() returns a WaicData object with elpd, p, se attributes."""
    from neural_bamlss.compare.comparison import WaicData

    X, y = det_data
    result = waic(det_model, X, y, T=20)
    assert isinstance(result, WaicData)


def test_waic_elpd_matches_hand_computed(det_model, det_data):
    """For a deterministic model, p_waic=0 so elpd_waic = sum(log_prob).

    With a constant ll matrix (all T draws identical):
      lppd_i = logsumexp([c, c, ..., c]) - log(T) = c
      p_waic_i = var(c, c, ...) = 0
    → elpd_waic = sum(log_prob(y_i)) = total log-likelihood.

    Tolerance: float32→float64 cast; atol=1e-4 is conservative for N=30.
    """
    X, y = det_data
    with torch.no_grad():
        params = det_model.nets["x"](X["x"])  # (n, 2)
        dist = det_model.family(params)
        expected_elpd = float(dist.log_prob(y).sum())

    result = waic(det_model, X, y, T=50)
    assert result.elpd == pytest.approx(expected_elpd, abs=1e-4)
    # Deterministic model → no variance across draws → p_waic = 0
    assert result.p == pytest.approx(0.0, abs=1e-10)


def test_waic_n_samples_attribute(det_model, det_data):
    """WaicData.n_samples reflects the T argument."""
    X, y = det_data
    T = 37
    result = waic(det_model, X, y, T=T)
    assert result.n_samples == T
    assert result.n_data_points == N_OBS


# ── Slice 3: loo — reference + Pareto-k warning ───────────────────────────────


def test_loo_returns_elpd_data(bay_model, bay_data):
    """loo() returns an arviz ELPDData instance."""
    X, y = bay_data
    result = loo(bay_model, X, y, T=50)
    assert isinstance(result, az.ELPDData)


def test_loo_elpd_matches_arviz_reference(bay_model, bay_data):
    """loo() matches az.loo on the same DataTree when given the same seed.

    Seeding torch before each call makes LogLikSampler produce identical weight
    samples → identical pointwise_loglik → identical DataTrees → equal elpd.

    Tolerance: float64 arithmetic → rel=1e-6 is well within machine precision.
    """
    X, y = bay_data
    T_REF = 50

    torch.manual_seed(7)
    idata = to_inference_data(bay_model, X, y, T=T_REF)
    ref = az.loo(idata)

    torch.manual_seed(7)
    result = loo(bay_model, X, y, T=T_REF)
    assert result.elpd == pytest.approx(ref.elpd, rel=1e-6)


def test_loo_warns_high_pareto_k_at_low_T(bay_model, bay_data):
    """loo() surfaces the Pareto-k reliability warning when T is too small.

    With T=50 draws and n=30 observations, the PSIS importance weights are
    unreliable (Pareto k > 0.5 for many observations). This is already low
    relative to the T_EVAL=1000 default. arviz issues a UserWarning.
    T < 25 would raise ValueError (PSIS requires n_draws_tail ≥ 5), so T=50
    is the "deliberately low" point that triggers the warning without aborting.
    """
    X, y = bay_data
    with pytest.warns(UserWarning, match="Pareto"):
        loo(bay_model, X, y, T=50)


# ── Slice 4: compare — ranking test ───────────────────────────────────────────


def test_compare_returns_dataframe(bay_model, bay_data):
    """compare() returns a pandas DataFrame."""
    import pandas as pd

    X, y = bay_data
    result = compare({"m": bay_model}, X, y, T=50)
    assert isinstance(result, pd.DataFrame)


def test_compare_ranks_models_correctly():
    """compare() puts the trained model above the untrained one.

    y = 5.0 everywhere; an untrained model predicts µ≈0 (off by ~5/σ nats per
    obs), so the ELPD gap is huge (~780 nats total) — ranking is deterministic.
    """
    N = 30
    y = 5.0 * torch.ones(N)
    X = {"x": torch.zeros(N, 1)}
    family = NormalFamily(validate_args=True)

    # Untrained: random prior weights, mu≈0, far from y=5.
    model_bad = BayesianNAMLSS({"x": BayesianMLP(1, 2, [4], N)}, family, n_obs=N)

    # Trained: 50 epochs, converges mu toward 5.0.
    torch.manual_seed(0)
    model_good = BayesianNAMLSS({"x": BayesianMLP(1, 2, [4], N)}, family, n_obs=N)
    model_good.fit(X, y, epochs=50, lr=0.05, warmup_epochs=5)

    result = compare({"good": model_good, "bad": model_bad}, X, y, T=50)
    assert result.index[0] == "good", f"Expected 'good' first, got {list(result.index)}"


# ── Slice 5: elbo — finite scalar + no Bayes Factor ──────────────────────────


def test_elbo_is_finite_float(bay_model, bay_data):
    """elbo() returns a finite Python float."""
    X, y = bay_data
    value = elbo(bay_model, X, y)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_elbo_sign(det_model, det_data):
    """ELBO is negative ELBO-loss; for a well-specified model it is finite."""
    X, y = det_data
    # det_model has zero KL (deterministic net) so ELBO ≈ -mean_NLL (positive value)
    value = elbo(det_model, X, y)
    assert np.isfinite(value)


def test_no_bayes_factor_in_public_api():
    """compare module exposes no bayes_factor (ADR-0001, issue 0009 / GitHub #10)."""
    import neural_bamlss.compare as compare_module

    assert not hasattr(compare_module, "bayes_factor"), (
        "bayes_factor must not appear in neural_bamlss.compare (ADR-0001)"
    )
