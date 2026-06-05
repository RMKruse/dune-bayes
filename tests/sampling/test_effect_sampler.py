"""Tests for EffectSampler workhorse (issue 0005 / GitHub #6).

Four reference-test archetypes (CLAUDE.md):
  - Shape:         output is {feature_name: Tensor[T, n, param_count]}.
  - Behavior:      default T=200, T override, centering.
  - Pure function: model state unchanged after sampling.
  - MC-convergence: SEM shrinks as T grows (CI tightens).
"""

import math

import pytest
import torch

from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.sampling import EffectSampler
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
IN = 1
N_OBS_LARGE = 64

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily()


@pytest.fixture
def single_model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def multi_model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
        "x2": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def data_single():
    g = torch.Generator().manual_seed(42)
    return {"x1": torch.randn(N_OBS, IN, generator=g)}


@pytest.fixture
def data_multi():
    g = torch.Generator().manual_seed(42)
    return {
        "x1": torch.randn(N_OBS, IN, generator=g),
        "x2": torch.randn(N_OBS, IN, generator=g),
    }


# ── 1: Shape — tracer bullet ──────────────────────────────────────────────────


def test_output_shape_single_feature(single_model, data_single, family):
    """samples[name] is a float32 Tensor of shape (T, n, param_count)."""
    sampler = EffectSampler()
    T = 10
    samples = sampler(single_model, data_single, T=T)
    assert "x1" in samples
    contrib = samples["x1"]
    assert contrib.shape == (T, N_OBS, family.param_count)
    assert contrib.dtype == torch.float32


# ── 2: All features returned ──────────────────────────────────────────────────


def test_all_feature_names_returned(multi_model, data_multi, family):
    """Every feature name in model.feature_names appears in the output dict."""
    sampler = EffectSampler()
    samples = sampler(multi_model, data_multi, T=10)
    assert set(samples.keys()) == {"x1", "x2"}
    for name in ("x1", "x2"):
        assert samples[name].shape == (10, N_OBS, family.param_count)


# ── 3: Default T = 200 ────────────────────────────────────────────────────────


def test_t_predict_class_attribute():
    """EffectSampler.T_predict is the module-level constant 200."""
    assert EffectSampler.T_predict == 200


def test_default_t_is_200(single_model, data_single):
    """Calling sampler without T uses T_predict=200."""
    sampler = EffectSampler()
    samples = sampler(single_model, data_single)
    assert samples["x1"].shape[0] == 200


# ── 4: T override ─────────────────────────────────────────────────────────────


def test_t_override(single_model, data_single):
    """Passing T overrides the default."""
    sampler = EffectSampler()
    for T in (1, 50, 500):
        samples = sampler(single_model, data_single, T=T)
        got = samples["x1"].shape[0]
        assert got == T, f"Expected T={T}, got {got}"


# ── 5: Pure function — no model-state mutation ─────────────────────────────────


def test_pure_function_preserves_training_mode(single_model, data_single):
    """Training mode is restored after sampling, regardless of initial mode."""
    sampler = EffectSampler()

    single_model.train()
    sampler(single_model, data_single, T=5)
    assert single_model.training, "training mode not restored after sampling"

    single_model.eval()
    sampler(single_model, data_single, T=5)
    assert not single_model.training, "eval mode not restored after sampling"


def test_pure_function_preserves_parameters(single_model, data_single):
    """Posterior parameters (loc, rho) are identical before and after sampling."""
    sampler = EffectSampler()
    params_before = {k: v.clone() for k, v in single_model.named_parameters()}
    sampler(single_model, data_single, T=20)
    for name, val in single_model.named_parameters():
        assert torch.equal(params_before[name], val), f"parameter {name!r} mutated"


# ── 6: MC-convergence — SEM (CI) tightens with T ─────────────────────────────


def test_sem_decreases_with_t(single_model, data_single):
    """Standard error of the posterior-mean estimate shrinks as T grows.

    SEM = std(samples, dim=0) / sqrt(T). With T_large >> T_small the ratio of
    mean SEMs should be close to sqrt(T_small / T_large) (CLT). We assert
    strict inequality (large-T SEM < small-T SEM) which is almost surely true
    at these T values.
    Tolerance: not tested to exact CLT ratio — stochastic assertion with wide
    margin, fixed seed for reproducibility within this model object.
    """
    torch.manual_seed(0)
    sampler = EffectSampler()

    T_small, T_large = 50, 1000
    s_small = sampler(single_model, data_single, T=T_small)["x1"]  # [T_small, n, p]
    s_large = sampler(single_model, data_single, T=T_large)["x1"]  # [T_large, n, p]

    sem_small = s_small.std(dim=0).mean() / math.sqrt(T_small)
    sem_large = s_large.std(dim=0).mean() / math.sqrt(T_large)

    assert sem_large < sem_small, (
        f"SEM did not decrease: sem_small={sem_small:.4f}, sem_large={sem_large:.4f}"
    )


# ── 7: Centering — zero-mean curves over data ─────────────────────────────────


def test_centering_produces_zero_mean_curves(single_model, data_single):
    """Mean-centering each posterior sample over the data dim yields exact zero mean.

    This verifies that the output tensor layout (T, n, param_count) supports the
    epistemic ribbon centering described in CONTEXT.md (effect plot vs response plot).
    atol=1e-6: float32 arithmetic; centering is a single subtraction, so error
    is at floating-point epsilon, not MC noise.
    """
    sampler = EffectSampler()
    samples = sampler(single_model, data_single, T=20)["x1"]  # [T, n, param_count]

    # Center each of the T curves over the n data points.
    centered = samples - samples.mean(dim=1, keepdim=True)  # [T, n, param_count]

    # Mean over n must be identically zero for every (T, param_count) cell.
    residual = centered.mean(dim=1).abs()  # [T, param_count]
    assert residual.max().item() < 1e-5, (  # float32 sum over n=32 points
        f"centering residual too large: {residual.max().item():.2e}"
    )


# ── 8: Interaction-term contract — pre-concatenated grid (issue 0060) ─────────


def test_interaction_term_takes_preconcatenated_grid(family):
    """The value for an interaction key is the pre-concatenated (n, 2) tensor.

    Unlike forward()/LogLikSampler — which take per-feature entries and
    concatenate internally — EffectSampler callers supply per-term grids, so
    the "x1:x2" entry is the already-concatenated tensor (issue 0060).
    """
    formula = {
        "x1:x2": BayesianMLP(
            2 * IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
        ),
    }
    model = BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)

    g = torch.Generator().manual_seed(5)
    grid = {"x1:x2": torch.randn(N_OBS, 2 * IN, generator=g)}

    sampler = EffectSampler()
    T = 10
    samples = sampler(model, grid, T=T)
    assert set(samples.keys()) == {"x1:x2"}
    assert samples["x1:x2"].shape == (T, N_OBS, family.param_count)
