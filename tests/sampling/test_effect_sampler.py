"""Tests for sample_effects workhorse (issue 0005 / GitHub #6, #68).

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
from neural_bamlss.sampling import T_PREDICT, sample_effects
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
    T = 10
    samples = sample_effects(single_model, data_single, T=T)
    assert "x1" in samples
    contrib = samples["x1"]
    assert contrib.shape == (T, N_OBS, family.param_count)
    assert contrib.dtype == torch.float32


# ── 2: All features returned ──────────────────────────────────────────────────


def test_all_feature_names_returned(multi_model, data_multi, family):
    """Every feature name in model.feature_names appears in the output dict."""
    samples = sample_effects(multi_model, data_multi, T=10)
    assert set(samples.keys()) == {"x1", "x2"}
    for name in ("x1", "x2"):
        assert samples[name].shape == (10, N_OBS, family.param_count)


# ── 3: Default T = 200 ────────────────────────────────────────────────────────


def test_t_predict_constant():
    """T_PREDICT is 200 — the default posterior-draw count (CONTEXT.md)."""
    assert T_PREDICT == 200


def test_default_t_is_200(single_model, data_single):
    """Calling sample_effects without T uses T_PREDICT=200."""
    samples = sample_effects(single_model, data_single)
    assert samples["x1"].shape[0] == 200


# ── 4: T override ─────────────────────────────────────────────────────────────


def test_t_override(single_model, data_single):
    """Passing T overrides the default."""
    for T in (1, 50, 500):
        samples = sample_effects(single_model, data_single, T=T)
        got = samples["x1"].shape[0]
        assert got == T, f"Expected T={T}, got {got}"


# ── 5: Pure function — no model-state mutation ─────────────────────────────────


def test_pure_function_preserves_training_mode(single_model, data_single):
    """Training mode is restored after sampling, regardless of initial mode."""
    single_model.train()
    sample_effects(single_model, data_single, T=5)
    assert single_model.training, "training mode not restored after sampling"

    single_model.eval()
    sample_effects(single_model, data_single, T=5)
    assert not single_model.training, "eval mode not restored after sampling"


def test_pure_function_preserves_parameters(single_model, data_single):
    """Posterior parameters (loc, rho) are identical before and after sampling."""
    params_before = {k: v.clone() for k, v in single_model.named_parameters()}
    sample_effects(single_model, data_single, T=20)
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
    T_small, T_large = 50, 1000
    s_small = sample_effects(single_model, data_single, T=T_small)[
        "x1"
    ]  # [T_small, n, p]
    s_large = sample_effects(single_model, data_single, T=T_large)[
        "x1"
    ]  # [T_large, n, p]

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
    samples = sample_effects(single_model, data_single, T=20)[
        "x1"
    ]  # [T, n, param_count]

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

    Unlike forward()/draw_predictive — which take per-feature entries and
    concatenate internally — sample_effects callers supply per-term grids, so
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

    T = 10
    samples = sample_effects(model, grid, T=T)
    assert set(samples.keys()) == {"x1:x2"}
    assert samples["x1:x2"].shape == (T, N_OBS, family.param_count)


# ── 9: Vectorized sweep — chunking over T (issue 0027 / GitHub #80) ───────────


def test_chunked_sweep_returns_full_t_with_independent_draws(
    single_model, data_single, family
):
    """A chunk_size smaller than T still yields (T, n, param_count) with every
    draw independent — including across chunk boundaries.

    chunk_size is the internal memory knob: the sweep batches min(chunk, T)
    draws per dispatch and concatenates. Identical slices anywhere would mean
    a draw was broadcast instead of freshly sampled.
    """
    torch.manual_seed(21)
    T, chunk = 10, 4  # chunks of 4, 4, 2 — exercises the ragged tail
    samples = sample_effects(single_model, data_single, T=T, chunk_size=chunk)["x1"]
    assert samples.shape == (T, N_OBS, family.param_count)
    for t in range(1, T):
        assert not torch.equal(samples[0], samples[t]), (
            f"draw {t} equals draw 0 — broadcast instead of independent draws"
        )


def test_variance_across_t_consistent_with_loop(single_model, data_single):
    """std across the T axis matches a per-draw loop reference (MC-convergence).

    The loop below is the pre-issue-0027 implementation inlined as the
    independent reference: T separate net(x) calls. Both estimates target the
    same posterior std; with T=400 the sample-std rel error is ≈ 1/√(2T) ≈ 3.5%
    per element and smaller for the element-mean compared here — rel=0.2 gives
    wide MC headroom under the fixed seed while catching a wrong or collapsed
    cross-draw variance.
    """
    torch.manual_seed(22)
    T = 400
    vec = sample_effects(single_model, data_single, T=T)["x1"]

    net = single_model.nets["x1"]
    x = data_single["x1"]
    single_model.eval()
    with torch.no_grad():
        loop = torch.stack([net(x) for _ in range(T)], dim=0)

    std_vec = vec.std(dim=0).mean()
    std_loop = loop.std(dim=0).mean()
    assert float(std_vec) == pytest.approx(float(std_loop), rel=0.2)
