"""Tests for LogLikSampler workhorse (issue 0007 / GitHub #8).

Four reference-test archetypes (CLAUDE.md):
  - Shape:         summed_samples (T, n, param_count), pointwise_loglik (T, n).
  - Behavior:      loglik is float64, predictive is MixtureSameFamily, T_eval=1000.
  - Pure function: model state unchanged after sampling.
  - Reference:     degenerate single-draw loglik == direct family log_prob;
                   law of total variance holds for the predictive.
"""

import pytest
import torch

from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.sampling import LogLikSampler
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
IN = 1

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily(validate_args=True)


@pytest.fixture
def model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def data_x():
    g = torch.Generator().manual_seed(0)
    return {"x1": torch.randn(N_OBS, IN, generator=g)}


@pytest.fixture
def data_y():
    g = torch.Generator().manual_seed(1)
    return torch.randn(N_OBS, generator=g)


# ── 1: Shape — tracer bullet ──────────────────────────────────────────────────


def test_summed_samples_and_loglik_shape(model, data_x, data_y, family):
    """summed_samples is (T, n, param_count); pointwise_loglik is (T, n)."""
    sampler = LogLikSampler()
    T = 10
    result = sampler(model, data_x, data_y, T=T)
    assert result.summed_samples.shape == (T, N_OBS, family.param_count)
    assert result.pointwise_loglik.shape == (T, N_OBS)


# ── 2: pointwise_loglik is float64 ────────────────────────────────────────────


def test_loglik_is_float64(model, data_x, data_y):
    """pointwise_loglik must be float64 for WAIC/LOO accumulation (numerical rule)."""
    sampler = LogLikSampler()
    result = sampler(model, data_x, data_y, T=5)
    assert result.pointwise_loglik.dtype == torch.float64


# ── 3: Predictive is MixtureSameFamily with correct batch shape ───────────────


def test_predictive_is_mixture_same_family(model, data_x, data_y):
    """predictive is a torch.distributions.MixtureSameFamily."""
    sampler = LogLikSampler()
    result = sampler(model, data_x, data_y, T=10)
    assert isinstance(result.predictive, torch.distributions.MixtureSameFamily)


def test_predictive_batch_shape(model, data_x, data_y):
    """predictive.batch_shape == (n,) — one predictive per observation."""
    sampler = LogLikSampler()
    result = sampler(model, data_x, data_y, T=10)
    assert result.predictive.batch_shape == (N_OBS,)


# ── 4: Degenerate single-draw loglik matches direct family log_prob ───────────


def test_single_draw_loglik_matches_direct(model, data_x, data_y, family):
    """With T=1, pointwise_loglik[0] == family(summed_samples[0]).log_prob(y).

    Tolerance: float32 forward pass cast to float64 vs a float32 log_prob;
    difference is at float32 epsilon (≈6e-8 relative) — atol=1e-5 is conservative.
    """
    sampler = LogLikSampler()
    torch.manual_seed(42)
    result = sampler(model, data_x, data_y, T=1)

    # Recompute directly from the stored summed_samples (T=0 draw).
    with torch.no_grad():
        dist_direct = family(result.summed_samples[0])  # family uses float32 params
        loglik_direct = dist_direct.log_prob(data_y).to(torch.float64)

    assert torch.allclose(
        result.pointwise_loglik[0],
        loglik_direct,
        atol=1e-5,
    ), "single-draw loglik deviates from direct family log_prob"


# ── 5: Pure function — no model-state mutation ────────────────────────────────


def test_pure_function_preserves_training_mode(model, data_x, data_y):
    """Training mode is restored after sampling, regardless of initial mode."""
    sampler = LogLikSampler()

    model.train()
    sampler(model, data_x, data_y, T=5)
    assert model.training, "training mode not restored after sampling"

    model.eval()
    sampler(model, data_x, data_y, T=5)
    assert not model.training, "eval mode not restored after sampling"


def test_pure_function_preserves_parameters(model, data_x, data_y):
    """Posterior parameters (loc, rho) are identical before and after sampling."""
    sampler = LogLikSampler()
    params_before = {k: v.clone() for k, v in model.named_parameters()}
    sampler(model, data_x, data_y, T=20)
    for name, val in model.named_parameters():
        assert torch.equal(params_before[name], val), f"parameter {name!r} mutated"


# ── 6: Law of total variance — Var[Y] = aleatoric + epistemic ─────────────────


def test_law_of_total_variance(model, data_x, data_y):
    """predictive.variance ≈ E[Var[Y|θ]] + Var[E[Y|θ]] across T weight draws.

    Law of total variance: total = aleatoric + epistemic.
    We compute both sides from summed_samples and check they match the mixture
    variance to within float32 MC noise (T=500, atol=0.05).
    Tolerance: MC estimate of aleatoric+epistemic for T=500 is noisy at ~1/√T;
    atol=0.05 is conservative given typical Normal variance magnitude.
    """
    torch.manual_seed(0)
    sampler = LogLikSampler()
    T = 500
    result = sampler(model, data_x, data_y, T=T)

    family = model.family
    # Component means and variances over the T draws.
    # summed_samples: (T, n, param_count)
    with torch.no_grad():
        comp_means = torch.stack(
            [family(result.summed_samples[t]).mean for t in range(T)], dim=0
        )  # (T, n)
        comp_vars = torch.stack(
            [family(result.summed_samples[t]).variance for t in range(T)], dim=0
        )  # (T, n)

    # Law of total variance decomposition.
    aleatoric = comp_vars.mean(dim=0)  # E[Var[Y|θ]], shape (n,)
    epistemic = comp_means.var(dim=0)  # Var[E[Y|θ]], shape (n,)
    loto_var = aleatoric + epistemic  # (n,)

    mixture_var = result.predictive.variance  # (n,)

    assert torch.allclose(
        mixture_var.float(),
        loto_var.float(),
        atol=0.05,
    ), (
        f"law of total variance violated: "
        f"max|Δ|={(mixture_var.float() - loto_var.float()).abs().max():.4f}"
    )


# ── 7: sample_posterior_predictive on BayesianNAMLSS ─────────────────────────


def test_sample_posterior_predictive_returns_mixture(model, data_x):
    """model.sample_posterior_predictive(X, T) returns a MixtureSameFamily."""
    T = 20
    predictive = model.sample_posterior_predictive(data_x, T=T)
    assert isinstance(predictive, torch.distributions.MixtureSameFamily)
    assert predictive.batch_shape == (N_OBS,)


# ── 8: T_eval class attribute ─────────────────────────────────────────────────


def test_t_eval_class_attribute():
    """LogLikSampler.T_eval is 1000 for information-criterion runs."""
    assert LogLikSampler.T_eval == 1000


# ── 9: Interaction terms — sampler accepts the same X dict as forward ─────────
# (issue 0060: predictor assembly is the model's concept; the sampler must not
# re-implement it and drift on interaction keys.)


@pytest.fixture
def interaction_model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
        "x1:x2": BayesianMLP(
            2 * IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
        ),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def interaction_x():
    g = torch.Generator().manual_seed(3)
    return {
        "x1": torch.randn(N_OBS, IN, generator=g),
        "x2": torch.randn(N_OBS, IN, generator=g),
    }


def test_interaction_model_loglik_shapes_and_finite(
    interaction_model, interaction_x, data_y, family
):
    """An "x1:x2" model works through the sampler on the same X dict as forward."""
    sampler = LogLikSampler()
    T = 5
    result = sampler(interaction_model, interaction_x, data_y, T=T)
    assert result.summed_samples.shape == (T, N_OBS, family.param_count)
    assert result.pointwise_loglik.shape == (T, N_OBS)
    assert torch.isfinite(result.pointwise_loglik).all()


def test_interaction_model_posterior_predictive(interaction_model, interaction_x):
    """sample_posterior_predictive works for interaction models (Goal 3 path)."""
    predictive = interaction_model.sample_posterior_predictive(interaction_x, T=10)
    assert isinstance(predictive, torch.distributions.MixtureSameFamily)
    assert predictive.batch_shape == (N_OBS,)
