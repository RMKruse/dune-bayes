"""Tests for draw_predictive + pointwise_log_lik (issue 0007 / GitHub #8, #68).

Four reference-test archetypes (CLAUDE.md):
  - Shape:         summed_samples (T, n, param_count), pointwise_log_lik (T, n).
  - Behavior:      loglik is float64, predictive is MixtureSameFamily, T_EVAL=1000;
                   sample_posterior_predictive never evaluates log_prob (#68).
  - Pure function: model state unchanged after sampling.
  - Reference:     degenerate single-draw loglik == direct family log_prob;
                   law of total variance holds for the predictive.
"""

import pytest
import torch

from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.sampling import T_EVAL, draw_predictive, pointwise_log_lik
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
    """summed_samples is (T, n, param_count); pointwise_log_lik is (T, n)."""
    T = 10
    draws = draw_predictive(model, data_x, T=T)
    ll = pointwise_log_lik(model, draws.summed_samples, data_y)
    assert draws.summed_samples.shape == (T, N_OBS, family.param_count)
    assert ll.shape == (T, N_OBS)


# ── 2: pointwise_log_lik is float64 ───────────────────────────────────────────


def test_loglik_is_float64(model, data_x, data_y):
    """pointwise_log_lik must be float64 for WAIC/LOO accumulation (numerical rule)."""
    draws = draw_predictive(model, data_x, T=5)
    ll = pointwise_log_lik(model, draws.summed_samples, data_y)
    assert ll.dtype == torch.float64


# ── 3: Predictive is MixtureSameFamily with correct batch shape ───────────────


def test_predictive_is_mixture_same_family(model, data_x):
    """predictive is a torch.distributions.MixtureSameFamily."""
    draws = draw_predictive(model, data_x, T=10)
    assert isinstance(draws.predictive, torch.distributions.MixtureSameFamily)


def test_predictive_batch_shape(model, data_x):
    """predictive.batch_shape == (n,) — one predictive per observation."""
    draws = draw_predictive(model, data_x, T=10)
    assert draws.predictive.batch_shape == (N_OBS,)


# ── 4: Degenerate single-draw loglik matches direct family log_prob ───────────


def test_single_draw_loglik_matches_direct(model, data_x, data_y, family):
    """With T=1, pointwise_log_lik[0] == family(summed_samples[0]).log_prob(y).

    Tolerance: float32 forward pass cast to float64 vs a float32 log_prob;
    difference is at float32 epsilon (≈6e-8 relative) — atol=1e-5 is conservative.
    """
    torch.manual_seed(42)
    draws = draw_predictive(model, data_x, T=1)
    ll = pointwise_log_lik(model, draws.summed_samples, data_y)

    # Recompute directly from the stored summed_samples (T=0 draw).
    with torch.no_grad():
        dist_direct = family(draws.summed_samples[0])  # family uses float32 params
        loglik_direct = dist_direct.log_prob(data_y).to(torch.float64)

    assert torch.allclose(
        ll[0],
        loglik_direct,
        atol=1e-5,
    ), "single-draw loglik deviates from direct family log_prob"


# ── 5: Pure function — no model-state mutation ────────────────────────────────


def test_pure_function_preserves_training_mode(model, data_x):
    """Training mode is restored after sampling, regardless of initial mode."""
    model.train()
    draw_predictive(model, data_x, T=5)
    assert model.training, "training mode not restored after sampling"

    model.eval()
    draw_predictive(model, data_x, T=5)
    assert not model.training, "eval mode not restored after sampling"


def test_pure_function_preserves_parameters(model, data_x, data_y):
    """Posterior parameters (loc, rho) are identical before and after sampling."""
    params_before = {k: v.clone() for k, v in model.named_parameters()}
    draws = draw_predictive(model, data_x, T=20)
    pointwise_log_lik(model, draws.summed_samples, data_y)
    for name, val in model.named_parameters():
        assert torch.equal(params_before[name], val), f"parameter {name!r} mutated"


# ── 6: Law of total variance — Var[Y] = aleatoric + epistemic ─────────────────


def test_law_of_total_variance(model, data_x):
    """predictive.variance ≈ E[Var[Y|θ]] + Var[E[Y|θ]] across T weight draws.

    Law of total variance: total = aleatoric + epistemic.
    We compute both sides from summed_samples and check they match the mixture
    variance to within float32 MC noise (T=500, atol=0.05).
    Tolerance: MC estimate of aleatoric+epistemic for T=500 is noisy at ~1/√T;
    atol=0.05 is conservative given typical Normal variance magnitude.
    """
    torch.manual_seed(0)
    T = 500
    draws = draw_predictive(model, data_x, T=T)

    family = model.family
    # Component means and variances over the T draws.
    # summed_samples: (T, n, param_count)
    with torch.no_grad():
        comp_means = torch.stack(
            [family(draws.summed_samples[t]).mean for t in range(T)], dim=0
        )  # (T, n)
        comp_vars = torch.stack(
            [family(draws.summed_samples[t]).variance for t in range(T)], dim=0
        )  # (T, n)

    # Law of total variance decomposition.
    aleatoric = comp_vars.mean(dim=0)  # E[Var[Y|θ]], shape (n,)
    epistemic = comp_means.var(dim=0)  # Var[E[Y|θ]], shape (n,)
    loto_var = aleatoric + epistemic  # (n,)

    mixture_var = draws.predictive.variance  # (n,)

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


class _LogProbTrap(NormalFamily):
    """Normal family whose log_prob raises — sentinel for the dummy-y hack (#68).

    The pre-#68 sample_posterior_predictive scored a fabricated y=0 through
    the full log-likelihood path; drawing the predictive must never evaluate
    log_prob, so a distribution that raises on it proves the path is gone.
    """

    def __call__(self, params):
        dist = super().__call__(params)

        def _raise(value):
            raise AssertionError(
                "sample_posterior_predictive evaluated log_prob (dummy-y hack)"
            )

        dist.log_prob = _raise
        return dist


def test_sample_posterior_predictive_never_scores(data_x):
    """Drawing the predictive evaluates no log_prob (GitHub #68).

    The old implementation fabricated y_dummy = zeros(n) and ran the scoring
    path — out of support for e.g. the Gamma family, surviving only because
    validate_args=False. A trap family that raises on log_prob would have
    tripped it; the split draw_predictive must pass.
    """
    family = _LogProbTrap()
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    model = BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)
    predictive = model.sample_posterior_predictive(data_x, T=5)
    assert isinstance(predictive, torch.distributions.MixtureSameFamily)


# ── 8: T_EVAL module constant ─────────────────────────────────────────────────


def test_t_eval_constant():
    """T_EVAL is 1000 for information-criterion runs (CONTEXT.md MC sample counts)."""
    assert T_EVAL == 1000


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
    T = 5
    draws = draw_predictive(interaction_model, interaction_x, T=T)
    ll = pointwise_log_lik(interaction_model, draws.summed_samples, data_y)
    assert draws.summed_samples.shape == (T, N_OBS, family.param_count)
    assert ll.shape == (T, N_OBS)
    assert torch.isfinite(ll).all()


def test_interaction_model_posterior_predictive(interaction_model, interaction_x):
    """sample_posterior_predictive works for interaction models (Goal 3 path)."""
    predictive = interaction_model.sample_posterior_predictive(interaction_x, T=10)
    assert isinstance(predictive, torch.distributions.MixtureSameFamily)
    assert predictive.batch_shape == (N_OBS,)
