"""ADR-0007 boundary tests — posterior sampling uses coherent global weight draws.

(issue #85)  Local reparameterization is a training-only variance-reduction
estimator: per-row independent noise destroys coherent function draws and
silently corrupts the MixtureSameFamily aleatoric/epistemic decomposition.
These tests pin the boundary:
  - Tripwire: sample_effects / draw_predictive never route through the
    local-reparam path, even on a model built with local_reparam=True.
  - Behavioral coherence: on a linear shape function, f(x)/x is constant
    across a grid within one draw and varies across draws.
"""

import pytest
import torch

from dune_bayes.families import NormalFamily
from dune_bayes.layers import VariationalDense
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.sampling import draw_predictive, sample_effects
from dune_bayes.shapes import BayesianMLP

N_OBS = 16
IN = 1
T = 4

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily()


@pytest.fixture
def local_reparam_model(family):
    torch.manual_seed(0)
    formula = {
        "x1": BayesianMLP(
            IN,
            family.param_count,
            hidden_dims=[8],
            kl_divisor=N_OBS,
            local_reparam=True,
        ),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(42)
    return {"x1": torch.randn(N_OBS, IN, generator=g)}


# ── tripwire: the vanilla path is forced ──────────────────────────────────────


def test_sampling_entry_points_never_use_local_reparam(
    local_reparam_model, data, monkeypatch
):
    """sample_effects and draw_predictive force the vanilla coherent path.

    Tripwire: the local-reparam forward is patched to raise; both entry points
    must complete on a model built with local_reparam=True, proving posterior
    sampling can never route through per-row noise (ADR-0007).
    pointwise_log_lik does no forward pass, so these two cover every
    posterior-sampling entry point.
    """

    def _trip(self, x, kernel_scale):
        raise AssertionError("local-reparam path reached during posterior sampling")

    monkeypatch.setattr(VariationalDense, "_forward_local_reparam", _trip)

    effects = sample_effects(local_reparam_model, data, T=T)
    assert effects["x1"].shape == (T, N_OBS, local_reparam_model.family.param_count)

    draws = draw_predictive(local_reparam_model, data, T=T)
    assert draws.summed_samples.shape == (
        T,
        N_OBS,
        local_reparam_model.family.param_count,
    )


# ── behavioral coherence: one weight realization per draw ─────────────────────


def test_linear_effect_draws_are_coherent_within_and_vary_across(family):
    """On a linear shape function, f(x)/x is constant within one draw.

    A coherent posterior draw is ONE weight realization w ~ q(w) evaluated at
    every grid point, so f(x) = x·w gives f(x)/x == w across the whole grid
    within a draw (ADR-0007).  Under local reparameterization each row would
    carry an independent implicit weight draw and the ratio would scatter.
    Across draws the ratio must differ — otherwise one draw was broadcast.
    """
    torch.manual_seed(1)
    # hidden_dims=[] → a single VariationalDense(1, P, use_bias=False, no
    # activation): exactly linear, f(x) = x · w.
    linear_model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(
                IN,
                family.param_count,
                hidden_dims=[],
                kl_divisor=N_OBS,
                local_reparam=True,
            ),
        },
        family=family,
        n_obs=N_OBS,
    )
    n_grid = 8
    # Grid bounded away from 0 so the ratio f(x)/x is well-conditioned.
    grid = torch.linspace(0.5, 2.0, n_grid).unsqueeze(1)

    effects = sample_effects(linear_model, {"x1": grid}, T=T)
    ratio = effects["x1"] / grid.unsqueeze(0)  # (T, n_grid, param_count)

    # Within a draw: ratio is the weight itself up to float32 rounding from
    # (x·w)/x — pure float error, no MC noise, so abs=1e-5 (~100× eps) is
    # tight enough to catch per-row noise (which scatters at the posterior
    # scale, softplus(-3) ≈ 0.05).
    spread_within = (ratio - ratio.mean(dim=1, keepdim=True)).abs().max()
    assert float(spread_within) < 1e-5, (
        f"f(x)/x varies within a draw (max spread {float(spread_within):.2e}) — "
        "the draw is not one coherent weight realization"
    )

    # Across draws: independent weight realizations must differ. With the
    # posterior scale ≈ 0.05, two draws coinciding to float32 equality has
    # probability ~0.
    per_draw_w = ratio.mean(dim=1)  # (T, param_count)
    for t in range(1, T):
        assert not torch.equal(per_draw_w[0], per_draw_w[t]), (
            f"draw {t} equals draw 0 — one weight draw was broadcast across T"
        )
