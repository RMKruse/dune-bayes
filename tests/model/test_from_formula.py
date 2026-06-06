"""Tests for BayesianNAMLSS.from_formula (issue 0016 / GitHub #37).

Boundary behavior: a model constructed from a formula string is a regular
BayesianNAMLSS — response name captured, registry-resolved shape functions,
end-to-end trainable, deterministic terms contributing zero KL.
"""

import pytest
import torch
import torch.nn as nn

from dune_bayes.families import NormalFamily
from dune_bayes.layers import collect_kl
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.shapes import BayesianMLP, NeuralLinearMLP, ShapeFunctionRegistry

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 64

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def X_y():
    g = torch.Generator().manual_seed(0)
    X = {
        "x1": torch.randn(N_OBS, 1, generator=g),
        "x2": torch.randn(N_OBS, 1, generator=g),
    }
    y = (
        2.0 * X["x1"].squeeze(-1)
        - 1.0 * X["x2"].squeeze(-1)
        + 0.1 * torch.randn(N_OBS, generator=g)
    )
    return X, y


# ── construction from a formula string ────────────────────────────────────────


def test_from_formula_builds_model_with_response_and_terms():
    model = BayesianNAMLSS.from_formula(
        "y ~ BayesianMLP(x1, hidden_dims=(8,)) + NeuralLinearMLP(x2)",
        family=NormalFamily(),
    )
    assert model.response == "y"
    assert set(model.feature_names) == {"x1", "x2"}
    assert isinstance(model.nets["x1"], BayesianMLP)
    assert isinstance(model.nets["x2"], NeuralLinearMLP)


def test_from_formula_model_trains_end_to_end(X_y):
    X, y = X_y
    torch.manual_seed(42)
    model = BayesianNAMLSS.from_formula(
        "y ~ BayesianMLP(x1, hidden_dims=(8,)) + NeuralLinearMLP(x2, hidden_dims=(8,))",
        family=NormalFamily(),
        n_obs=N_OBS,
    )
    history = model.fit(X, y, epochs=60, lr=1e-2)

    assert all(torch.isfinite(torch.tensor(history["loss"])))
    # Mean over first/last 5 epochs smooths single-pass MC noise in the ELBO;
    # 60 epochs at lr=1e-2 reliably cuts the loss on this near-linear toy.
    first = sum(history["loss"][:5]) / 5
    last = sum(history["loss"][-5:]) / 5
    assert last < first


# ── mixed Bayesian + deterministic terms ──────────────────────────────────────


class _PlainLinear(nn.Module):
    """Deterministic shape function: a single nn.Linear, zero KL by design."""

    def __init__(self, in_features: int, param_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, param_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def plain_linear_registered():
    # register() is the public extension point; the unique test-only name
    # cannot collide with real entries, so leaking it across tests is benign.
    ShapeFunctionRegistry.register("_TestPlainLinear", _PlainLinear)
    return "_TestPlainLinear"


def test_mixed_bayesian_and_deterministic_terms_train(plain_linear_registered, X_y):
    X, y = X_y
    torch.manual_seed(42)
    model = BayesianNAMLSS.from_formula(
        f"y ~ BayesianMLP(x1, hidden_dims=(8,)) + {plain_linear_registered}(x2)",
        family=NormalFamily(),
        n_obs=N_OBS,
    )
    history = model.fit(X, y, epochs=5, lr=1e-2)
    assert all(torch.isfinite(torch.tensor(history["loss"])))

    # The Bayesian term contributes KL; KL is never silently dropped.
    model(X)
    assert collect_kl(model).item() > 0.0


def test_deterministic_only_formula_terms_contribute_zero_kl(
    plain_linear_registered, X_y
):
    X, _ = X_y
    torch.manual_seed(42)
    model = BayesianNAMLSS.from_formula(
        f"y ~ {plain_linear_registered}(x1) + {plain_linear_registered}(x2)",
        family=NormalFamily(),
        n_obs=N_OBS,
    )
    model(X)
    # Deterministic terms are degenerate zero-variance contributors
    # (CONTEXT.md): every bit of model KL is the intercept's — exactly, not
    # approximately (same tensor in the walk).
    assert collect_kl(model).item() == model.intercept.kl.item()
    assert model.intercept.kl.item() > 0.0


def test_from_formula_point_intercept_gives_zero_kl(plain_linear_registered, X_y):
    X, _ = X_y
    torch.manual_seed(42)
    model = BayesianNAMLSS.from_formula(
        f"y ~ {plain_linear_registered}(x1) + {plain_linear_registered}(x2)",
        family=NormalFamily(),
        n_obs=N_OBS,
        intercept_mode="point",
    )
    model(X)
    # Point intercept + deterministic terms: the fully-deterministic model.
    assert collect_kl(model).item() == 0.0
