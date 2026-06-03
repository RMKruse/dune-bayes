"""Tests for the formula-string parser — interaction terms (issue 0017 / GitHub #41).

Boundary tests: joint-net parsing, combined-key contract, mixed formulas,
kwarg forwarding, and end-to-end training. Asserts external behavior only.
"""

import pytest

from neural_bamlss.families import NormalFamily
from neural_bamlss.formula import build_formula, parse_formula
from neural_bamlss.shapes import BayesianMLP, NeuralLinearMLP

# ── tracer bullet: single interaction term ────────────────────────────────────


def test_interaction_term_builds_joint_net_with_two_inputs():
    parsed = parse_formula("y ~ BayesianMLP(x1):BayesianMLP(x2)")
    formula = build_formula(parsed, family=NormalFamily())

    assert set(formula.keys()) == {"x1:x2"}
    assert isinstance(formula["x1:x2"], BayesianMLP)
    # Joint net: both features concatenated → in_features == 2.
    assert formula["x1:x2"].in_features == 2
    assert formula["x1:x2"].param_count == NormalFamily.param_count


# ── shape name from first factor (ADR-0005) ───────────────────────────────────


def test_interaction_uses_shape_from_first_factor():
    # First factor is BayesianMLP; second is NeuralLinearMLP — joint net must
    # be BayesianMLP (first-factor-wins, ADR-0005).
    parsed = parse_formula("y ~ BayesianMLP(x1):NeuralLinearMLP(x2)")
    formula = build_formula(parsed, family=NormalFamily())

    assert "x1:x2" in formula
    assert isinstance(formula["x1:x2"], BayesianMLP)


# ── mixed additive + interaction formula ──────────────────────────────────────


def test_mixed_formula_builds_additive_and_interaction_entries():
    parsed = parse_formula(
        "y ~ BayesianMLP(x1) + BayesianMLP(x2):BayesianMLP(x3)"
    )
    formula = build_formula(parsed, family=NormalFamily())

    assert set(formula.keys()) == {"x1", "x2:x3"}
    assert isinstance(formula["x1"], BayesianMLP)
    assert isinstance(formula["x2:x3"], BayesianMLP)
    # Additive term: single input; interaction term: two inputs concatenated.
    assert formula["x1"].in_features == 1
    assert formula["x2:x3"].in_features == 2


# ── per-term kwargs forwarded from the first factor ───────────────────────────


def test_interaction_kwargs_from_first_factor_are_forwarded():
    parsed = parse_formula(
        "y ~ BayesianMLP(x1, prior_scale=0.5, hidden_dims=(16,))"
        ":BayesianMLP(x2)"
    )
    formula = build_formula(parsed, family=NormalFamily())

    assert formula["x1:x2"].prior_scale == 0.5
    assert formula["x1:x2"].hidden_dims == [16]


# ── end-to-end: from_formula + elbo_loss with interaction term ────────────────


def test_mixed_interaction_formula_trains_end_to_end():
    import torch

    from neural_bamlss.model import BayesianNAMLSS

    torch.manual_seed(0)
    X = {
        "x1": torch.randn(12, 1),
        "x2": torch.randn(12, 1),
        "x3": torch.randn(12, 1),
    }
    y = torch.randn(12)

    model = BayesianNAMLSS.from_formula(
        "y ~ BayesianMLP(x1) + BayesianMLP(x2):BayesianMLP(x3)",
        family=NormalFamily(),
        n_obs=12,
    )
    # from_formula must capture the response name.
    assert model.response == "y"
    # Formula dict has additive entry for x1 and interaction entry for x2:x3.
    assert set(model.nets.keys()) == {"x1", "x2:x3"}

    loss = model.Loss(X, y)
    assert loss.isfinite(), f"ELBO loss is not finite: {loss}"
