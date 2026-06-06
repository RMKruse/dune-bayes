"""Tests for the formula-string parser — additive terms (issue 0016 / GitHub #37).

Boundary tests per CLAUDE.md: term parsing, kwarg forwarding, response-name
capture, and the unknown-name error. Asserts external behavior only —
parsed structure, built dict contents, public constructor attributes.
"""

import pytest

from neural_bamlss.families import NormalFamily
from neural_bamlss.formula import build_formula, parse_formula
from neural_bamlss.shapes import BayesianMLP, NeuralLinearMLP

# ── single-term parse: response capture + registry resolution ─────────────────


def test_single_term_captures_response_and_resolves_via_registry():
    parsed = parse_formula("y ~ BayesianMLP(x1)")
    assert parsed.response == "y"

    formula = build_formula(parsed, family=NormalFamily())
    assert set(formula.keys()) == {"x1"}
    assert isinstance(formula["x1"], BayesianMLP)
    # Additive single-input term: one feature in, family.param_count out.
    assert formula["x1"].in_features == 1
    assert formula["x1"].param_count == NormalFamily.param_count


# ── multiple +-separated terms ────────────────────────────────────────────────


def test_plus_separated_terms_build_one_entry_per_feature():
    parsed = parse_formula("y ~ BayesianMLP(x1) + NeuralLinearMLP(x2)")
    formula = build_formula(parsed, family=NormalFamily())

    assert set(formula.keys()) == {"x1", "x2"}
    assert isinstance(formula["x1"], BayesianMLP)
    assert isinstance(formula["x2"], NeuralLinearMLP)


# ── per-term kwargs forwarded to the constructor ──────────────────────────────


def test_term_kwargs_are_forwarded_to_shape_constructor():
    parsed = parse_formula(
        "y ~ BayesianMLP(x1, prior_scale=0.5, hidden_dims=(32, 32))"
        " + NeuralLinearMLP(x2, hidden_dims=(16,))"
    )
    formula = build_formula(parsed, family=NormalFamily())

    # Asserted via public constructor attributes, not parser internals.
    assert formula["x1"].prior_scale == 0.5
    assert formula["x1"].hidden_dims == [32, 32]
    assert formula["x2"].hidden_dims == [16]


# ── prior tiers reachable from the formula (ADR-0002, issue #73) ──────────────


def test_prior_kwarg_builds_handle_and_auto_wires_kl_divisor():
    """ADR-0002's formula surface: the prior tier is a literal term kwarg.

    The n_obs auto-wiring must reach the handle too — the hyperprior KL is
    part of the same KL/N objective (ADR-0001).
    """
    parsed = parse_formula("y ~ BayesianMLP(x1, prior='empirical_bayes')")
    formula = build_formula(parsed, family=NormalFamily(), n_obs=500)

    handle = formula["x1"].prior_scale_handle
    assert handle is not None
    assert handle.mode == "empirical_bayes"
    assert handle.kl_divisor == 500.0


def test_prior_dict_literal_forwards_hyperprior_config():
    """A dict literal selects the hierarchical tier with its hyperprior knobs."""
    parsed = parse_formula(
        "y ~ NeuralLinearMLP(x1, prior={'mode': 'hierarchical', 'tau': 2.0})"
    )
    formula = build_formula(parsed, family=NormalFamily())

    handle = formula["x1"].prior_scale_handle
    assert handle.mode == "hierarchical"
    assert handle.hyperprior == "half_cauchy"
    assert handle.tau == 2.0


# ── unknown shape-function name → clear, actionable error ─────────────────────


def test_unknown_shape_name_raises_with_registered_alternatives():
    parsed = parse_formula("y ~ BogusNet(x1)")
    with pytest.raises(ValueError, match="BogusNet") as excinfo:
        build_formula(parsed, family=NormalFamily())
    # Actionable: the error points at the registry's actual entries.
    assert "BayesianMLP" in str(excinfo.value)
    assert "NeuralLinearMLP" in str(excinfo.value)


# ── duplicate feature names → error (no silent dict-overwrite) ────────────────


def test_duplicate_feature_across_terms_raises():
    # The formula dict is keyed by feature name; a second term on the same
    # feature would silently drop the first. Raise instead (issue 0017 may
    # revisit multi-term-per-feature).
    with pytest.raises(ValueError, match="x1"):
        parse_formula("y ~ BayesianMLP(x1) + NeuralLinearMLP(x1)")
