"""Top-level package API: ``import dune_bayes as db`` exposes the headline names.

CLAUDE.md promises the ``db.``-style import surface; this pins it so a
re-export can't silently vanish from ``__init__``.
"""

import importlib

import pytest

import dune_bayes as db


@pytest.mark.parametrize("name", sorted(db.__all__))
def test_all_names_resolve(name: str) -> None:
    # Every advertised name must be an actual attribute, not a stale entry.
    assert getattr(db, name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "BayesianNAMLSS",
        "DataModule",
        "PriorScale",
        "seed_everything",
        "BaseFamily",
        "NormalFamily",
        "StudentTFamily",
        "GammaFamily",
        "waic",
        "loo",
        "compare",
        "elbo",
        "to_inference_data",
        "plot_effect_ribbon",
        "plot_dist",
        "plot_interaction_surface",
        "ribbon_quantiles",
        "predictive_quantiles",
        "surface_stats",
    ],
)
def test_headline_api_in_all(name: str) -> None:
    assert name in db.__all__


def test_compare_function_shadows_subpackage() -> None:
    # `db.compare` is deliberately the function (the headline call, like
    # `az.compare`), while the subpackage stays importable for its other names.
    assert callable(db.compare)
    comparison_mod = importlib.import_module("dune_bayes.compare")
    assert comparison_mod.waic is db.waic


@pytest.mark.parametrize(
    "subpackage",
    ["layers", "shapes", "sampling", "formula", "families", "data", "priors"],
)
def test_subpackages_still_importable(subpackage: str) -> None:
    importlib.import_module(f"dune_bayes.{subpackage}")
