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
        "BetaFamily",
        "NormalFamily",
        "StudentTFamily",
        "GammaFamily",
        "JohnsonSUFamily",
        "NegativeBinomialFamily",
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


def test_no_flipout_naming_anywhere_in_package() -> None:
    """The misnomer "flipout" must not survive in package source (ADR-0007).

    The estimator is local reparameterization (Kingma et al., 2015), not
    flipout (Wen et al., 2018); the old name would be a methods-section
    error. Enforced mechanically, like the no-TF rule (CLAUDE.md).
    """
    from pathlib import Path

    src_root = Path(db.__file__).parent
    offenders = [
        str(path.relative_to(src_root))
        for path in sorted(src_root.rglob("*.py"))
        if "flipout" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == [], f'"flipout" still present in: {offenders}'
