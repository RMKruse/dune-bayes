"""DUNE — Distributional Uncertainty in Neural-additive Estimation.

Bayesian additive models for location, scale, and shape (``dune-bayes``).

Headline API re-exported at the top level so ``import dune_bayes as db``
gives ``db.BayesianNAMLSS``, ``db.DataModule``, the families, and the
plotting / comparison functions without memorizing the submodule layout.
Subpackages (``db.layers``, ``db.shapes``, ``db.sampling``, …) remain
importable for the lower-level pieces.
"""

from dune_bayes.compare import (
    compare,  # shadows the `compare` subpackage attribute on purpose:
    # `db.compare([m1, m2])` is the headline call (cf. `az.compare`);
    # `from dune_bayes.compare import waic` still resolves the module.
    elbo,
    loo,
    to_inference_data,
    waic,
)
from dune_bayes.data import DataModule
from dune_bayes.families import (
    BaseFamily,
    GammaFamily,
    NormalFamily,
    StudentTFamily,
)
from dune_bayes.metrics import VarianceDecomposition, crps, variance_decomposition
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.plotting import (
    plot_dist,
    plot_effect_ribbon,
    plot_interaction_surface,
    predictive_quantiles,
    ribbon_quantiles,
    surface_stats,
)
from dune_bayes.priors import PriorScale
from dune_bayes.utils import seed_everything

__version__ = "0.1.0.dev0"

__all__ = [
    "BaseFamily",
    "BayesianNAMLSS",
    "DataModule",
    "GammaFamily",
    "NormalFamily",
    "PriorScale",
    "StudentTFamily",
    "VarianceDecomposition",
    "__version__",
    "compare",
    "crps",
    "elbo",
    "loo",
    "plot_dist",
    "plot_effect_ribbon",
    "plot_interaction_surface",
    "predictive_quantiles",
    "ribbon_quantiles",
    "seed_everything",
    "surface_stats",
    "to_inference_data",
    "variance_decomposition",
    "waic",
]
