"""neural-bamlss: Bayesian additive models for location, scale, and shape.

Headline API re-exported at the top level so ``import neural_bamlss as nb``
gives ``nb.BayesianNAMLSS``, ``nb.DataModule``, the families, and the
plotting / comparison functions without memorizing the submodule layout.
Subpackages (``nb.layers``, ``nb.shapes``, ``nb.sampling``, …) remain
importable for the lower-level pieces.
"""

from neural_bamlss.compare import (
    compare,  # shadows the `compare` subpackage attribute on purpose:
    # `nb.compare([m1, m2])` is the headline call (cf. `az.compare`);
    # `from neural_bamlss.compare import waic` still resolves the module.
    elbo,
    loo,
    to_inference_data,
    waic,
)
from neural_bamlss.data import DataModule
from neural_bamlss.families import (
    BaseFamily,
    GammaFamily,
    NormalFamily,
    StudentTFamily,
)
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.plotting import (
    plot_dist,
    plot_effect_ribbon,
    plot_interaction_surface,
    predictive_quantiles,
    ribbon_quantiles,
    surface_stats,
)
from neural_bamlss.priors import PriorScale
from neural_bamlss.utils import seed_everything

__version__ = "0.1.0.dev0"

__all__ = [
    "BaseFamily",
    "BayesianNAMLSS",
    "DataModule",
    "GammaFamily",
    "NormalFamily",
    "PriorScale",
    "StudentTFamily",
    "__version__",
    "compare",
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
    "waic",
]
