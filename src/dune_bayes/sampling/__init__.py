"""Posterior sampling workhorses (issues 0005, 0007 / GitHub #6, #8, #68).

sample_effects:    per-feature contribution samples → effect ribbons (issue 0005).
draw_predictive:   summed-predictor draws + MixtureSameFamily predictive (issue 0007).
pointwise_log_lik: scores y against drawn samples → WAIC/LOO (issues 0007, #68).
"""

from dune_bayes.sampling.effect_sampler import T_PREDICT, sample_effects
from dune_bayes.sampling.log_lik_sampler import (
    T_EVAL,
    PredictiveDraws,
    draw_predictive,
    pointwise_log_lik,
)

__all__ = [
    "sample_effects",
    "T_PREDICT",
    "draw_predictive",
    "pointwise_log_lik",
    "PredictiveDraws",
    "T_EVAL",
]
