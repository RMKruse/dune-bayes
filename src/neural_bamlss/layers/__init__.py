"""Variational layer primitives (ADR-0004)."""

from neural_bamlss.layers.bayesian_embedding import BayesianEmbedding
from neural_bamlss.layers.bayesian_intercept import BayesianIntercept
from neural_bamlss.layers.variational_dense import (
    VariationalDense,
    collect_kl,
    set_kl_beta,
)

__all__ = [
    "BayesianEmbedding",
    "BayesianIntercept",
    "VariationalDense",
    "collect_kl",
    "set_kl_beta",
]
