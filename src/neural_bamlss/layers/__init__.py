"""Variational layer primitives (ADR-0004)."""

from neural_bamlss.layers.base import (
    VariationalLayer,
    collect_kl,
    gaussian_kl,
    set_kl_beta,
)
from neural_bamlss.layers.bayesian_embedding import BayesianEmbedding
from neural_bamlss.layers.bayesian_intercept import BayesianIntercept
from neural_bamlss.layers.variational_dense import VariationalDense

__all__ = [
    "BayesianEmbedding",
    "BayesianIntercept",
    "VariationalDense",
    "VariationalLayer",
    "collect_kl",
    "gaussian_kl",
    "set_kl_beta",
]
