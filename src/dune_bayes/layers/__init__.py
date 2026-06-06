"""Variational layer primitives (ADR-0004)."""

from dune_bayes.layers.base import (
    VariationalLayer,
    collect_kl,
    gaussian_kl,
    set_kl_beta,
)
from dune_bayes.layers.bayesian_embedding import BayesianEmbedding
from dune_bayes.layers.bayesian_intercept import BayesianIntercept
from dune_bayes.layers.variational_dense import VariationalDense

__all__ = [
    "BayesianEmbedding",
    "BayesianIntercept",
    "VariationalDense",
    "VariationalLayer",
    "collect_kl",
    "gaussian_kl",
    "set_kl_beta",
]
