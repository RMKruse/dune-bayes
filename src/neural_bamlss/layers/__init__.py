"""Variational layer primitives (ADR-0004)."""

from neural_bamlss.layers.variational_dense import (
    VariationalDense,
    collect_kl,
    set_kl_beta,
)

__all__ = ["VariationalDense", "collect_kl", "set_kl_beta"]
