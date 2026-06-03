"""Bayesian and deterministic shape functions (issue 0002+ / GitHub #3+)."""

from neural_bamlss.shapes.bayesian_mlp import BayesianMLP
from neural_bamlss.shapes.registry import ShapeFunctionRegistry

ShapeFunctionRegistry.register("BayesianMLP", BayesianMLP)

__all__ = ["BayesianMLP", "ShapeFunctionRegistry"]
