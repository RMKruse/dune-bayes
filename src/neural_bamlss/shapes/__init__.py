"""Bayesian and deterministic shape functions (issue 0002+ / GitHub #3+)."""

from neural_bamlss.shapes.bayesian_mlp import BayesianMLP
from neural_bamlss.shapes.neural_linear_mlp import NeuralLinearMLP
from neural_bamlss.shapes.registry import ShapeFunctionRegistry

ShapeFunctionRegistry.register("BayesianMLP", BayesianMLP)
ShapeFunctionRegistry.register("NeuralLinearMLP", NeuralLinearMLP)

__all__ = ["BayesianMLP", "NeuralLinearMLP", "ShapeFunctionRegistry"]
