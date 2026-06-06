"""Bayesian and deterministic shape functions (issue 0002+ / GitHub #3+)."""

from dune_bayes.shapes.bayesian_mlp import BayesianMLP
from dune_bayes.shapes.deterministic_mlp import DeterministicMLP
from dune_bayes.shapes.deterministic_resnet import DeterministicResNet
from dune_bayes.shapes.neural_linear_mlp import NeuralLinearMLP
from dune_bayes.shapes.registry import ShapeFunctionRegistry

ShapeFunctionRegistry.register("BayesianMLP", BayesianMLP)
ShapeFunctionRegistry.register("NeuralLinearMLP", NeuralLinearMLP)
ShapeFunctionRegistry.register("MLP", DeterministicMLP)
ShapeFunctionRegistry.register("ResNet", DeterministicResNet)

__all__ = [
    "BayesianMLP",
    "DeterministicMLP",
    "DeterministicResNet",
    "NeuralLinearMLP",
    "ShapeFunctionRegistry",
]
