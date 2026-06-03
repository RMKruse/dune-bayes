"""Bayesian and deterministic shape functions (issue 0002+ / GitHub #3+)."""

from neural_bamlss.shapes.bayesian_mlp import BayesianMLP
from neural_bamlss.shapes.deterministic_mlp import DeterministicMLP
from neural_bamlss.shapes.deterministic_resnet import DeterministicResNet
from neural_bamlss.shapes.neural_linear_mlp import NeuralLinearMLP
from neural_bamlss.shapes.registry import ShapeFunctionRegistry

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
