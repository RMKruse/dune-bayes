"""ShapeFunctionRegistry — name-to-class lookup for shape functions.

Issue 0002 / GitHub #3. Reimplemented from scratch in PyTorch; not ported
from NAMpy's TF registry. Shape functions are nn.Module subclasses.
"""

import torch.nn as nn


class ShapeFunctionRegistry:
    """Registry mapping string names to shape-function classes.

    Shape functions are nn.Module subclasses that map a feature tensor
    (batch, in_features) to (batch, param_count).

    Args:
        None — the registry is class-level state.
    """

    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str, shape_cls: type[nn.Module]) -> None:
        """Register a shape-function class under a name.

        Args:
            name: The lookup key (e.g. "BayesianMLP").
            shape_cls: A nn.Module subclass implementing the shape-function
                contract (forward maps (batch, in_features) → (batch, param_count)).
        """
        cls._registry[name] = shape_cls

    @classmethod
    def get(cls, name: str) -> type[nn.Module] | None:
        """Return the registered class for *name*, or None if unknown.

        Args:
            name: Registry key.

        Returns:
            The class, or None.
        """
        return cls._registry.get(name)

    @classmethod
    def names(cls) -> list[str]:
        """Return the registered shape-function names, sorted.

        Returns:
            Sorted list of registry keys.
        """
        return sorted(cls._registry)
