"""Fully-deterministic MLP shape function (issue 0020 / GitHub #39).

DeterministicMLP: every layer is a plain nn.Linear — no VariationalDense.
Contributes zero KL inside BayesianNAMLSS; collect_kl sees nothing from it.
Useful as a Bayes-vs-deterministic baseline in WAIC/LOO comparisons (user
stories 21, 22).

Design mirrors BayesianMLP but replaces VariationalDense with nn.Linear
throughout. Output layer has no bias (intercept handled by BayesianIntercept).
"""

import torch
import torch.nn as nn

from dune_bayes.utils import resolve_activation

_DEFAULT_HIDDEN_DIMS: list[int] = [64, 64]


class DeterministicMLP(nn.Module):
    """Fully-deterministic MLP shape function for dune-bayes.

    Maps a feature tensor (batch, in_features) to (batch, param_count).
    All layers are plain nn.Linear — no VariationalDense — so collect_kl()
    returns 0.0 for this module. Output is identical for the same input
    (no stochasticity).

    Args:
        in_features: Input dimension.
        param_count: Output dimension — equals the family's param_count.
        hidden_dims: Widths of the intermediate nn.Linear hidden layers.
            Defaults to [64, 64].
        activation: Activation between hidden layers. One of
            {None, "linear", "relu", "tanh"}.
    """

    def __init__(
        self,
        in_features: int,
        param_count: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.param_count = int(param_count)
        if hidden_dims is not None:
            self.hidden_dims = list(hidden_dims)
        else:
            self.hidden_dims = list(_DEFAULT_HIDDEN_DIMS)
        # Resolved once here (validates the name); config keeps the string.
        self._act = resolve_activation(activation)
        self.activation = activation

        layers: list[nn.Module] = []
        prev = self.in_features
        for width in self.hidden_dims:
            layers.append(nn.Linear(prev, width))
            prev = width
        # Output layer: no bias (intercept handled by BayesianIntercept).
        layers.append(nn.Linear(prev, self.param_count, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, in_features) → (batch, param_count).

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, param_count).
        """
        # All but the last layer get activation; output layer is linear.
        for layer in self.layers[:-1]:
            x = layer(x)
            if self._act is not None:
                x = self._act(x)
        # nn.Module.__call__ is untyped in torch's stubs; anchor the return type.
        out: torch.Tensor = self.layers[-1](x)
        return out

    # ── serialization ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints, strings, lists only)."""
        return {
            "in_features": self.in_features,
            "param_count": self.param_count,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config: dict) -> "DeterministicMLP":
        """Reconstruct a DeterministicMLP from a get_config() dict.

        Args:
            config: Dict as returned by get_config().

        Returns:
            New DeterministicMLP with the same architecture.
        """
        return cls(**config)
