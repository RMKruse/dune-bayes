"""Last-layer-only Bayesian MLP shape function (ADR-0004, issue 0014 / GitHub #15).

NeuralLinearMLP: deterministic hidden basis (nn.Linear) with a single variational
output layer (VariationalDense). Cheaper and more stable than the fully-variational
BayesianMLP; the natural baseline for large-data regimes and quick runs.

Design:
  - hidden_dims: widths of the deterministic nn.Linear hidden layers.
  - output layer: VariationalDense with use_bias=False (intercept handled by
    BayesianIntercept, keeping the additive decomposition clean).
  - Activation applied manually between hidden layers; output layer has no activation.
  - No nn.Dropout: weight posterior supplies stochasticity (CONTEXT.md "Dropout
    interaction"); here the stochasticity is confined to the output layer alone.
  - KL is zero for hidden layers (deterministic); collect_kl aggregates only the
    output-layer KL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_bamlss.layers import VariationalDense

_DEFAULT_HIDDEN_DIMS: list[int] = [64, 64]

_ACTIVATIONS: dict[str | None, object] = {
    None: None,
    "linear": None,
    "relu": F.relu,
    "tanh": torch.tanh,
}


class NeuralLinearMLP(nn.Module):
    """Last-layer-only variational MLP shape function for neural-BAMLSS.

    Maps a feature tensor (batch, in_features) to (batch, param_count).
    Hidden layers are deterministic nn.Linear; only the output layer is a
    VariationalDense so KL is collected solely from the last layer.

    Args:
        in_features: Input dimension.
        param_count: Output dimension — equals the family's param_count.
        hidden_dims: Widths of the intermediate nn.Linear hidden layers.
            Defaults to [64, 64].
        prior_scale: Std of the N(0, prior_scale²) weight prior on the output
            VariationalDense.
        kl_divisor: KL term denominator — set to N (training-set size).
        flipout: Use the local-reparameterization estimator for the output layer
            (ADR-0004).
        activation: Activation applied between hidden layers. One of
            {None, "linear", "relu", "tanh"}.
        validate_args: Passed to the output VariationalDense constructor. False
            in the training hot path; True in test fixtures (numerical rule 6).
    """

    def __init__(
        self,
        in_features: int,
        param_count: int,
        hidden_dims: list[int] | None = None,
        prior_scale: float = 1.0,
        kl_divisor: float = 1.0,
        flipout: bool = False,
        activation: str = "relu",
        validate_args: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.param_count = int(param_count)
        if hidden_dims is not None:
            self.hidden_dims = list(hidden_dims)
        else:
            self.hidden_dims = list(_DEFAULT_HIDDEN_DIMS)
        self.prior_scale = float(prior_scale)
        self.kl_divisor = float(kl_divisor)
        self.flipout = bool(flipout)
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from {set(_ACTIVATIONS)}"
            )
        self.activation = activation
        self.validate_args = bool(validate_args)

        hidden_layers: list[nn.Module] = []
        prev = self.in_features
        for width in self.hidden_dims:
            hidden_layers.append(nn.Linear(prev, width))
            prev = width
        self.hidden_layers = nn.ModuleList(hidden_layers)

        # Output layer: variational, no activation, no bias (intercept elsewhere).
        self.output_layer = VariationalDense(
            prev,
            self.param_count,
            prior_scale=prior_scale,
            kl_divisor=kl_divisor,
            flipout=flipout,
            activation=None,
            use_bias=False,
            validate_args=validate_args,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, in_features) → (batch, param_count).

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, param_count).
        """
        act = _ACTIVATIONS[self.activation]
        for layer in self.hidden_layers:
            x = layer(x)
            if act is not None:
                x = act(x)  # type: ignore[operator]
        return self.output_layer(x)

    # ── serialization ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints, floats, strings, bools, lists only)."""
        return {
            "in_features": self.in_features,
            "param_count": self.param_count,
            "hidden_dims": self.hidden_dims,
            "prior_scale": self.prior_scale,
            "kl_divisor": self.kl_divisor,
            "flipout": self.flipout,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config: dict) -> "NeuralLinearMLP":
        """Reconstruct a NeuralLinearMLP from a get_config() dict.

        Args:
            config: Dict as returned by get_config().

        Returns:
            New NeuralLinearMLP with the same architecture.
        """
        return cls(**config)
