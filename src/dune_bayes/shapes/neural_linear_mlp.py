"""Last-layer-only Bayesian MLP shape fn (ADR-0004, ADR-0007, issues #15, #73, #85).

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

from dune_bayes.layers import VariationalDense
from dune_bayes.priors.prior_scale import PriorScale
from dune_bayes.utils import resolve_activation

_DEFAULT_HIDDEN_DIMS: list[int] = [64, 64]


class NeuralLinearMLP(nn.Module):
    """Last-layer-only variational MLP shape function for dune-bayes.

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
        prior: Optional prior-tier spec (ADR-0002, issue #73): a mode string
            (``"fixed"`` | ``"empirical_bayes"`` | ``"hierarchical"``) or a
            PriorScale config dict, applied to the output layer — the net's
            only variational layer, so the one handle is still the per-net
            smoothness scalar. ``prior_scale`` is its default scale and
            ``kl_divisor`` is forwarded. Literal-friendly for formula kwargs.
            None keeps the plain fixed-float prior.
        kl_divisor: KL term denominator — set to N (training-set size).
        local_reparam: Use the local-reparameterization estimator for the
            output layer's training-time forward passes (ADR-0007). Defaults
            to True — the training default; eval always takes coherent
            global weight draws.
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
        prior: str | dict | None = None,
        kl_divisor: float = 1.0,
        local_reparam: bool = True,
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
        # Copy dict specs so a caller mutating theirs can't skew get_config().
        self.prior = dict(prior) if isinstance(prior, dict) else prior
        self.kl_divisor = float(kl_divisor)
        self.local_reparam = bool(local_reparam)
        # Resolved once here (validates the name); config keeps the string.
        self._act = resolve_activation(activation)
        self.activation = activation
        self.validate_args = bool(validate_args)

        # One handle per net (ADR-0002 granularity) — held by the single
        # variational layer; the handle stashes its own hyperprior KL.
        if prior is not None:
            self.prior_scale_handle: PriorScale | None = PriorScale.from_spec(
                prior, scale=self.prior_scale, kl_divisor=self.kl_divisor
            )
        else:
            self.prior_scale_handle = None

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
            prior_scale_handle=self.prior_scale_handle,
            kl_divisor=kl_divisor,
            local_reparam=local_reparam,
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
        for layer in self.hidden_layers:
            x = layer(x)
            if self._act is not None:
                x = self._act(x)
        # nn.Module.__call__ is untyped in torch's stubs; anchor the return type.
        out: torch.Tensor = self.output_layer(x)
        return out

    # ── serialization ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints, floats, strings, bools, lists only)."""
        return {
            "in_features": self.in_features,
            "param_count": self.param_count,
            "hidden_dims": self.hidden_dims,
            "prior_scale": self.prior_scale,
            "prior": self.prior,
            "kl_divisor": self.kl_divisor,
            "local_reparam": self.local_reparam,
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
