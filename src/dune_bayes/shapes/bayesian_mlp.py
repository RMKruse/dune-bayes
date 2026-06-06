"""Fully-variational MLP shape function (ADR-0004, issues #3, #73).

BayesianMLP: every dense layer is a VariationalDense so uncertainty propagates
through the function shape, not just a final rescaling. Internal per-layer
dropout is absent — the weight posterior supplies the stochasticity.

Design:
  - hidden_dims: list of widths for the stacked variational hidden layers.
  - output layer: VariationalDense with use_bias=False (intercept is handled by
    BayesianIntercept, keeping the additive decomposition clean).
  - No nn.Dropout: epistemic uncertainty must not be conflated with dropout noise
    (CONTEXT.md "Dropout interaction").
  - prior=: one PriorScale handle per net, shared by every layer — ADR-0002's
    one-smoothness-scalar-per-feature-net granularity (issue #73).
"""

import torch
import torch.nn as nn

from dune_bayes.layers import VariationalDense
from dune_bayes.priors.prior_scale import PriorScale

_DEFAULT_HIDDEN_DIMS: list[int] = [64, 64]


class BayesianMLP(nn.Module):
    """Fully-variational MLP shape function for dune-bayes.

    Maps a feature tensor (batch, in_features) to (batch, param_count).
    Every layer is a VariationalDense so KL is collectable via collect_kl().

    Args:
        in_features: Input dimension.
        param_count: Output dimension — equals the family's param_count.
        hidden_dims: Widths of the intermediate VariationalDense layers.
            Defaults to [64, 64].
        prior_scale: Std of the N(0, prior_scale²) weight prior. Shared
            across all layers; per-layer override is future work (ADR-0002).
        prior: Optional prior-tier spec (ADR-0002, issue #73): a mode string
            (``"fixed"`` | ``"empirical_bayes"`` | ``"hierarchical"``) or a
            PriorScale config dict. Builds ONE handle shared by every layer —
            the per-feature-net smoothness scalar — with ``prior_scale`` as
            its default scale and ``kl_divisor`` forwarded. Literal-friendly,
            so it works verbatim as a formula kwarg. None keeps the plain
            fixed-float prior.
        kl_divisor: KL term denominator — set to N (training-set size).
        flipout: Use the local-reparameterization estimator (ADR-0004).
        activation: Activation between hidden layers.
        validate_args: Passed to VariationalDense constructors. False in the
            training hot path; True in test fixtures (numerical rule 6).
    """

    def __init__(
        self,
        in_features: int,
        param_count: int,
        hidden_dims: list[int] | None = None,
        prior_scale: float = 1.0,
        prior: str | dict | None = None,
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
        # Copy dict specs so a caller mutating theirs can't skew get_config().
        self.prior = dict(prior) if isinstance(prior, dict) else prior
        self.kl_divisor = float(kl_divisor)
        self.flipout = bool(flipout)
        self.activation = activation
        self.validate_args = bool(validate_args)

        # One handle per net (ADR-0002 granularity), shared by every layer.
        # In hierarchical mode each layer's forward draws its own s ~ q(s);
        # each per-layer KL term is then an unbiased estimate of
        # E_q(s) KL[q(w) ‖ p(w|s)], so the summed ELBO term stays unbiased —
        # one smoothness scalar per net in expectation.
        if prior is not None:
            self.prior_scale_handle: PriorScale | None = PriorScale.from_spec(
                prior, scale=self.prior_scale, kl_divisor=self.kl_divisor
            )
        else:
            self.prior_scale_handle = None

        vd_kwargs = dict(
            prior_scale=prior_scale,
            prior_scale_handle=self.prior_scale_handle,
            kl_divisor=kl_divisor,
            flipout=flipout,
            activation=activation,
            validate_args=validate_args,
        )

        layers: list[nn.Module] = []
        prev = self.in_features
        for width in self.hidden_dims:
            layers.append(VariationalDense(prev, width, **vd_kwargs))
            prev = width

        # Output layer: no activation, no bias (intercept handled elsewhere).
        layers.append(
            VariationalDense(
                prev,
                self.param_count,
                prior_scale=prior_scale,
                prior_scale_handle=self.prior_scale_handle,
                kl_divisor=kl_divisor,
                flipout=flipout,
                activation=None,
                use_bias=False,
                validate_args=validate_args,
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, in_features) → (batch, param_count).

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, param_count).
        """
        for layer in self.layers:
            x = layer(x)
        return x

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
            "flipout": self.flipout,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config: dict) -> "BayesianMLP":
        """Reconstruct a BayesianMLP from a get_config() dict.

        Args:
            config: Dict as returned by get_config().

        Returns:
            New BayesianMLP with the same architecture.
        """
        return cls(**config)
