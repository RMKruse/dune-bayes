"""VariationalDense — the variational atom (ADR-0004, issue 0001 / GitHub #2).

Mean-field Normal weight posterior (loc + softplus(rho) scale), closed-form
Gaussian–Gaussian KL, and an optional local-reparameterization / flipout-style
estimator for lower gradient variance.  Prior is specified by a serializable
float (never a closure) so get_config/from_config round-trips cleanly.

Design decisions encoded here:
  - softplus(rho) keeps scale strictly positive (CLAUDE.md numerical rule 1).
  - KL is analytic, never MC-estimated (numerical rule 4).
  - kl_beta buffer: non-trainable, moves with .to(device), saved in state_dict.
    The warm-up callback (issue 0004 / GitHub #5) drives it via set_kl_beta().
  - validate_args flag: False in the training hot path, True in test fixtures
    (numerical rule 6).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(-3) ≈ 0.049 — tight-ish initial posterior scale.
_RHO_INIT = -3.0

_ACTIVATIONS: dict[str | None, object] = {
    None: None,
    "linear": None,
    "relu": F.relu,
    "tanh": torch.tanh,
}


class VariationalDense(nn.Module):
    """Dense layer with a mean-field Gaussian weight posterior.

    Args:
        in_features: Input dimension. PyTorch builds eagerly so this is
            required at construction time (unlike Keras's lazy build).
        units: Output dimension.
        prior_scale: Std of the N(0, prior_scale²) weight prior. A plain
            float so it survives get_config/from_config without closures.
        kl_divisor: KL term denominator — set to N (training-set size) for
            the KL/N weighting prescribed by ADR-0001.
        flipout: If True use the local-reparameterization estimator: sample
            pre-activations directly from their marginal Normal instead of
            sampling the full weight matrix. Same expectation as vanilla,
            lower gradient variance (see ADR-0004).
        activation: Optional activation name applied after the affine map.
            One of {None, "linear", "relu", "tanh"}.
        use_bias: Whether to include a variational bias vector.
        validate_args: Passed to torch.distributions constructors. False in
            the training hot path; True in test fixtures (numerical rule 6).
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        prior_scale: float = 1.0,
        kl_divisor: float = 1.0,
        flipout: bool = False,
        activation: str | None = None,
        use_bias: bool = True,
        validate_args: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.units = int(units)
        self.prior_scale = float(prior_scale)
        self.kl_divisor = float(kl_divisor)
        self.flipout = bool(flipout)
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from {set(_ACTIVATIONS)}"
            )
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.validate_args = bool(validate_args)

        # Mean-field Normal posterior: loc + softplus(rho).
        self.kernel_loc = nn.Parameter(torch.empty(self.in_features, self.units))
        self.kernel_rho = nn.Parameter(
            torch.full((self.in_features, self.units), _RHO_INIT)
        )
        nn.init.xavier_normal_(self.kernel_loc)

        if self.use_bias:
            self.bias_loc = nn.Parameter(torch.zeros(self.units))
            self.bias_rho = nn.Parameter(torch.full((self.units,), _RHO_INIT))
        else:
            self.register_parameter("bias_loc", None)
            self.register_parameter("bias_rho", None)

        # Non-trainable warm-up factor β ∈ [0, 1] driven by KLWarmup / set_kl_beta.
        # A buffer so it moves with .to(device) and is saved in state_dict.
        self.register_buffer("kl_beta", torch.tensor(1.0))

        # Live autograd tensor stashed each forward() for collect_kl().
        # Initialised to a scalar zero so it is always defined before the
        # first forward pass.
        self.kl: torch.Tensor = torch.zeros(())

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_kl(
        loc: torch.Tensor,
        scale: torch.Tensor,
        prior_scale: float,
    ) -> torch.Tensor:
        """Closed-form KL[ N(loc, scale²) ‖ N(0, prior_scale²) ], summed over all dims.

        KL = log(prior_scale/scale) + (scale² + loc²)/(2·prior_scale²) − ½
        """
        return torch.sum(
            math.log(prior_scale)
            - torch.log(scale)
            + (scale.pow(2) + loc.pow(2)) / (2.0 * prior_scale**2)
            - 0.5
        )

    def _forward_vanilla(
        self,
        x: torch.Tensor,
        kernel_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Global reparameterization: one weight-matrix sample per forward pass."""
        kernel = self.kernel_loc + kernel_scale * torch.randn_like(self.kernel_loc)
        return x @ kernel

    def _forward_flipout(
        self,
        x: torch.Tensor,
        kernel_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Local reparameterization: sample pre-activations from their marginal Normal.

        For a linear map h = x W with W ~ N(loc, diag(scale²)):
          h ~ N(x @ loc, x² @ scale²)  (each output element independently)

        Sampling h directly from this marginal decouples noise across the batch,
        which strictly reduces gradient variance vs. the global weight draw.
        """
        mean_out = x @ self.kernel_loc
        # (x² @ scale²) gives the per-element variance of the pre-activation.
        var_out = (x**2) @ (kernel_scale**2)
        std_out = torch.sqrt(var_out)
        return mean_out + std_out * torch.randn_like(mean_out)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_scale = F.softplus(self.kernel_rho)

        if self.flipout:
            out = self._forward_flipout(x, kernel_scale)
        else:
            out = self._forward_vanilla(x, kernel_scale)

        kl = self._gaussian_kl(self.kernel_loc, kernel_scale, self.prior_scale)

        if self.use_bias:
            bias_scale = F.softplus(self.bias_rho)
            bias = self.bias_loc + bias_scale * torch.randn_like(self.bias_loc)
            out = out + bias
            kl = kl + self._gaussian_kl(self.bias_loc, bias_scale, self.prior_scale)

        # KL/N annealed by β, stashed so collect_kl() can aggregate after the pass.
        self.kl = self.kl_beta * kl / self.kl_divisor

        act = _ACTIVATIONS[self.activation]
        if act is not None:
            out = act(out)  # type: ignore[operator]
        return out

    # ── serialization ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints, floats, strings, bools only)."""
        return {
            "in_features": self.in_features,
            "units": self.units,
            "prior_scale": self.prior_scale,
            "kl_divisor": self.kl_divisor,
            "flipout": self.flipout,
            "activation": self.activation,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config: dict) -> "VariationalDense":
        return cls(**config)


# ── module-level utilities ────────────────────────────────────────────────────


def _iter_variational_layers(model: nn.Module):
    """Yield every variational layer (VariationalDense or BayesianIntercept)."""
    # Local import avoids a circular dependency: bayesian_intercept imports this module.
    from neural_bamlss.layers.bayesian_intercept import (
        BayesianIntercept,  # noqa: PLC0415
    )

    for module in model.modules():
        if isinstance(module, (VariationalDense, BayesianIntercept)):
            yield module


def collect_kl(model: nn.Module) -> torch.Tensor:
    """Sum the most-recent KL of every VariationalDense in the module tree.

    Call after a forward pass. The explicit module-walk replaces Keras
    add_loss auto-propagation (ADR-0004 / ADR-0006): model.modules() recurses
    through all sub-modules, so KL crosses nested-module boundaries trivially.

    Returns:
        Scalar tensor (with autograd) equal to sum of all stashed .kl values.
    """
    total = torch.zeros(())
    for layer in _iter_variational_layers(model):
        total = total + layer.kl
    return total


def set_kl_beta(model: nn.Module, beta: float) -> None:
    """Set the warm-up annealing factor on every VariationalDense in a model.

    Args:
        model: Any nn.Module containing VariationalDense sub-modules.
        beta: New value for kl_beta, typically in [0, 1].
    """
    for layer in _iter_variational_layers(model):
        layer.kl_beta.fill_(float(beta))
