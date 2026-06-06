"""VariationalDense — the variational atom (ADR-0004, issues #2, #73).

Mean-field Normal weight posterior (loc + softplus(rho) scale), closed-form
Gaussian–Gaussian KL, and an optional local-reparameterization / flipout-style
estimator for lower gradient variance.  Prior is specified by a serializable
float (never a closure) so get_config/from_config round-trips cleanly — or by
a PriorScale handle for the ADR-0002 empirical-Bayes / hierarchical tiers.

Design decisions encoded here:
  - softplus(rho) keeps scale strictly positive (CLAUDE.md numerical rule 1).
  - KL is analytic, never MC-estimated (numerical rule 4).
  - KL plumbing (kl_beta buffer, .kl stash, β·KL/N scaling) is inherited from
    VariationalLayer (layers/base.py, GitHub #64). The warm-up callback
    (issue 0004 / GitHub #5) drives kl_beta via set_kl_beta().
  - validate_args flag: False in the training hot path, True in test fixtures
    (numerical rule 6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_bamlss.layers.base import _RHO_INIT, VariationalLayer, gaussian_kl
from neural_bamlss.utils import resolve_activation

if TYPE_CHECKING:
    from neural_bamlss.priors.prior_scale import PriorScale

# PriorScale imports stay local (TYPE_CHECKING + function bodies): PriorScale
# subclasses VariationalLayer, so priors → layers/__init__ → this module is a
# real cycle if PriorScale is needed at module-exec time (issue #73).


class VariationalDense(VariationalLayer):
    """Dense layer with a mean-field Gaussian weight posterior.

    Args:
        in_features: Input dimension. PyTorch builds eagerly so this is
            required at construction time (unlike Keras's lazy build).
        units: Output dimension.
        prior_scale: Std of the N(0, prior_scale²) weight prior. A plain
            float so it survives get_config/from_config without closures.
        prior_scale_handle: Optional PriorScale (ADR-0002, issue #73). When
            given it overrides ``prior_scale``: the KL uses a live scalar
            tensor s = handle() so the empirical-Bayes / hierarchical scale
            keeps its gradient path. Several layers may share one handle
            (one smoothness scalar per feature net); the handle stashes its
            own hyperprior KL, counted once by collect_kl.
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
        prior_scale_handle: PriorScale | None = None,
        kl_divisor: float = 1.0,
        flipout: bool = False,
        activation: str | None = None,
        use_bias: bool = True,
        validate_args: bool = False,
    ) -> None:
        super().__init__(kl_divisor=kl_divisor)
        self.in_features = int(in_features)
        self.units = int(units)
        self.prior_scale = float(prior_scale)
        self.prior_scale_handle = prior_scale_handle
        self.flipout = bool(flipout)
        # Resolved once here (validates the name); config keeps the string.
        self._act = resolve_activation(activation)
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

    # ── private helpers ───────────────────────────────────────────────────────

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

        # One handle() call per forward: kernel and bias share the same prior
        # scale draw, and the handle stashes its own hyperprior KL on the call.
        prior: float | torch.Tensor
        if self.prior_scale_handle is not None:
            prior = self.prior_scale_handle()
        else:
            prior = self.prior_scale

        kl = gaussian_kl(self.kernel_loc, kernel_scale, prior)

        if self.use_bias:
            bias_scale = F.softplus(self.bias_rho)
            bias = self.bias_loc + bias_scale * torch.randn_like(self.bias_loc)
            out = out + bias
            kl = kl + gaussian_kl(self.bias_loc, bias_scale, prior)

        # β·KL/N stashed (via the base) so collect_kl() can aggregate after the pass.
        self._stash_kl(kl)

        if self._act is not None:
            out = self._act(out)
        return out

    # ── serialization ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free config (ints, floats, strings, bools, dicts only).

        A standalone round-trip rebuilds the handle from ``prior_scale_config``;
        a net that shares one handle across layers reconstructs and re-injects
        it itself (see BayesianMLP.from_config), bypassing this key.
        """
        return {
            "in_features": self.in_features,
            "units": self.units,
            "prior_scale": self.prior_scale,
            "prior_scale_config": (
                None
                if self.prior_scale_handle is None
                else self.prior_scale_handle.get_config()
            ),
            "kl_divisor": self.kl_divisor,
            "flipout": self.flipout,
            "activation": self.activation,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config: dict) -> VariationalDense:
        from neural_bamlss.priors.prior_scale import PriorScale

        cfg = dict(config)
        ps_config = cfg.pop("prior_scale_config", None)
        handle = None if ps_config is None else PriorScale.from_config(ps_config)
        return cls(prior_scale_handle=handle, **cfg)
