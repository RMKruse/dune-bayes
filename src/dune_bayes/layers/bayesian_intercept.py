"""BayesianIntercept — variational intercept layer (issue 0010 / GitHub #11).

One scalar per distributional parameter with a wide weakly-informative
Normal(0, prior_scale²) prior, deliberately independent of the per-feature
prior_scale (which is a smoothness term, not an intercept term).

Effect plots mean-centre each feature curve into the intercept, so the absorbed
overall level — and its uncertainty — accumulates here.  Bayesian by default;
a `point` deterministic fallback is selectable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dune_bayes.layers.base import _RHO_INIT, VariationalLayer, gaussian_kl


class BayesianIntercept(VariationalLayer):
    """Variational intercept with a wide prior, independent of per-feature prior_scale.

    Args:
        units: Number of intercept scalars (one per distributional parameter).
        prior_scale: Std of N(0, prior_scale²) prior. Wide default (10.0) so
            the intercept is not shrunk toward zero like a smoothness term.
        kl_divisor: KL term denominator — set to N for KL/N weighting (ADR-0001).
        mode: ``"variational"`` (default) for a mean-field Gaussian posterior;
            ``"point"`` for a deterministic nn.Parameter with no KL contribution.
        validate_args: Passed to torch.distributions. False in training; True
            in test fixtures (numerical rule 6).
    """

    def __init__(
        self,
        units: int = 1,
        prior_scale: float = 10.0,
        kl_divisor: float = 1.0,
        mode: str = "variational",
        validate_args: bool = False,
    ) -> None:
        super().__init__(kl_divisor=kl_divisor)
        if mode not in ("variational", "point"):
            raise ValueError(f"unknown mode {mode!r}; choose 'variational' or 'point'")
        self.units = int(units)
        self.prior_scale = float(prior_scale)
        self.mode = mode
        self.validate_args = bool(validate_args)

        self.loc = nn.Parameter(torch.zeros(self.units))
        if self.mode == "variational":
            self.rho = nn.Parameter(torch.full((self.units,), _RHO_INIT))
        else:
            self.register_parameter("rho", None)

    def forward(self, n_samples: int | None = None) -> torch.Tensor:
        """Draw the intercept; optionally a batch of independent draws.

        Args:
            n_samples: When given, return (n_samples, units) with one fresh
                posterior draw per row — the sample-dimension form used by the
                vectorized T-sweeps (issue 0027 / GitHub #80). None (default)
                keeps the single-draw (units,) training-path behavior.

        Returns:
            Tensor of shape (units,) or (n_samples, units).
        """
        if self.mode == "variational":
            scale = F.softplus(self.rho)
            if n_samples is None:
                noise = torch.randn_like(self.loc)
            else:
                # Fresh noise per sample row — never one draw broadcast S ways.
                noise = torch.randn(
                    n_samples, self.units, device=self.loc.device, dtype=self.loc.dtype
                )
            sample = self.loc + scale * noise
            self._stash_kl(gaussian_kl(self.loc, scale, self.prior_scale))
        else:
            sample = self.loc if n_samples is None else self.loc.expand(n_samples, -1)
            self.kl = torch.zeros(())
        return sample

    def get_config(self) -> dict:
        """Return a closure-free config (ints, floats, strings only)."""
        return {
            "units": self.units,
            "prior_scale": self.prior_scale,
            "kl_divisor": self.kl_divisor,
            "mode": self.mode,
        }

    @classmethod
    def from_config(cls, config: dict) -> "BayesianIntercept":
        return cls(**config)
