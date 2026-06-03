"""Gamma family for neural-BAMLSS (issue 0019 / GitHub #42).

Maps a (batch, 2) parameter tensor to a torch.distributions.Gamma.
Links: softplus for concentration and rate (both strictly positive,
numerical rule 1). Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F

from neural_bamlss.families.base import BaseFamily


class GammaFamily(BaseFamily):
    """Gamma response family for positive continuous outcomes.

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
    """

    param_count: int = 2

    def __init__(self, validate_args: bool = False) -> None:
        self.validate_args = bool(validate_args)

    def __call__(self, params: torch.Tensor) -> torch.distributions.Gamma:
        """Apply parameter links and return a Gamma distribution.

        Args:
            params: Tensor of shape (..., 2) — column 0 is raw concentration
                pre-activation (softplus), column 1 is raw rate pre-activation
                (softplus).

        Returns:
            torch.distributions.Gamma with batch_shape (...).
        """
        concentration = F.softplus(params[..., 0]) + 1e-6
        rate = F.softplus(params[..., 1]) + 1e-6
        return torch.distributions.Gamma(
            concentration=concentration, rate=rate, validate_args=self.validate_args
        )

    def log_prob(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pointwise Gamma log-likelihood.

        Args:
            params: Raw network output, shape (batch, 2).
            y: Observed positive responses, shape (batch,).

        Returns:
            Tensor of shape (batch,) with pointwise log p(y | params).
        """
        return self(params).log_prob(y)
