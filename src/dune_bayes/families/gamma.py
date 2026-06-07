"""Gamma family for dune-bayes (issue 0019 / GitHub #42).

Maps a (batch, 2) parameter tensor to a torch.distributions.Gamma.
Links: softplus for concentration and rate (both strictly positive,
numerical rule 1). Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F
from scipy import stats

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS, to_numpy


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
        # softplus enforces strict positivity (numerical rule 1); EPS floor
        # guards very negative raw inputs.
        concentration = F.softplus(params[..., 0]) + EPS
        rate = F.softplus(params[..., 1]) + EPS
        return torch.distributions.Gamma(
            concentration=concentration, rate=rate, validate_args=self.validate_args
        )

    def cdf(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gamma CDF via scipy (issue 0093 / GitHub #93).

        torch parameterizes by rate, scipy.stats.gamma by scale — the
        conversion is ``scale = 1/rate`` (cross-checked against torch's own
        sampler in tests/families/test_cdf.py). Reuses ``__call__`` so the
        links stay single-sourced; eval-time only, no gradient path.

        Args:
            params: Tensor of shape (..., 2), as in ``__call__``.
            y: Observed responses, broadcastable to the batch shape (...).

        Returns:
            Tensor of shape (...), dtype float64.
        """
        with torch.no_grad():
            dist = self(params)
            value = stats.gamma.cdf(
                to_numpy(y),
                a=to_numpy(dist.concentration),
                scale=1.0 / to_numpy(dist.rate),
            )
        return torch.from_numpy(value).to(torch.float64)
