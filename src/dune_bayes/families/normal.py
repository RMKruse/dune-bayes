"""Normal (Gaussian) family for dune-bayes (issue 0003 / GitHub #4).

Maps a (batch, 2) parameter tensor to a torch.distributions.Normal.
Links: identity for loc, softplus for scale (positivity via softplus,
numerical rule 1). Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F
from scipy import stats

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS, to_numpy


class NormalFamily(BaseFamily):
    """Normal response family.

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
    """

    param_count: int = 2

    def __init__(self, validate_args: bool = False) -> None:
        self.validate_args = bool(validate_args)

    def __call__(self, params: torch.Tensor) -> torch.distributions.Normal:
        """Apply parameter links and return a Normal distribution.

        Args:
            params: Tensor of shape (batch, 2) — column 0 is loc (identity
                link), column 1 is the raw scale pre-activation (softplus).

        Returns:
            torch.distributions.Normal with batch_shape (batch,).
        """
        loc = params[..., 0]
        # softplus enforces strict positivity (numerical rule 1); EPS floor
        # guards the corner case of very negative raw inputs.
        scale = F.softplus(params[..., 1]) + EPS
        return torch.distributions.Normal(
            loc=loc, scale=scale, validate_args=self.validate_args
        )

    def cdf(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gaussian CDF Φ((y − μ)/σ) via scipy (issue 0093 / GitHub #93).

        Reuses ``__call__`` so the links stay single-sourced; eval-time only,
        no gradient path (BaseFamily.cdf contract).

        Args:
            params: Tensor of shape (..., 2), as in ``__call__``.
            y: Observed responses, broadcastable to the batch shape (...).

        Returns:
            Tensor of shape (...), dtype float64.
        """
        with torch.no_grad():
            dist = self(params)
            value = stats.norm.cdf(
                to_numpy(y), loc=to_numpy(dist.loc), scale=to_numpy(dist.scale)
            )
        return torch.from_numpy(value).to(torch.float64)
