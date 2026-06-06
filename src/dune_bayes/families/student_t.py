"""Student-T family for dune-bayes (issue 0019 / GitHub #42).

Maps a (batch, 3) parameter tensor to a torch.distributions.StudentT.
Links: identity for loc, softplus for scale (numerical rule 1),
softplus+1 for df (ensures df > 1, guaranteeing finite variance).
Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS


class StudentTFamily(BaseFamily):
    """Student-T response family with loc, scale, and degrees-of-freedom.

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
    """

    param_count: int = 3

    def __init__(self, validate_args: bool = False) -> None:
        self.validate_args = bool(validate_args)

    def __call__(self, params: torch.Tensor) -> torch.distributions.StudentT:
        """Apply parameter links and return a StudentT distribution.

        Args:
            params: Tensor of shape (..., 3) — column 0 is loc (identity link),
                column 1 is raw scale pre-activation (softplus), column 2 is
                raw df pre-activation (softplus + 1, ensures df > 1).

        Returns:
            torch.distributions.StudentT with batch_shape (...).
        """
        loc = params[..., 0]
        # softplus enforces strict positivity; EPS floor for extreme inputs
        scale = F.softplus(params[..., 1]) + EPS
        # df > 1 guarantees finite variance; offset of 1 avoids Cauchy regime
        df = F.softplus(params[..., 2]) + 1.0
        return torch.distributions.StudentT(
            df=df, loc=loc, scale=scale, validate_args=self.validate_args
        )
