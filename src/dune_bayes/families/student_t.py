"""Student-T family for dune-bayes (issue 0019 / GitHub #42; df_min: GitHub #91).

Maps a (batch, 3) parameter tensor to a torch.distributions.StudentT.
Links: identity for loc, softplus for scale (numerical rule 1),
softplus + EPS + df_min for df.
Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS


class StudentTFamily(BaseFamily):
    """Student-T response family with loc, scale, and degrees-of-freedom.

    Parameterization (issue 0091 / GitHub #91): the df link is
    ``df = softplus(raw_df) + EPS + df_min``, so ``df > df_min`` *strictly*
    for any pre-link value (the EPS floor matters: bare softplus underflows
    to exact 0.0 in float32, and torch's StudentT moments are NaN at the
    df-boundary). Moment existence depends on where df lands:

      - ``df > 1`` — mean is finite (the default, ``df_min=1.0``, avoids the
        Cauchy regime).
      - ``df > 2`` — variance is finite. With the default link, draws with
        ``df ≤ 2`` have truly infinite aleatoric variance; the variance
        decomposition surfaces that honestly (``inf`` + a warning), never
        clamped. Experiments that need finite variance pin ``df_min=2.0``.

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
        df_min: Lower bound of the df link, ``df = softplus(raw_df) + df_min``.
            Default 1.0 (finite mean); use 2.0 to pin finite variance.
    """

    param_count: int = 3

    def __init__(self, validate_args: bool = False, df_min: float = 1.0) -> None:
        self.validate_args = bool(validate_args)
        self.df_min = float(df_min)

    def __call__(self, params: torch.Tensor) -> torch.distributions.StudentT:
        """Apply parameter links and return a StudentT distribution.

        Args:
            params: Tensor of shape (..., 3) — column 0 is loc (identity link),
                column 1 is raw scale pre-activation (softplus), column 2 is
                raw df pre-activation (softplus + EPS + df_min).

        Returns:
            torch.distributions.StudentT with batch_shape (...).
        """
        loc = params[..., 0]
        # softplus enforces strict positivity; EPS floor for extreme inputs
        scale = F.softplus(params[..., 1]) + EPS
        # df > df_min STRICTLY: bare softplus underflows to exact 0.0 near
        # pre-link −104 in float32 (numerical rule 1), which would make
        # df == df_min exactly — and torch's StudentT moments are NaN at the
        # boundary (mean needs df > 1 strictly). The EPS floor keeps df
        # strictly above df_min: 1.0 (default) keeps the mean defined and
        # avoids the Cauchy regime; 2.0 additionally pins the variance
        # finite (#91).
        df = F.softplus(params[..., 2]) + EPS + self.df_min
        return torch.distributions.StudentT(
            df=df, loc=loc, scale=scale, validate_args=self.validate_args
        )
