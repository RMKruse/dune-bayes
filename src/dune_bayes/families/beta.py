"""Beta family for dune-bayes (issue 0096 / GitHub #96).

Maps a (batch, 2) parameter tensor to a torch.distributions.Beta. The pinned
convention is beta-regression mean/precision (Ferrari & Cribari-Neto 2004;
the GAMLSS-adjacent convention): heads are (μ, φ) with mean μ ∈ (0, 1) and
precision φ > 0, and Var(y) = μ(1−μ)/(1+φ). torch parameterizes by
concentrations instead; the translation is

    concentration1 = μφ,    concentration0 = (1−μ)φ,

so concentration1 + concentration0 = φ (the precision IS the total
concentration) and the distribution's mean is exactly μ.
Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F
from scipy import stats
from torch.distributions import constraints

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS, to_numpy


class _OpenUnitInterval(constraints.interval):
    """Strictly open unit interval (0, 1).

    torch has no open-interval constraint (2.12), and its Beta pins the
    CLOSED ``unit_interval`` — but the Beta log-density is ±inf (or NaN) at
    the endpoints, so the honest support is open. Without this, the closed
    support admits boundary y into the #88/#89 support-filtered gates and
    into ``validate_args=True`` fixtures, which then meet an unavoidable
    −inf instead of a clean validation error. Subclasses the closed
    ``interval`` (rather than bare ``Constraint``) so it stays
    type-compatible with the ``support`` attribute Beta declares.
    """

    def __init__(self) -> None:
        super().__init__(0.0, 1.0)

    def check(self, value: torch.Tensor) -> torch.Tensor:
        # Literal bounds: torch types lower_bound/upper_bound as Any.
        return (value > 0.0) & (value < 1.0)


class _OpenSupportBeta(torch.distributions.Beta):
    """torch Beta with the support tightened to the open interval (0, 1).

    No ``__init__`` override, so ``expand`` / ``MixtureSameFamily`` keep
    working through ``_get_checked_instance`` and preserve the subclass.
    """

    support = _OpenUnitInterval()


class BetaFamily(BaseFamily):
    """Beta response family for bounded (0, 1) outcomes.

    Parameterization (beta-regression mean/precision, pinned in issue #96):
    column 0 is the mean μ, column 1 the precision φ, with
    Var(y) = μ(1−μ)/(1+φ) — φ → ∞ concentrates mass at μ. The mean is
    linked ``EPS + (1 − 2·EPS)·sigmoid(x)`` (sigmoid saturates to exactly
    0.0/1.0 in float32 near pre-link ∓90, which would zero a concentration —
    the bounded-link analogue of numerical rule 1), the precision
    ``softplus(x) + EPS``.

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
    """

    param_count: int = 2

    def __init__(self, validate_args: bool = False) -> None:
        self.validate_args = bool(validate_args)

    def __call__(self, params: torch.Tensor) -> torch.distributions.Beta:
        """Apply parameter links and return a Beta distribution.

        Args:
            params: Tensor of shape (..., 2) — column 0 is raw mean
                pre-activation (floored sigmoid), column 1 is raw precision
                pre-activation (softplus).

        Returns:
            torch.distributions.Beta with batch_shape (...).
        """
        # Floored sigmoid keeps μ in [EPS, 1−EPS] (numerical rule 1 for a
        # bounded parameter); softplus + EPS keeps φ strictly positive — so
        # both concentrations below are ≥ EPS² > 0 for any finite pre-link.
        mu = EPS + (1.0 - 2.0 * EPS) * torch.sigmoid(params[..., 0])
        phi = F.softplus(params[..., 1]) + EPS
        return _OpenSupportBeta(
            concentration1=mu * phi,
            concentration0=(1.0 - mu) * phi,
            validate_args=self.validate_args,
        )

    def cdf(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Beta CDF via scipy (issue 0093 / GitHub #93).

        scipy's ``beta(a, b)`` are torch's concentrations verbatim
        (a = concentration1 = μφ, b = concentration0 = (1−μ)φ; cross-checked
        against torch's own sampler in tests/families/test_cdf.py). Reuses
        ``__call__`` so the links stay single-sourced; eval-time only, no
        gradient path.

        Args:
            params: Tensor of shape (..., 2), as in ``__call__``.
            y: Observed responses, broadcastable to the batch shape (...).

        Returns:
            Tensor of shape (...), dtype float64.
        """
        with torch.no_grad():
            dist = self(params)
            value = stats.beta.cdf(
                to_numpy(y),
                a=to_numpy(dist.concentration1),
                b=to_numpy(dist.concentration0),
            )
        return torch.from_numpy(value).to(torch.float64)
