"""Negative-binomial family for dune-bayes (issue 0095 / GitHub #95).

Maps a (batch, 2) parameter tensor to a torch.distributions.NegativeBinomial.
The pinned convention is GAMLSS NBI (mean/dispersion): heads are (μ, σ) with
Var(y) = μ + σμ², both linked softplus + EPS (numerical rule 1). torch
parameterizes by (total_count, logits) instead; the translation is

    total_count = 1/σ,    logits = log(p/(1−p)) = log(μσ) = log(μ) + log(σ),

since torch's per-trial success probability is p = μ/(μ + 1/σ). scipy's
``nbinom`` counts the OTHER outcome, so its probability is 1 − p = 1/(1 + σμ)
(tested explicitly in tests/families/test_negative_binomial.py, AC1).
Implements BaseFamily (issue 0018 / GitHub #38).
"""

import torch
import torch.nn.functional as F
from scipy import stats

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS, to_numpy


class NegativeBinomialFamily(BaseFamily):
    """Negative-binomial response family for overdispersed counts.

    Parameterization (GAMLSS NBI, pinned in issue #95): column 0 is the mean
    μ, column 1 the dispersion σ, with Var(y) = μ + σμ² — σ → 0 recovers
    Poisson(μ). Both are linked ``softplus(x) + EPS``, so the logs in the
    torch translation (``logits = log(μ) + log(σ)``) act on EPS-floored
    quantities, never bare learned values (numerical rule 3).

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
    """

    param_count: int = 2

    def __init__(self, validate_args: bool = False) -> None:
        self.validate_args = bool(validate_args)

    def __call__(self, params: torch.Tensor) -> torch.distributions.NegativeBinomial:
        """Apply parameter links and return a NegativeBinomial distribution.

        Args:
            params: Tensor of shape (..., 2) — column 0 is raw mean
                pre-activation (softplus), column 1 is raw dispersion
                pre-activation (softplus).

        Returns:
            torch.distributions.NegativeBinomial with batch_shape (...).
        """
        # softplus enforces strict positivity (numerical rule 1); EPS floor
        # guards very negative raw inputs — and floors the logs below.
        mu = F.softplus(params[..., 0]) + EPS
        sigma = F.softplus(params[..., 1]) + EPS
        # GAMLSS NBI → torch: total_count = 1/σ, logits = log(μσ). The logits
        # route keeps p strictly inside (0, 1) for any finite pre-link value
        # (probs would saturate to exactly 1.0 in float32 for large μσ).
        return torch.distributions.NegativeBinomial(
            total_count=1.0 / sigma,
            logits=torch.log(mu) + torch.log(sigma),
            validate_args=self.validate_args,
        )

    def cdf(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative-binomial CDF via scipy (issue 0093 / GitHub #93).

        scipy's ``nbinom`` success probability is the complement of torch's:
        torch counts successes before ``total_count`` failures, scipy counts
        failures before ``n`` successes — so ``p = sigmoid(−logits)``
        (cross-checked against torch's own sampler in
        tests/families/test_cdf.py). Reuses ``__call__`` so the links stay
        single-sourced; eval-time only, no gradient path.

        Args:
            params: Tensor of shape (..., 2), as in ``__call__``.
            y: Observed responses, broadcastable to the batch shape (...).

        Returns:
            Tensor of shape (...), dtype float64.
        """
        with torch.no_grad():
            dist = self(params)
            value = stats.nbinom.cdf(
                to_numpy(y),
                n=to_numpy(dist.total_count),
                p=to_numpy(torch.sigmoid(-dist.logits)),
            )
        return torch.from_numpy(value).to(torch.float64)
