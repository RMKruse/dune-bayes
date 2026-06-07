"""Johnson's SU distribution + family (PRD 0002 / GitHub #84, issue #94).

``torch.distributions`` has no Johnson's SU, so this module implements it as a
custom ``Distribution`` subclass (a ``TransformedDistribution`` was rejected in
issue #94 — it has no closed-form moments). The parameterization is pinned to
**scipy's ``johnsonsu``**:

    z = γ + δ·arcsinh((y − ξ)/λ),   z ~ N(0, 1)

with parameters (γ skew, δ > 0 tailweight, ξ loc, λ > 0 scale) mapping
one-to-one onto scipy's ``(a, b, loc, scale)`` — scipy is a zero-translation
reference for every test.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints

from dune_bayes.families.base import BaseFamily
from dune_bayes.utils import EPS

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
# Module-level singleton: ruff B008 forbids torch.Size() in argument defaults.
_EMPTY_SHAPE = torch.Size()


class JohnsonSU(Distribution):
    """Johnson's SU distribution, scipy ``johnsonsu`` parameterization (#94).

    ``z = skew + tail · arcsinh((y − loc)/scale)`` is standard normal; the
    arguments map to scipy's ``johnsonsu(a=skew, b=tail, loc=loc,
    scale=scale)`` exactly.

    Args:
        skew: γ — skewness parameter (scipy ``a``), real.
        tail: δ — tailweight (scipy ``b``), strictly positive.
        loc: ξ — location (scipy ``loc``), real.
        scale: λ — scale (scipy ``scale``), strictly positive.
        validate_args: torch.distributions validation flag (numerical rule 6).
    """

    arg_constraints = {
        "skew": constraints.real,
        "tail": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        skew: torch.Tensor,
        tail: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.skew, self.tail, self.loc, self.scale = (
            torch.distributions.utils.broadcast_all(skew, tail, loc, scale)
        )
        super().__init__(self.skew.shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        """Closed-form mean: ξ − λ·exp(1/(2δ²))·sinh(γ/δ).

        Validity note (#94): the exp(δ⁻²) factor explodes for small δ — at
        the family link's EPS floor the true mean is ~e^(5e11), finite
        mathematically but unrepresentable in any float. The naive product
        form is worse than overflow: exp(·)·sinh(γ/δ) gives inf·0 = NaN at
        γ = 0, where the true value is exactly ξ. So the magnitude is
        assembled in log-space, exp(log λ + 1/(2δ²) + log|sinh(γ/δ)|), with
        log|sinh(b)| = |b| − log 2 + log1p(−exp(−2|b|)): at γ = 0 the
        log1p(−1) = −inf cancels the exploding 1/(2δ²) term to exp(−inf) = 0
        exactly, and genuine overflow surfaces as an honest ±inf, never NaN
        (the #91 moment-conformance contract). Assumes δ, λ above the EPS
        link floor (1e-6); δ² must not underflow to 0.
        """
        b = self.skew / self.tail
        abs_b = torch.abs(b)
        log_sinh_mag = abs_b - math.log(2.0) + torch.log1p(-torch.exp(-2.0 * abs_b))
        log_mag = torch.log(self.scale) + 0.5 / (self.tail * self.tail) + log_sinh_mag
        return self.loc - torch.sign(b) * torch.exp(log_mag)

    @property
    def variance(self) -> torch.Tensor:
        """Closed-form variance: (λ²/2)·(ω − 1)·(ω·cosh(2γ/δ) + 1), ω = exp(δ⁻²).

        ω − 1 is expm1(δ⁻²) (numerical rule 3 — exact for large δ, where
        exp(δ⁻²) − 1 would cancel to float-0). Same small-δ validity note as
        ``mean``: ω overflows for δ ≲ 0.1 in float32, and every factor is
        non-negative, so overflow propagates as an honest +inf — never NaN,
        never negative (#91 moment-conformance contract).
        """
        inv_tail_sq = 1.0 / (self.tail * self.tail)
        omega_m1 = torch.expm1(inv_tail_sq)
        return (
            0.5
            * self.scale
            * self.scale
            * omega_m1
            * (torch.exp(inv_tail_sq) * torch.cosh(2.0 * self.skew / self.tail) + 1.0)
        )

    def rsample(
        self, sample_shape: torch.Size | list[int] | tuple[int, ...] = _EMPTY_SHAPE
    ) -> torch.Tensor:
        """Reparameterized draw: y = ξ + λ·sinh((ε − γ)/δ), ε ~ N(0, 1).

        The exact inverse of the defining transform z = γ + δ·arcsinh(u) —
        no rejection, no approximation — and differentiable in all four
        parameters (#94: a TransformedDistribution was rejected for lacking
        moments, not for its sampler; the sampler IS the inverse transform).

        Args:
            sample_shape: Leading sample dimensions.

        Returns:
            Tensor of shape sample_shape + batch_shape.
        """
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * torch.sinh((eps - self.skew) / self.tail)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """Closed-form CDF: F(y) = Φ(γ + δ·arcsinh((y − ξ)/λ)).

        z is standard normal by construction, so the CDF is the normal CDF
        of the z-score — closed form via erf (rule 4: never MC-estimate
        what has a closed form). This is what makes PIT calibration exact
        for this family.

        Args:
            value: Responses, broadcastable to the batch shape.

        Returns:
            Tensor of F(value), batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)
        z = self.skew + self.tail * torch.asinh((value - self.loc) / self.scale)
        # Φ(z) = ½(1 + erf(z/√2)) — torch.special.ndtr is the fused form.
        out: torch.Tensor = torch.special.ndtr(z)
        return out

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Pointwise log-density, fully in log-space (numerical rule 2).

        log f = log δ − log λ − ½·log 2π − ½·log1p(u²) − ½·z²
        with u = (y − ξ)/λ and z = γ + δ·arcsinh(u). ``log1p`` instead of
        ``log(1 + u²)`` per numerical rule 3.

        Args:
            value: Responses, broadcastable to the batch shape.

        Returns:
            Tensor of pointwise log p(value), batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)
        u = (value - self.loc) / self.scale
        z = self.skew + self.tail * torch.asinh(u)
        return (
            torch.log(self.tail)
            - torch.log(self.scale)
            - _HALF_LOG_2PI
            - 0.5 * torch.log1p(u * u)
            - 0.5 * z * z
        )


class JohnsonSUFamily(BaseFamily):
    """Johnson's SU response family — 4 distributional parameters (#94).

    The NAMLSS-continuity family (PRD 0002 / GitHub #84): epistemic
    uncertainty over skew and kurtosis is the paper's headline demo. Column
    order is the scipy ``johnsonsu`` order, so a fitted parameter vector
    reads directly as scipy's ``(a, b, loc, scale)``:

      - column 0: γ skew — identity link,
      - column 1: δ tailweight — ``softplus + EPS`` (numerical rule 1),
      - column 2: ξ loc — identity link,
      - column 3: λ scale — ``softplus + EPS``.

    Moment validity note: ``mean``/``variance`` carry an exp(δ⁻²) factor
    that overflows float32 for δ ≲ 0.1 — mathematically finite, but
    surfaced as honest ±inf (never NaN) per the #91 moment-conformance
    contract; see ``JohnsonSU.mean``.

    Args:
        validate_args: Passed to torch.distributions. False in training hot
            path; True in test fixtures (numerical rule 6).
    """

    param_count: int = 4

    def __init__(self, validate_args: bool = False) -> None:
        self.validate_args = bool(validate_args)

    def __call__(self, params: torch.Tensor) -> JohnsonSU:
        """Apply parameter links and return a JohnsonSU distribution.

        Args:
            params: Tensor of shape (..., 4) — columns (raw γ, raw δ,
                raw ξ, raw λ); see the class docstring for links.

        Returns:
            JohnsonSU with batch_shape (...).
        """
        skew = params[..., 0]
        # softplus enforces strict positivity; EPS floor for extreme inputs
        # (bare softplus underflows to exact 0.0 near pre-link −104 in
        # float32 — numerical rule 1 / GitHub #88).
        tail = F.softplus(params[..., 1]) + EPS
        loc = params[..., 2]
        scale = F.softplus(params[..., 3]) + EPS
        return JohnsonSU(
            skew=skew,
            tail=tail,
            loc=loc,
            scale=scale,
            validate_args=self.validate_args,
        )

    def cdf(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Johnson's SU CDF for PIT calibration (BaseFamily contract, #93).

        Unlike StudentT, no scipy detour is needed: the distribution has a
        closed-form torch ``cdf`` (Φ of the z-score). Reuses ``__call__``
        so the links stay single-sourced; eval-time only, float64 out
        (calibration metrics accumulate in float64, dtype rule).

        Args:
            params: Tensor of shape (..., 4), as in ``__call__``.
            y: Observed responses, broadcastable to the batch shape (...).

        Returns:
            Tensor of shape (...), dtype float64.
        """
        with torch.no_grad():
            return self(params).cdf(y).to(torch.float64)
