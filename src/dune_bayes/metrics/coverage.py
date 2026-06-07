"""Per-parameter quantile coverage of credible bands.

Issue 0093 / GitHub #93 — empirical coverage of central credible intervals at
nominal 50/80/90/95, per distribution parameter (PRD 0002 §24). Coverage
doctrine (CONTEXT.md): credible-band quality is MEASURED, REPORTED, and
explained — never asserted as "correct". Mean-field VI is expected to
under-cover (ADR-0001's accepted narrowness); this function quantifies that
narrowness instead of denying it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# CONTEXT.md: simulation studies report empirical coverage per distribution
# parameter at nominal 50/80/90/95.
_DEFAULT_LEVELS = (0.5, 0.8, 0.9, 0.95)


@dataclass
class QuantileCoverage:
    """Empirical coverage table: distribution parameters × nominal levels.

    Attributes:
        levels: The nominal central-interval levels, as given.
        coverage: Tensor of shape (param_count, len(levels)), float64 —
            ``coverage[p, l]`` is the fraction of observations whose true
            parameter value falls inside the central ``levels[l]`` credible
            interval of the draws for parameter ``p``.
    """

    levels: tuple[float, ...]
    coverage: torch.Tensor


def quantile_coverage(
    samples: torch.Tensor,
    truth: torch.Tensor,
    levels: tuple[float, ...] = _DEFAULT_LEVELS,
) -> QuantileCoverage:
    """Measure empirical coverage of central credible bands against truth.

    Generic tensors in, table out — consumes effect-sampler draws
    (``sample_effects``) or parameter draws against the simulation truth;
    nothing here is family-specific. Coverage is computed from sample
    quantiles of the draws, so it is invariant under monotone per-parameter
    links (raw predictor draws vs linked-parameter draws give the same table
    when truth is stated on the matching scale).

    Args:
        samples: Posterior draws, shape (T, n, param_count).
        truth: True values, shape (n, param_count), on the same scale as
            ``samples``.
        levels: Nominal central-interval levels in (0, 1). Default is the
            CONTEXT.md reporting grid (0.5, 0.8, 0.9, 0.95).

    Returns:
        QuantileCoverage with ``coverage`` of shape (param_count, len(levels)).
    """
    with torch.no_grad():
        # float64: quantile interpolation + coverage fractions are reporting
        # quantities, not hot-path tensors (dtype rule).
        x = samples.to(torch.float64)
        truth64 = truth.to(torch.float64)
        rows = []
        for level in levels:
            lo_q, hi_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0
            qs = torch.quantile(
                x, torch.tensor([lo_q, hi_q], dtype=torch.float64), dim=0
            )  # (2, n, param_count)
            covered = (truth64 >= qs[0]) & (truth64 <= qs[1])  # (n, param_count)
            rows.append(covered.to(torch.float64).mean(dim=0))  # (param_count,)
        return QuantileCoverage(
            levels=tuple(levels),
            coverage=torch.stack(rows, dim=-1),  # (param_count, len(levels))
        )
