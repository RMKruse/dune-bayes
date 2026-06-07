"""Probability integral transform of the posterior predictive.

Issue 0093 / GitHub #93 — PIT_i = (1/T) Σ_t F(y_i | θ_t): the predictive
(mixture) CDF at the observed response, averaged over coherent posterior draws
(PRD 0002 §23). Uniform PIT values diagnose calibration; the CDFs come from
each family's scipy-backed ``cdf`` (eval-time only, no gradient path).
"""

from __future__ import annotations

import torch

from dune_bayes.families.base import BaseFamily


def pit(
    family: BaseFamily,
    summed_samples: torch.Tensor,
    y: torch.Tensor,
    *,
    randomized: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute per-observation PIT values from posterior predictive draws.

    Consumes the ``summed_samples`` produced by ``draw_predictive`` (the
    draw-once-score-once pattern, GitHub #68): the uniform T-draw mixture has
    CDF equal to the mean of the component CDFs, so averaging F(y | θ_t) over
    draws IS the predictive CDF at y — closed form, no MC re-sampling.

    For discrete (integer) support, plain PIT is non-uniform even under a
    perfect model (F only takes the jump-top values); pass ``randomized=True``
    to get u·F(y) + (1−u)·F(y−1) with u ~ U(0, 1), which spreads each jump's
    mass evenly and restores exact uniformity in distribution.

    Args:
        family: Response family providing the scipy-backed ``cdf``.
        summed_samples: Summed-predictor draws, shape (T, n, param_count),
            as returned by ``draw_predictive``.
        y: Observed responses, shape (n,).
        randomized: Apply the randomized PIT for discrete (integer) support.
            Caller-stated intent — never auto-detected from the family.
        seed: Seeds the u ~ U(0, 1) draw (own ``torch.Generator``, never the
            global stream) for reproducible randomized PIT. None draws from
            the global stream. Ignored when ``randomized=False``.

    Returns:
        Tensor of shape (n,), dtype float64, with values in [0, 1].
    """
    with torch.no_grad():
        # Mixture CDF at y: mean of component CDFs over the T draws.
        f_y = family.cdf(summed_samples, y).mean(dim=0)  # (n,) float64
        if not randomized:
            return f_y
        # Randomized PIT (discrete support): u·F(y) + (1−u)·F(y−1). One u per
        # observation, applied AFTER mixture averaging — the jump being
        # randomized is the predictive's jump at y, not each component's.
        f_y_prev = family.cdf(summed_samples, y - 1).mean(dim=0)
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        u = torch.rand(y.shape, generator=generator, dtype=torch.float64)
        return u * f_y + (1.0 - u) * f_y_prev
