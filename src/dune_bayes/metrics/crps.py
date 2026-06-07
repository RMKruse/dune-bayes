"""Fair sample-based CRPS (issue 0092 / GitHub #92, PRD 0002 §22).

One generic proper scoring rule for the T-draw posterior predictive,
family-agnostic by construction (samples in, scores out) so comparison tables
never depend on per-family closed forms: the fair (unbiased) sample estimator

    CRPS(x₁..x_M, y) = (1/M) Σᵢ |xᵢ − y|  −  1/(2M(M−1)) Σᵢⱼ |xᵢ − xⱼ|,

with the pairwise term computed via the sort-based O(M log M) identity (the
naive O(M²) pairwise matrix is never materialized) and accumulated in float64
(CLAUDE.md dtype rule: metric accumulation is where float32 bites).
"""

from __future__ import annotations

import torch


def crps(
    samples: torch.Tensor,
    y: torch.Tensor,
    reduce: str = "none",
) -> torch.Tensor:
    """Fair (unbiased) sample-based CRPS.

    Args:
        samples: Predictive draws, shape (M, n) — e.g.
            ``draws.predictive.sample((M,))``. Any real dtype (float32 draws,
            integer counts); promoted to float64 before accumulation.
        y: Observed responses, shape (n,).
        reduce: "none" (default) for per-observation scores, "mean" for the
            scalar mean — the comparison-table reducer.

    Returns:
        Per-observation CRPS, shape (n,), float64 — or its scalar mean
        under ``reduce="mean"``.

    Raises:
        ValueError: If ``reduce`` is not "none" or "mean".
    """
    if reduce not in ("none", "mean"):
        raise ValueError(f'reduce must be "none" or "mean", got {reduce!r}')
    x = samples.to(torch.float64)
    y64 = y.to(torch.float64)
    m = x.shape[0]

    term1 = (x - y64).abs().mean(dim=0)  # (n,)

    # Sort-based pairwise identity: with x_(1) ≤ … ≤ x_(M) per column,
    #   Σᵢⱼ |xᵢ − xⱼ| = 2 Σₖ (2k − M − 1) x_(k)   (k = 1..M),
    # because the k-th order statistic appears with +1 against the k−1 smaller
    # values and −1 against the M−k larger ones. O(M log M), never O(M²).
    x_sorted, _ = x.sort(dim=0)
    k = torch.arange(1, m + 1, dtype=torch.float64).unsqueeze(-1)  # (M, 1)
    pair_sum = ((2.0 * k - m - 1.0) * x_sorted).sum(dim=0)  # = ½ Σᵢⱼ|xᵢ−xⱼ|
    term2 = pair_sum / (m * (m - 1))

    per_obs = term1 - term2
    return per_obs.mean() if reduce == "mean" else per_obs
