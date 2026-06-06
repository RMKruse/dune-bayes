"""Response-level predictive bands (issue 0008 / GitHub #9).

Converts a MixtureSameFamily posterior predictive into a full predictive band
(epistemic + aleatoric) by sampling K draws from the mixture and computing
tail quantiles per observation.

Distinct from effect_ribbon.py (issue 0006), which is epistemic-only and
operates on per-feature sample_effects draws. Here aleatoric uncertainty is a
response-level property and is deliberately not attributed to individual features.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_bamlss.utils import to_numpy


def predictive_quantiles(
    predictive: torch.distributions.MixtureSameFamily,
    credible_interval: float = 0.90,
    n_samples: int = 2000,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute response-level quantiles from a MixtureSameFamily predictive.

    Draws n_samples from the mixture and estimates empirical quantiles per
    observation.  The full predictive variance = epistemic (across components)
    + aleatoric (within each component family), so the resulting band is a
    proper prediction interval.

    Args:
        predictive: MixtureSameFamily with batch_shape (n,), as returned by
            BayesianNAMLSS.sample_posterior_predictive or draw_predictive.
        credible_interval: Width of the predictive band, e.g. 0.90 for 90%.
        n_samples: Number of Monte-Carlo draws used to estimate quantiles.
            Default 2000 balances accuracy and speed.
        seed: Optional integer seed for reproducible quantile estimation.
            Applied inside a forked RNG scope, so the caller's global torch
            RNG state is left untouched.

    Returns:
        Dict with keys "lo", "mid", "hi", each a float32 Tensor of shape (n,).
        "mid" is the 50th percentile (median) of the predictive.
        "lo"/"hi" are the tail quantiles defined by credible_interval.
    """
    alpha = (1.0 - credible_interval) / 2.0

    # torch.distributions has no generator= hook on .sample(), so the only way
    # to seed is via the global RNG. fork_rng restores the caller's RNG state
    # on exit — seeding here must not perturb downstream sampling.
    with torch.random.fork_rng(enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            # (n_samples, n) — K independent draws from each observation's
            # mixture.
            samples = predictive.sample([n_samples]).float()

    # Reduce over dim=0 (the n_samples axis) to get per-observation quantiles.
    lo = torch.quantile(samples, alpha, dim=0)  # (n,)
    mid = torch.quantile(samples, 0.5, dim=0)  # (n,)
    hi = torch.quantile(samples, 1.0 - alpha, dim=0)  # (n,)

    return {"lo": lo, "mid": mid, "hi": hi}


def plot_dist(
    predictive: torch.distributions.MixtureSameFamily,
    y: torch.Tensor | np.ndarray,
    credible_interval: float = 0.90,
    n_samples: int = 2000,
    ax: Optional[plt.Axes] = None,
    seed: int | None = None,
) -> plt.Axes:
    """Plot the full predictive band against actual response values.

    Observations are sorted by the predicted median so that the band is
    monotone and visually clean. Actual y values are overlaid as a scatter.

    Args:
        predictive: MixtureSameFamily with batch_shape (n,).
        y: Actual response values, shape (n,).
        credible_interval: Width of the predictive band. Default 0.90.
        n_samples: MC draws for quantile estimation. Default 2000.
        ax: Existing Axes to draw on; creates a new figure if None.
        seed: Optional seed for reproducible sampling.

    Returns:
        The matplotlib Axes with the predictive ribbon and actual-y scatter.
    """
    if ax is None:
        _, ax = plt.subplots()

    quants = predictive_quantiles(
        predictive,
        credible_interval=credible_interval,
        n_samples=n_samples,
        seed=seed,
    )
    lo = quants["lo"].numpy()
    mid = quants["mid"].numpy()
    hi = quants["hi"].numpy()

    y_np = to_numpy(y)

    # Sort by predicted median for a monotone, visually clean ribbon.
    order = np.argsort(mid)
    x_idx = np.arange(len(order))  # integer x-axis (sorted observation index)

    lo, mid, hi, y_sorted = lo[order], mid[order], hi[order], y_np[order]

    pct = int(credible_interval * 100)
    ax.fill_between(x_idx, lo, hi, alpha=0.3, label=f"{pct}% predictive band")
    ax.scatter(x_idx, y_sorted, s=10, alpha=0.6, label="observed y", zorder=3)

    return ax
