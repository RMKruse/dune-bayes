"""Centered epistemic effect ribbons (issue 0006 / GitHub #7).

Converts sample_effects draws [T, n, param_count] into credible ribbons by:
  1. Optionally mean-centering each of the T draws over the n data points
     (isolates shape uncertainty from overall level — CONTEXT.md "Effect plot").
  2. Computing lo/mid/hi quantiles across T.

Default credible interval: 90% (configurable). Centering is on by default,
flag-able for uncentered mode (deviates from NAMpy's uncentered default).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_bamlss.utils import to_numpy


def ribbon_quantiles(
    samples: torch.Tensor,
    credible_interval: float = 0.90,
    center: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute credible ribbon quantiles from posterior contribution draws.

    Args:
        samples: Tensor[T, n, param_count] from sample_effects for one feature.
        credible_interval: Width of the credible band, e.g. 0.90 for 90%.
        center: If True, mean-center each draw over n before quantile
            computation. Isolates shape uncertainty from the overall level
            (epistemic ribbon; CONTEXT.md "Effect plot vs response plot").

    Returns:
        Dict with keys "lo", "mid", "hi", each Tensor[n, param_count].
        "mid" is the 50th percentile (median) across T.
        "lo"/"hi" are the tail quantiles defined by credible_interval.
    """
    if center:
        # Subtract each draw's mean over n — zero-centers every curve.
        # keepdim preserves [T, 1, param_count] for broadcast.
        samples = samples - samples.mean(dim=1, keepdim=True)

    alpha = (1.0 - credible_interval) / 2.0
    q_lo = alpha
    q_hi = 1.0 - alpha

    # torch.quantile(input, q, dim) reduces over dim=0 (the T axis).
    lo = torch.quantile(samples.float(), q_lo, dim=0)  # [n, param_count]
    mid = torch.quantile(samples.float(), 0.5, dim=0)  # [n, param_count]
    hi = torch.quantile(samples.float(), q_hi, dim=0)  # [n, param_count]

    return {"lo": lo, "mid": mid, "hi": hi}


def plot_effect_ribbon(
    samples: torch.Tensor,
    x_values: np.ndarray | torch.Tensor,
    credible_interval: float = 0.90,
    center: bool = True,
    param_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    feature_name: Optional[str] = None,
) -> plt.Axes:
    """Plot centered epistemic credible ribbon for one feature.

    Args:
        samples: Tensor[T, n, param_count] from sample_effects for one feature.
        x_values: Array[n] of x-axis values (sorted for a smooth ribbon).
        credible_interval: Width of the credible band. Default 0.90.
        center: Mean-center each draw before computing quantiles. Default True.
        param_idx: Which distributional parameter to plot (0 = location).
        ax: Existing Axes to draw on; creates a new figure if None.
        feature_name: Optional label for the x-axis / title.

    Returns:
        The matplotlib Axes with the ribbon drawn.
    """
    if ax is None:
        _, ax = plt.subplots()

    quants = ribbon_quantiles(
        samples, credible_interval=credible_interval, center=center
    )
    lo = quants["lo"][:, param_idx].numpy()
    mid = quants["mid"][:, param_idx].numpy()
    hi = quants["hi"][:, param_idx].numpy()

    x = to_numpy(x_values)

    # Sort along x for a smooth visual (sample_effects gives unordered n rows).
    order = np.argsort(x)
    x, lo, mid, hi = x[order], lo[order], mid[order], hi[order]

    pct = int(credible_interval * 100)
    ax.fill_between(x, lo, hi, alpha=0.3, label=f"{pct}% credible ribbon")
    ax.plot(x, mid, label="median")

    if feature_name is not None:
        ax.set_xlabel(feature_name)

    return ax
