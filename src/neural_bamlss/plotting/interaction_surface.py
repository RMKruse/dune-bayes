"""Interaction surface plots (issue 0013 / GitHub #14).

Renders a two-panel figure for a Bayesian interaction term:
  - Posterior-mean surface (mean over T weight draws).
  - Epistemic-SD surface  (std over T weight draws).

A 1D ribbon (effect_ribbon.py) does not generalize to two-feature interactions;
these surfaces are the 2D equivalent (ADR-0005).

Both surfaces use matplotlib.tri.tricontourf, which handles the irregular
(x1, x2) scatter that sample_effects returns without requiring a regular grid.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def surface_stats(
    samples: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute posterior mean and epistemic SD from T weight draws.

    Args:
        samples: Tensor[T, n, param_count] from sample_effects for an
            interaction feature (in_features=2 BayesianMLP).

    Returns:
        Dict with keys:
          "mean": Tensor[n, param_count] — posterior mean over T draws.
          "sd":   Tensor[n, param_count] — epistemic std over T draws
                  (Bessel-corrected sample std, always ≥ 0).
    """
    f = samples.float()
    return {
        "mean": f.mean(dim=0),  # [n, param_count]
        "sd": f.std(dim=0),  # [n, param_count]
    }


def plot_interaction_surface(
    samples: torch.Tensor,
    x1_values: np.ndarray | torch.Tensor,
    x2_values: np.ndarray | torch.Tensor,
    param_idx: int = 0,
    feature_names: Optional[tuple[str, str]] = None,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    """Plot posterior-mean surface and epistemic-SD surface for an interaction.

    Args:
        samples: Tensor[T, n, param_count] from sample_effects for an
            interaction feature.
        x1_values: Array[n] of first-feature values (x-axis).
        x2_values: Array[n] of second-feature values (y-axis).
        param_idx: Which distributional parameter to plot (0 = location).
        feature_names: Optional (name1, name2) used as x/y axis labels on
            both panels.
        fig: Existing Figure with 2 Axes to draw on; creates a new 1×2
            figure if None.

    Returns:
        Figure with two Axes: axes[0] = posterior-mean surface,
        axes[1] = epistemic-SD surface.
    """
    stats = surface_stats(samples)
    mean_vals = stats["mean"][:, param_idx].numpy()  # [n]
    sd_vals = stats["sd"][:, param_idx].numpy()  # [n]

    if isinstance(x1_values, torch.Tensor):
        x1 = x1_values.numpy()
    else:
        x1 = np.asarray(x1_values)

    if isinstance(x2_values, torch.Tensor):
        x2 = x2_values.numpy()
    else:
        x2 = np.asarray(x2_values)

    if fig is None:
        fig, _ = plt.subplots(1, 2, figsize=(10, 4))

    axes = fig.axes
    ax_mean, ax_sd = axes[0], axes[1]

    ax_mean.tricontourf(x1, x2, mean_vals, levels=12)
    ax_mean.set_title("Posterior mean")

    ax_sd.tricontourf(x1, x2, sd_vals, levels=12)
    ax_sd.set_title("Epistemic SD")

    if feature_names is not None:
        name1, name2 = feature_names
        for ax in (ax_mean, ax_sd):
            ax.set_xlabel(name1)
            ax.set_ylabel(name2)

    return fig
