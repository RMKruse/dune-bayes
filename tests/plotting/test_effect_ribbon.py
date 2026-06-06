"""Tests for centered epistemic effect ribbons (issue 0006 / GitHub #7).

Four reference-test archetypes (CLAUDE.md):
  - Shape:        ribbon_quantiles output keys and tensor shapes.
  - Closed-form:  centering correctness, interval bounds, mid=median.
  - Behavior:     plot_effect_ribbon returns Axes with ribbon + line.
  - Behavior:     uncentered mode passes through raw quantiles.
"""

import pytest
import torch

from dune_bayes.plotting import plot_effect_ribbon, ribbon_quantiles

# ── constants ─────────────────────────────────────────────────────────────────

T = 50
N = 32
P = 2  # param_count (e.g. Normal: mu, sigma)

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def uniform_samples():
    """T draws that are all identical and non-zero: shape [T, N, P]."""
    return torch.ones(T, N, P) * 3.0


@pytest.fixture
def random_samples():
    """T draws from a random normal: shape [T, N, P]."""
    g = torch.Generator().manual_seed(7)
    return torch.randn(T, N, P, generator=g)


# ── 1: Shape — tracer bullet ──────────────────────────────────────────────────


def test_ribbon_quantiles_output_shape(random_samples):
    """lo/mid/hi each have shape [n, param_count]."""
    result = ribbon_quantiles(random_samples)
    for key in ("lo", "mid", "hi"):
        assert key in result, f"missing key {key!r}"
        assert result[key].shape == (N, P), (
            f"{key} shape {result[key].shape} != ({N}, {P})"
        )


# ── 2: Centering — constant non-zero draws become zero ────────────────────────


def test_centering_zeros_constant_draws(uniform_samples):
    """Constant draws (all 3.0) centered → lo=mid=hi=0 exactly.

    A draw that is constant over n has mean 3.0; subtracting it yields 0.
    All T quantiles are then 0, so the ribbon collapses to zero.
    atol=1e-6: pure float32 arithmetic, no MC noise.
    """
    result = ribbon_quantiles(uniform_samples, center=True)
    for key in ("lo", "mid", "hi"):
        assert result[key].abs().max().item() < 1e-6, (
            f"{key} not zero after centering constant draws: "
            f"max={result[key].abs().max().item():.2e}"
        )


# ── 3: Uncentered mode — constant draws stay at their value ───────────────────


def test_uncentered_preserves_constant_draws(uniform_samples):
    """Constant draws (all 3.0) with center=False → lo=mid=hi=3.0 exactly.

    No centering applied, so all T quantiles equal the constant value.
    atol=1e-6: pure float32, no MC noise.
    """
    result = ribbon_quantiles(uniform_samples, center=False)
    for key in ("lo", "mid", "hi"):
        residual = (result[key] - 3.0).abs().max().item()
        assert residual < 1e-6, (
            f"{key} deviates from 3.0 in uncentered mode: max Δ={residual:.2e}"
        )


# ── 4: Interval bounds — lo/hi are correct tail quantiles ────────────────────


def test_interval_bounds_90(random_samples):
    """90% CI: lo=5th percentile, hi=95th percentile across T, per (n, p).

    Checked against torch.quantile directly; atol=1e-5 for float32 rounding.
    """
    result = ribbon_quantiles(random_samples, credible_interval=0.90, center=False)
    expected_lo = torch.quantile(random_samples.float(), 0.05, dim=0)
    expected_hi = torch.quantile(random_samples.float(), 0.95, dim=0)
    assert torch.allclose(result["lo"], expected_lo, atol=1e-5), "lo != 5th percentile"
    assert torch.allclose(result["hi"], expected_hi, atol=1e-5), "hi != 95th percentile"


def test_interval_bounds_50(random_samples):
    """50% CI: lo=25th percentile, hi=75th percentile across T, per (n, p)."""
    result = ribbon_quantiles(random_samples, credible_interval=0.50, center=False)
    expected_lo = torch.quantile(random_samples.float(), 0.25, dim=0)
    expected_hi = torch.quantile(random_samples.float(), 0.75, dim=0)
    assert torch.allclose(result["lo"], expected_lo, atol=1e-5), "lo != 25th percentile"
    assert torch.allclose(result["hi"], expected_hi, atol=1e-5), "hi != 75th percentile"


# ── 5: Mid is median (50th percentile), not mean ──────────────────────────────


def test_mid_is_median_not_mean(random_samples):
    """mid equals torch.quantile(..., 0.5) — the median, not the arithmetic mean.

    For a skewed distribution of T draws, median ≠ mean. We construct a skewed
    tensor to ensure the two differ, then verify mid matches the median.
    atol=1e-5: float32 rounding in torch.quantile.
    """
    # Skew by squaring: values in [0, T) → heavy right tail, median < mean.
    skewed = torch.arange(T * N * P, dtype=torch.float32).reshape(T, N, P) ** 2
    result = ribbon_quantiles(skewed, center=False)
    expected_median = torch.quantile(skewed.float(), 0.5, dim=0)
    expected_mean = skewed.mean(dim=0)
    # Sanity: median and mean differ for this skewed input.
    assert not torch.allclose(expected_median, expected_mean, atol=1.0), (
        "test setup error: median and mean unexpectedly equal"
    )
    assert torch.allclose(result["mid"], expected_median, atol=1e-5), (
        "mid is not the median"
    )


# ── 6: plot_effect_ribbon returns Axes ────────────────────────────────────────


def test_plot_returns_axes(random_samples):
    """plot_effect_ribbon returns a matplotlib Axes object."""
    import matplotlib

    matplotlib.use("Agg")  # headless — no display needed
    import matplotlib.pyplot as plt

    x = torch.linspace(-1, 1, N)
    ax = plot_effect_ribbon(random_samples, x)
    assert isinstance(ax, plt.Axes)


# ── 7: Axes contains ribbon (PolyCollection) and median line ─────────────────


def test_plot_contains_ribbon_and_line(random_samples):
    """plot_effect_ribbon draws a fill_between ribbon and a median line.

    fill_between → PolyCollection; median line → Line2D.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    x = torch.linspace(-1, 1, N)
    ax = plot_effect_ribbon(random_samples, x)

    poly_collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
    lines = [ln for ln in ax.lines if isinstance(ln, Line2D)]

    assert len(poly_collections) >= 1, "no fill_between ribbon found"
    assert len(lines) >= 1, "no median line found"


# ── 8: plot_effect_ribbon uncentered mode ─────────────────────────────────────


def test_plot_uncentered_differs_from_centered(uniform_samples):
    """center=False produces a different (non-zero) ribbon than center=True.

    For constant draws of 3.0:
      - center=True  → ribbon collapses to 0 (already tested in unit layer).
      - center=False → ribbon sits at 3.0.
    The median line's y-data should reflect this.
    """
    import matplotlib

    matplotlib.use("Agg")

    x = torch.linspace(-1, 1, N)

    ax_centered = plot_effect_ribbon(uniform_samples, x, center=True)
    ax_uncentered = plot_effect_ribbon(uniform_samples, x, center=False)

    mid_centered = ax_centered.lines[0].get_ydata().mean()
    mid_uncentered = ax_uncentered.lines[0].get_ydata().mean()

    assert abs(mid_centered) < 1e-5, f"centered median not ~0: {mid_centered:.4f}"
    assert abs(mid_uncentered - 3.0) < 1e-5, (
        f"uncentered median not ~3.0: {mid_uncentered:.4f}"
    )
