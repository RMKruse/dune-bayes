"""Tests for response-level predictive bands (issue 0008 / GitHub #9).

Four reference-test archetypes (CLAUDE.md):
  - Shape:       predictive_quantiles output keys and tensor shapes.
  - Reference:   lo/hi are correct tail quantiles; mid is the median.
  - Behavior:    credible interval is configurable; 50% CI narrower than 90%.
  - Behavior:    plot_dist returns Axes with ribbon (PolyCollection) + scatter.
"""

import pytest
import torch

# ── constants ──────────────────────────────────────────────────────────────────

N = 32  # observations
T = 20  # mixture components (posterior weight draws)

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def predictive():
    """MixtureSameFamily for N observations, T components per observation.

    Each component is a distinct Normal drawn from N(0,1) / LogNormal scale, so
    the mixture is non-degenerate (has real epistemic + aleatoric spread).
    batch_shape = (N,), event_shape = ().
    """
    g = torch.Generator().manual_seed(42)
    locs = torch.randn(N, T, generator=g)  # (N, T) — loc per obs/component
    scales = torch.abs(torch.randn(N, T, generator=g)) + 0.5  # (N, T) positive
    mix = torch.distributions.Categorical(logits=torch.zeros(N, T))
    component = torch.distributions.Normal(locs, scales)
    return torch.distributions.MixtureSameFamily(mix, component)


@pytest.fixture
def y():
    """Response tensor of shape (N,)."""
    g = torch.Generator().manual_seed(7)
    return torch.randn(N, generator=g)


# ── 1: Shape — tracer bullet ───────────────────────────────────────────────────


def test_predictive_quantiles_output_shape(predictive):
    """lo/mid/hi each have shape (n,) — one scalar quantile per observation."""
    from neural_bamlss.plotting import predictive_quantiles

    result = predictive_quantiles(predictive, seed=0)
    for key in ("lo", "mid", "hi"):
        assert key in result, f"missing key {key!r}"
        assert result[key].shape == (N,), f"{key} shape {result[key].shape} != ({N},)"


# ── 2: lo/hi are correct tail quantiles ───────────────────────────────────────


def test_interval_bounds_90(predictive):
    """90% CI: lo = 5th percentile, hi = 95th percentile of predictive samples.

    Both the reference and the implementation draw 5000 samples from seed=0,
    so they produce identical sample tensors → identical quantiles.
    atol=1e-5: float32 rounding in torch.quantile interpolation.
    """
    from neural_bamlss.plotting import predictive_quantiles

    # Reference: draw 5000 samples with seed=0.
    torch.manual_seed(0)
    ref_samples = predictive.sample([5000]).float()  # (5000, N)
    ref_lo = torch.quantile(ref_samples, 0.05, dim=0)  # (N,)
    ref_hi = torch.quantile(ref_samples, 0.95, dim=0)  # (N,)

    # Same seed, same n_samples → identical sample tensor → identical quantiles.
    result = predictive_quantiles(
        predictive, credible_interval=0.90, n_samples=5000, seed=0
    )

    assert torch.allclose(result["lo"], ref_lo, atol=1e-5), (
        f"lo != 5th percentile; max|Δ|={(result['lo'] - ref_lo).abs().max():.2e}"
    )
    assert torch.allclose(result["hi"], ref_hi, atol=1e-5), (
        f"hi != 95th percentile; max|Δ|={(result['hi'] - ref_hi).abs().max():.2e}"
    )


# ── 3: Mid is median (50th percentile), not mean ──────────────────────────────


def test_mid_is_median_not_mean(predictive):
    """mid equals the 50th percentile of predictive samples, not the mean.

    Both reference and implementation draw 5000 samples from seed=0 →
    identical tensors → exact comparison (atol=1e-5 for float32 rounding).
    """
    from neural_bamlss.plotting import predictive_quantiles

    # Reference with seed=0 and n_samples=5000.
    torch.manual_seed(0)
    ref_samples = predictive.sample([5000]).float()  # (5000, N)
    ref_median = torch.quantile(ref_samples, 0.5, dim=0)  # (N,)
    ref_mean = ref_samples.mean(dim=0)  # (N,)

    # Sanity: the two should differ for a non-symmetric mixture.
    assert not torch.allclose(ref_median, ref_mean, atol=0.01), (
        "test setup error: median and mean unexpectedly equal"
    )

    # Same seed + same n_samples → identical mid.
    result = predictive_quantiles(predictive, n_samples=5000, seed=0)
    assert torch.allclose(result["mid"], ref_median, atol=1e-5), (
        f"mid is not the median; max|Δ|={(result['mid'] - ref_median).abs().max():.2e}"
    )


# ── 4: Credible interval is configurable ──────────────────────────────────────


def test_credible_interval_configurable(predictive):
    """50% CI produces narrower bands than 90% CI.

    Both are drawn with the same seed so the sample is identical; only the
    quantile levels differ.  The 50% band must be strictly narrower in all
    obs: hi_50 < hi_90 and lo_50 > lo_90 everywhere.
    """
    from neural_bamlss.plotting import predictive_quantiles

    r90 = predictive_quantiles(predictive, credible_interval=0.90, seed=0)
    r50 = predictive_quantiles(predictive, credible_interval=0.50, seed=0)

    assert (r50["hi"] < r90["hi"]).all(), "50% upper bound not below 90% upper bound"
    assert (r50["lo"] > r90["lo"]).all(), "50% lower bound not above 90% lower bound"


# ── 5: plot_dist returns Axes ──────────────────────────────────────────────────


def test_plot_dist_returns_axes(predictive, y):
    """plot_dist returns a matplotlib Axes object."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neural_bamlss.plotting import plot_dist

    ax = plot_dist(predictive, y, seed=0)
    assert isinstance(ax, plt.Axes)


# ── 6: Axes contains ribbon (PolyCollection) and scatter ──────────────────────


def test_plot_dist_contains_ribbon_and_scatter(predictive, y):
    """plot_dist draws a fill_between ribbon and a scatter of actual y values.

    fill_between → PolyCollection; actual-y scatter → PathCollection.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection, PolyCollection

    from neural_bamlss.plotting import plot_dist

    ax = plot_dist(predictive, y, seed=0)

    poly = [c for c in ax.collections if isinstance(c, PolyCollection)]
    scatter = [c for c in ax.collections if isinstance(c, PathCollection)]

    assert len(poly) >= 1, "no fill_between ribbon found"
    assert len(scatter) >= 1, "no scatter of actual y values found"


# ── 7: plot_dist accepts existing Axes ────────────────────────────────────────


def test_plot_dist_accepts_existing_axes(predictive, y):
    """plot_dist draws onto a provided Axes and returns the same object."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neural_bamlss.plotting import plot_dist

    _, ax_in = plt.subplots()
    ax_out = plot_dist(predictive, y, ax=ax_in, seed=0)
    assert ax_out is ax_in
