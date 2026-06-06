"""Tests for interaction surface plots (issue 0013 / GitHub #14).

Acceptance criteria:
  - BayesianMLP(x1):BayesianMLP(x2) builds as a single joint net over both
    inputs — verified here as BayesianMLP(in_features=2) accepting (batch, 2).
  - sample_effects handles the interaction term as-is.
  - Interaction renders as a posterior-mean surface + epistemic-SD surface.

Four reference-test archetypes (CLAUDE.md):
  - Shape:       joint net output, surface_stats shapes.
  - Closed-form: mean and SD match torch.mean/torch.std references.
  - Behavior:    plot returns Figure with two Axes, each with a collection.
  - Behavior:    feature_names set as axis labels.
"""

import pytest
import torch

# ── constants ─────────────────────────────────────────────────────────────────

T = 50
N = 30
P = 2  # param_count

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def random_samples():
    """T draws from a random normal: shape [T, N, P]."""
    g = torch.Generator().manual_seed(42)
    return torch.randn(T, N, P, generator=g)


@pytest.fixture
def x1_x2():
    """(x1, x2) as regular (N,) arrays for a 2D grid."""
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(N, generator=g)
    x2 = torch.randn(N, generator=g)
    return x1, x2


# ── 1: Joint BayesianMLP accepts (batch, 2) input — shape test ────────────────


def test_joint_bayesian_mlp_shape():
    """BayesianMLP(in_features=2) with (batch, 2) input → (batch, param_count).

    A Bayesian interaction is just a BayesianMLP with a 2-feature input;
    no new machinery is needed (ADR-0005, issue 0013).
    """
    from dune_bayes.shapes.bayesian_mlp import BayesianMLP

    net = BayesianMLP(in_features=2, param_count=P, hidden_dims=[8], validate_args=True)
    x = torch.randn(N, 2)
    out = net(x)
    assert out.shape == (N, P), f"expected ({N}, {P}), got {out.shape}"


# ── 2: sample_effects handles joint net as-is — shape test ───────────────────


def test_effect_sampler_joint_net():
    """sample_effects with a joint-net key returns [T, n, param_count].

    The net receives a (n, 2) input tensor (the concatenated interaction
    features); sample_effects is agnostic to in_features.
    """
    from dune_bayes.families.normal import NormalFamily
    from dune_bayes.model import BayesianNAMLSS
    from dune_bayes.sampling.effect_sampler import sample_effects
    from dune_bayes.shapes.bayesian_mlp import BayesianMLP

    net = BayesianMLP(in_features=2, param_count=P, hidden_dims=[8], validate_args=True)
    model = BayesianNAMLSS(formula={"x1x2": net}, family=NormalFamily())

    # data["x1x2"] is shape (N, 2) — the stacked interaction inputs.
    data = {"x1x2": torch.randn(N, 2)}
    samples = sample_effects(model, data, T=T)

    assert "x1x2" in samples, "expected 'x1x2' key in sample_effects output"
    assert samples["x1x2"].shape == (T, N, P), (
        f"expected ({T}, {N}, {P}), got {samples['x1x2'].shape}"
    )


# ── 3: surface_stats keys and shapes — tracer bullet for new module ───────────


def test_surface_stats_keys_and_shape(random_samples):
    """surface_stats returns dict with 'mean' and 'sd', each [n, param_count]."""
    from dune_bayes.plotting.interaction_surface import surface_stats

    result = surface_stats(random_samples)

    assert "mean" in result, "missing key 'mean'"
    assert "sd" in result, "missing key 'sd'"
    assert result["mean"].shape == (N, P), (
        f"mean shape {result['mean'].shape} != ({N}, {P})"
    )
    assert result["sd"].shape == (N, P), f"sd shape {result['sd'].shape} != ({N}, {P})"


# ── 4: surface_stats SD is non-negative ──────────────────────────────────────


def test_surface_stats_sd_nonnegative(random_samples):
    """Epistemic SD is non-negative everywhere (std is always ≥ 0)."""
    from dune_bayes.plotting.interaction_surface import surface_stats

    result = surface_stats(random_samples)
    assert result["sd"].min().item() >= 0.0, (
        f"SD has negative values: min={result['sd'].min().item():.4f}"
    )


# ── 5: surface_stats mean matches torch.mean reference ───────────────────────


def test_surface_stats_mean_correctness(random_samples):
    """surface_stats mean equals samples.float().mean(dim=0).

    atol=1e-5: pure float32 arithmetic, no MC noise.
    """
    from dune_bayes.plotting.interaction_surface import surface_stats

    result = surface_stats(random_samples)
    expected = random_samples.float().mean(dim=0)
    max_delta = (result["mean"] - expected).abs().max().item()
    assert torch.allclose(result["mean"], expected, atol=1e-5), (
        f"mean deviates from torch.mean: max Δ={max_delta:.2e}"
    )


# ── 6: surface_stats SD matches torch.std reference ──────────────────────────


def test_surface_stats_sd_correctness(random_samples):
    """surface_stats SD equals samples.float().std(dim=0) (Bessel correction).

    atol=1e-5: pure float32 arithmetic, no MC noise.
    """
    from dune_bayes.plotting.interaction_surface import surface_stats

    result = surface_stats(random_samples)
    expected = random_samples.float().std(dim=0)
    max_delta = (result["sd"] - expected).abs().max().item()
    assert torch.allclose(result["sd"], expected, atol=1e-5), (
        f"sd deviates from torch.std: max Δ={max_delta:.2e}"
    )


# ── 7: plot_interaction_surface returns a Figure ──────────────────────────────


def test_plot_interaction_surface_returns_figure(random_samples, x1_x2):
    """plot_interaction_surface returns a matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dune_bayes.plotting.interaction_surface import plot_interaction_surface

    x1, x2 = x1_x2
    fig = plot_interaction_surface(random_samples, x1, x2)
    assert isinstance(fig, plt.Figure)


# ── 8: Figure has exactly 2 Axes (mean + SD panels) ─────────────────────────


def test_plot_has_two_axes(random_samples, x1_x2):
    """Returned Figure has exactly 2 Axes: one for mean, one for SD."""
    import matplotlib

    matplotlib.use("Agg")

    from dune_bayes.plotting.interaction_surface import plot_interaction_surface

    x1, x2 = x1_x2
    fig = plot_interaction_surface(random_samples, x1, x2)
    assert len(fig.axes) == 2, f"expected 2 axes, got {len(fig.axes)}"


# ── 9: Each Axes contains a collection from tricontourf ──────────────────────


def test_plot_axes_have_collections(random_samples, x1_x2):
    """Each Axes contains at least one collection (from tricontourf fill).

    tricontourf produces PathCollection/PolyCollection entries in ax.collections.
    """
    import matplotlib

    matplotlib.use("Agg")

    from dune_bayes.plotting.interaction_surface import plot_interaction_surface

    x1, x2 = x1_x2
    fig = plot_interaction_surface(random_samples, x1, x2)
    for i, ax in enumerate(fig.axes):
        assert len(ax.collections) >= 1, (
            f"Axes[{i}] has no collections — tricontourf did not render"
        )


# ── 10: Optional feature_names are used as axis labels ───────────────────────


def test_plot_feature_names_as_labels(random_samples, x1_x2):
    """When feature_names=(name1, name2) are given, axes use them as x/y labels.

    Both axes (mean + SD) share the same 2D coordinate system, so both get
    the same feature name labels.
    """
    import matplotlib

    matplotlib.use("Agg")

    from dune_bayes.plotting.interaction_surface import plot_interaction_surface

    x1, x2 = x1_x2
    fig = plot_interaction_surface(
        random_samples, x1, x2, feature_names=("income", "age")
    )
    for ax in fig.axes:
        assert ax.get_xlabel() == "income", (
            f"x-label expected 'income', got {ax.get_xlabel()!r}"
        )
        assert ax.get_ylabel() == "age", (
            f"y-label expected 'age', got {ax.get_ylabel()!r}"
        )
