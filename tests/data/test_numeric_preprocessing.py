"""Tests for DataModule numeric preprocessing (issue 0023 / GitHub #50).

Acceptance criteria:
  - Standardize default: mean ≈ 0 / sd ≈ 1 on training columns
  - Min-max scaling selectable per feature
  - transform() reuses train statistics without refitting
  - inverse_transform(transform(x)) round-trips within float tolerance
  - End-to-end: plot grid expressible on original feature scale
"""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from neural_bamlss.data import DataModule

# ── constants ─────────────────────────────────────────────────────────────────

N_TRAIN = 64
N_TEST = 16
SEED = 42


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dfs():
    """Train and held-out DataFrames with two numeric features."""
    rng = np.random.default_rng(SEED)
    # Deliberately non-zero-mean, non-unit-variance to make scaling visible.
    x1_train = rng.normal(loc=5.0, scale=2.0, size=N_TRAIN)
    x2_train = rng.uniform(0.0, 100.0, size=N_TRAIN)
    y_train = 3.0 * x1_train + 0.01 * x2_train + rng.normal(size=N_TRAIN)

    x1_test = rng.normal(loc=5.0, scale=2.0, size=N_TEST)
    x2_test = rng.uniform(0.0, 100.0, size=N_TEST)
    y_test = 3.0 * x1_test + 0.01 * x2_test + rng.normal(size=N_TEST)

    df_train = pd.DataFrame({"x1": x1_train, "x2": x2_train, "y": y_train})
    df_test = pd.DataFrame({"x1": x1_test, "x2": x2_test, "y": y_test})
    return df_train, df_test


# ── 1. tracer bullet: standardize default ─────────────────────────────────────


def test_standardize_default_mean_zero_std_one(dfs):
    """Standardized training columns have mean ≈ 0 and std ≈ 1.

    Tolerance 1e-5: float32 arithmetic on N=64 observations — not MC noise.
    """
    df_train, _ = dfs
    dm = DataModule(df_train, response="y", numeric_scaling={})

    for name in ("x1", "x2"):
        col = dm.features[name].squeeze(-1)
        # atol=1e-5: float32 rounding across N=64 observations
        assert col.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert col.std(correction=0).item() == pytest.approx(1.0, abs=1e-5)


# ── 2. feature shape unchanged after scaling ──────────────────────────────────


def test_feature_shape_preserved_after_scaling(dfs):
    df_train, _ = dfs
    dm = DataModule(df_train, response="y", numeric_scaling={})

    for name in ("x1", "x2"):
        assert dm.features[name].shape == (N_TRAIN, 1)
        assert dm.features[name].dtype == torch.float32


# ── 3. min-max scaling per feature ────────────────────────────────────────────


def test_minmax_scaling_puts_feature_in_unit_interval(dfs):
    """numeric_scaling={"x2": "minmax"} → x2 in [0, 1]; x1 stays standardized."""
    df_train, _ = dfs
    dm = DataModule(df_train, response="y", numeric_scaling={"x2": "minmax"})

    x2 = dm.features["x2"].squeeze(-1)
    # atol=1e-6: exact min/max with float32 — the endpoints are pinned by construction
    assert x2.min().item() == pytest.approx(0.0, abs=1e-6)
    assert x2.max().item() == pytest.approx(1.0, abs=1e-6)

    # x1 still standardized (default method for unlisted features)
    x1 = dm.features["x1"].squeeze(-1)
    assert x1.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert x1.std(correction=0).item() == pytest.approx(1.0, abs=1e-5)


# ── 4. transform() reuses train statistics — sklearn cross-check ───────────────


def test_transform_held_out_matches_sklearn_reference(dfs):
    """transform() applies train statistics to held-out data without refitting.

    Independent cross-check: sklearn StandardScaler fit on train, applied to test.
    Tolerance 1e-5: float32 vs float64 sklearn conversion — not MC noise.
    """
    df_train, df_test = dfs
    dm = DataModule(df_train, response="y", numeric_scaling={})
    scaled = dm.transform(df_test)

    # Independent sklearn reference
    sk = StandardScaler()
    sk.fit(df_train[["x1", "x2"]].to_numpy())
    ref = sk.transform(df_test[["x1", "x2"]].to_numpy()).astype(np.float32)

    for i, name in enumerate(("x1", "x2")):
        result = scaled[name].squeeze(-1).numpy()
        # atol=1e-5: float32 vs sklearn float64→float32 conversion
        np.testing.assert_allclose(result, ref[:, i], atol=1e-5)


# ── 5. inverse_transform round-trip ───────────────────────────────────────────


def test_inverse_transform_round_trips_within_float32_tolerance(dfs):
    """inverse_transform(transform(x)) recovers x within float32 precision.

    atol=1e-5: two float32 linear ops (scale + unscale) accumulate ~1 ULP error.
    """
    df_train, df_test = dfs
    dm = DataModule(df_train, response="y", numeric_scaling={"x2": "minmax"})
    scaled = dm.transform(df_test)

    for name in ("x1", "x2"):
        original = torch.tensor(df_test[name].to_numpy(), dtype=torch.float32)
        recovered = dm.inverse_transform(name, scaled[name].squeeze(-1))
        # atol=1e-5: two float32 linear operations
        assert torch.allclose(recovered, original, atol=1e-5)


# ── 6. end-to-end: plot grid on original feature scale ────────────────────────


def test_plot_grid_inverse_transform_recovers_original_scale(dfs):
    """A linspace grid, scaled then inverted, matches the original grid values.

    This is the end-to-end plot-axis use case: a regular grid in original units
    is passed through the model (scaled) and the x-axis is rendered by inverting
    back to original units.
    """
    df_train, _ = dfs
    dm = DataModule(df_train, response="y", numeric_scaling={})

    # Build a plot grid over the observed range of x1
    x1_raw = df_train["x1"].to_numpy(dtype=np.float32)
    grid_vals = np.linspace(x1_raw.min(), x1_raw.max(), 50, dtype=np.float32)
    grid_df = pd.DataFrame({"x1": grid_vals, "x2": np.zeros(50, dtype=np.float32)})

    scaled_grid = dm.transform(grid_df)
    recovered = dm.inverse_transform("x1", scaled_grid["x1"].squeeze(-1))

    # atol=1e-5: float32 round-trip over a regular grid
    np.testing.assert_allclose(
        recovered.numpy(),
        grid_vals,
        atol=1e-5,
        err_msg="inverse_transform must recover original grid values for plot axes",
    )
