"""Tests for DataModule state round-trip (issue 0025 / GitHub #52).

Round-trip archetype: save_state() → load_state() gives a DataModule
whose transform() outputs are bit-identical to the original (max|Δ| == 0).
"""

import pandas as pd
import pytest
import torch

from neural_bamlss.data import DataModule
from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def df_numeric():
    rng = torch.Generator().manual_seed(0)
    x1 = torch.randn(N_OBS, generator=rng)
    x2 = torch.randn(N_OBS, generator=rng)
    y = 2.0 * x1 - x2 + 0.1 * torch.randn(N_OBS, generator=rng)
    return pd.DataFrame({"x1": x1.numpy(), "x2": x2.numpy(), "y": y.numpy()})


@pytest.fixture
def df_mixed():
    rng = torch.Generator().manual_seed(1)
    x1 = torch.randn(N_OBS, generator=rng)
    cats = ["a", "b", "c"]
    x_cat = [cats[i % 3] for i in range(N_OBS)]
    y = x1 + 0.1 * torch.randn(N_OBS, generator=rng)
    return pd.DataFrame({"x1": x1.numpy(), "group": x_cat, "y": y.numpy()})


@pytest.fixture
def dm_numeric(df_numeric):
    return DataModule(df_numeric, response="y", numeric_scaling={})


@pytest.fixture
def dm_mixed(df_mixed):
    return DataModule(df_mixed, response="y", numeric_scaling={})


# ── 1. save creates a file (tracer bullet) — AC1 ─────────────────────────────


def test_save_state_creates_file(dm_numeric, tmp_path):
    """save_state() writes a file to the given path."""
    path = tmp_path / "dm.pt"
    dm_numeric.save_state(path)
    assert path.exists()


# ── 2. n_obs survives round-trip — AC3 ───────────────────────────────────────


def test_n_obs_survives_round_trip(dm_numeric, tmp_path):
    """n_obs is exactly preserved after save_state() → load_state()."""
    path = tmp_path / "dm.pt"
    dm_numeric.save_state(path)
    loaded = DataModule.load_state(path)
    assert loaded.n_obs == dm_numeric.n_obs


# ── 3. numeric transform is bit-identical after round-trip — AC1 ─────────────


def test_numeric_transform_bit_identical(dm_numeric, df_numeric, tmp_path):
    """save → load → transform equals original transform with max|Δ| == 0.

    Exact equality — both sides apply the same float statistics to the same
    float32 data; no MC noise or float accumulation should occur.
    """
    path = tmp_path / "dm.pt"
    dm_numeric.save_state(path)
    loaded = DataModule.load_state(path)

    fresh = df_numeric.head(10)
    orig_out = dm_numeric.transform(fresh)
    load_out = loaded.transform(fresh)

    assert set(orig_out.keys()) == set(load_out.keys())
    for name in orig_out:
        delta = (orig_out[name] - load_out[name]).abs().max().item()
        assert delta == 0.0, f"max|Δ| = {delta} for feature {name!r}"


# ── 4. state dict is closure-free — AC5 ──────────────────────────────────────


def _assert_closure_free(obj, path: str = "state") -> None:
    """Recursively assert all values are plain scalars, strings, or lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_closure_free(v, f"{path}[{k!r}]")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_closure_free(v, f"{path}[{i}]")
    else:
        assert isinstance(obj, (int, float, str, bool)), (
            f"{path}: expected plain scalar, got {type(obj).__name__!r}"
        )


def test_state_is_closure_free(dm_numeric):
    """get_state() contains only plain Python scalars, strings, and lists."""
    _assert_closure_free(dm_numeric.get_state())


def test_state_is_closure_free_mixed(dm_mixed):
    """get_state() is closure-free for a DataModule with categorical features."""
    _assert_closure_free(dm_mixed.get_state())


# ── 5. categorical num_levels survive round-trip — AC2 ───────────────────────


def test_num_levels_survive_round_trip(dm_mixed, tmp_path):
    """Category maps and level counts survive save → load exactly."""
    path = tmp_path / "dm.pt"
    dm_mixed.save_state(path)
    loaded = DataModule.load_state(path)
    assert loaded.num_levels == dm_mixed.num_levels


# ── 6. categorical transform bit-identical after round-trip — AC1 ────────────


def test_categorical_transform_bit_identical(dm_mixed, df_mixed, tmp_path):
    """Categorical integer codes are exact after save → load → transform."""
    path = tmp_path / "dm.pt"
    dm_mixed.save_state(path)
    loaded = DataModule.load_state(path)

    fresh = df_mixed.head(10)
    orig_out = dm_mixed.transform(fresh)
    load_out = loaded.transform(fresh)

    assert set(orig_out.keys()) == set(load_out.keys())
    for name in orig_out:
        delta = (orig_out[name].long() - load_out[name].long()).abs().max().item()
        assert delta == 0, f"max|Δ| = {delta} for categorical feature {name!r}"


# ── 7. unseen level errors identically after reload — AC2 ────────────────────


def test_unseen_level_errors_after_reload(dm_mixed, df_mixed, tmp_path):
    """Unseen category level raises ValueError identically before and after reload."""
    path = tmp_path / "dm.pt"
    dm_mixed.save_state(path)
    loaded = DataModule.load_state(path)

    unseen_df = df_mixed.head(3).copy()
    unseen_df["group"] = "UNSEEN_LEVEL"

    with pytest.raises(ValueError, match="Unseen category"):
        dm_mixed.transform(unseen_df)

    with pytest.raises(ValueError, match="Unseen category"):
        loaded.transform(unseen_df)


# ── 8. end-to-end: reloaded (model, DataModule) pair — AC4 ───────────────────


def test_end_to_end_model_datamodule_round_trip(df_numeric, tmp_path):
    """Reloaded (model, DataModule) pair transforms fresh data identically
    and runs sample_posterior_predictive without refit.

    Acceptance criterion 4: transform outputs are bit-identical to pre-save;
    the posterior predictive has the correct batch shape.
    """
    torch.manual_seed(0)
    dm = DataModule(df_numeric, response="y", numeric_scaling={})
    family = NormalFamily()
    formula = {
        "x1": BayesianMLP(1, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
    }
    model = BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)
    model.fit(dm, epochs=3, lr=1e-2)

    # Save both halves.
    model_path = tmp_path / "model.pt"
    dm_path = tmp_path / "dm.pt"
    model.save(model_path)
    dm.save_state(dm_path)

    # Reload both.
    formula_fresh = {
        "x1": BayesianMLP(1, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
    }
    loaded_model = BayesianNAMLSS.load(model_path, formula=formula_fresh, family=family)
    loaded_dm = DataModule.load_state(dm_path)

    # Fresh raw data — a held-out slice of the same DataFrame.
    fresh = df_numeric.head(8)

    orig_X = dm.transform(fresh)
    load_X = loaded_dm.transform(fresh)

    # Transform outputs must be bit-identical.
    for name in orig_X:
        delta = (orig_X[name] - load_X[name]).abs().max().item()
        assert delta == 0.0, (
            f"max|Δ| = {delta} for feature {name!r} after end-to-end round-trip"
        )

    # Loaded model must be able to generate a posterior predictive from the
    # transformed inputs — correct batch shape, no error.
    torch.manual_seed(0)
    pred = loaded_model.sample_posterior_predictive(load_X, T=10)
    assert pred.batch_shape == (8,)
