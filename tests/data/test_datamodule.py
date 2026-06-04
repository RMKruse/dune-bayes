"""Tests for DataModule (issue 0022 / GitHub #49).

Boundary behavior: a DataModule built from a DataFrame + response name yields
the model-ready feature dict and target tensor matching the fit contract, and
exposes n_obs so KL/N is wired from the data — never asserted via internals.
"""

import pandas as pd
import pytest
import torch

from neural_bamlss.data import DataModule

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def df():
    rng = torch.Generator().manual_seed(0)
    x1 = torch.randn(N_OBS, generator=rng)
    x2 = torch.randn(N_OBS, generator=rng)
    y = 2.0 * x1 - 1.0 * x2 + 0.1 * torch.randn(N_OBS, generator=rng)
    return pd.DataFrame({"x1": x1.numpy(), "x2": x2.numpy(), "y": y.numpy()})


# ── feature dict ──────────────────────────────────────────────────────────────


def test_features_are_float32_column_tensors(df):
    dm = DataModule(df, response="y")

    assert set(dm.features.keys()) == {"x1", "x2"}
    for name, tensor in dm.features.items():
        assert tensor.dtype == torch.float32
        assert tensor.shape == (N_OBS, 1)
        # Raw pass-through (no preprocessing yet): values match the DataFrame.
        # Exact equality — both sides are the same float32 values, no math done.
        # torch.tensor copies — pandas ≥3 to_numpy() views are read-only.
        assert torch.equal(
            tensor.squeeze(-1),
            torch.tensor(df[name].to_numpy(), dtype=torch.float32),
        )


# ── target tensor ─────────────────────────────────────────────────────────────


def test_target_is_float32_response_vector(df):
    dm = DataModule(df, response="y")

    assert dm.target.dtype == torch.float32
    # (n,) not (n, 1): matches the fit contract — dist.log_prob(y) needs the
    # target's shape to equal the family batch_shape.
    assert dm.target.shape == (N_OBS,)
    assert torch.equal(dm.target, torch.tensor(df["y"].to_numpy(), dtype=torch.float32))
    # The response is data, not a feature.
    assert "y" not in dm.features


# ── n_obs ─────────────────────────────────────────────────────────────────────


def test_n_obs_is_training_set_size(df):
    dm = DataModule(df, response="y")
    assert dm.n_obs == N_OBS


# ── constructor errors ────────────────────────────────────────────────────────


def test_missing_response_column_raises(df):
    with pytest.raises(ValueError, match="z"):
        DataModule(df, response="z")
