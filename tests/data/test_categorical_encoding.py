"""Tests for categorical encoding on DataModule (issue 0024 / GitHub #51).

Acceptance criteria: categorical columns transform to torch.long codes 0..K-1
with a deterministic mapping; num_levels exposes level counts; transform()
applies the fitted mapping; unseen levels raise a clear error; an end-to-end
path from DataModule → BayesianEmbedding extracts per-level effects.

Boundary tests only: codes, shapes, level counts, error surfaces.
"""

import pandas as pd
import pytest
import torch

from neural_bamlss.data import DataModule

# ── fixtures ──────────────────────────────────────────────────────────────────

COLORS = ["red", "blue", "green", "red", "blue"]
N = len(COLORS)


@pytest.fixture
def color_df():
    return pd.DataFrame({"color": COLORS, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})


# ── tracer bullet: codes are long, bounded, correct shape ─────────────────────


def test_categorical_codes_are_long_bounded_and_shaped(color_df):
    dm = DataModule(color_df, response="y")

    codes = dm.features["color"]
    assert codes.dtype == torch.long
    assert codes.shape == (N, 1)
    # 3 unique levels → codes in [0, 2]
    assert int(codes.min()) >= 0
    assert int(codes.max()) <= 2


# ── deterministic mapping ─────────────────────────────────────────────────────


def test_categorical_mapping_is_deterministic(color_df):
    dm1 = DataModule(color_df, response="y")
    dm2 = DataModule(color_df, response="y")

    assert torch.equal(dm1.features["color"], dm2.features["color"])


# ── num_levels ────────────────────────────────────────────────────────────────


def test_num_levels_exposes_count_per_categorical_feature():
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "size": ["S", "M", "L", "XL"],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dm = DataModule(df, response="y")

    assert dm.num_levels["color"] == 3
    assert dm.num_levels["size"] == 4


def test_num_levels_absent_for_numeric_features():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    dm = DataModule(df, response="y")

    assert "x" not in dm.num_levels


# ── transform() applies the fitted categorical mapping ────────────────────────


def test_transform_returns_same_codes_for_known_levels(color_df):
    dm = DataModule(color_df, response="y")
    # Applying the fitted mapping to the same data must reproduce the training codes.
    result = dm.transform(color_df.drop(columns=["y"]))

    assert result["color"].dtype == torch.long
    assert torch.equal(result["color"], dm.features["color"])


# ── unseen level error ────────────────────────────────────────────────────────


def test_unseen_level_at_transform_raises_with_level_name(color_df):
    dm = DataModule(color_df, response="y")
    df_new = pd.DataFrame({"color": ["purple"]})

    with pytest.raises((ValueError, KeyError), match="purple"):
        dm.transform(df_new)


# ── end-to-end: DataModule → BayesianEmbedding ───────────────────────────────


def test_end_to_end_categorical_feeds_bayesian_embedding():
    from neural_bamlss.layers import BayesianEmbedding

    df = pd.DataFrame(
        {
            "group": ["A", "B", "C", "A", "B", "C"],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    dm = DataModule(df, response="y")

    K = dm.num_levels["group"]
    emb = BayesianEmbedding(num_embeddings=K, embedding_dim=4, validate_args=True)

    idx = dm.features["group"].squeeze(-1)  # (n,) long tensor
    out = emb(idx)

    assert out.shape == (6, 4)
    # KL must be accumulated (variational mode, non-zero embedding table).
    assert emb.kl.item() != 0.0
