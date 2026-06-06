"""Tests for BayesianNAMLSS save/load round-trip (issue 0015 / GitHub #16).

Four reference-test archetypes (CLAUDE.md):
  - Round-trip:    save → load gives max|Δw| = 0 (exact weight equality).
  - Shape:         loaded model forward() returns distribution with correct batch shape.
  - MC-convergence: architecture / hyperparameters preserved after load.
  - Closed-form:   full-model pickle also round-trips with max|Δw| = 0.
"""

import pytest
import torch

from dune_bayes.families import NormalFamily
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
IN = 1
BATCH = 8


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily()


@pytest.fixture
def fitted_model(family):
    """A BayesianNAMLSS that has been through a few training steps."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    model = BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)
    X = {"x1": torch.randn(N_OBS, IN, generator=g)}
    y = 2.0 * X["x1"].squeeze(-1) + 0.1 * torch.randn(N_OBS, generator=g)
    model.fit(X, y, epochs=3, lr=1e-2)
    return model


@pytest.fixture
def formula_fresh(family):
    """A fresh (untrained) formula with the same architecture as fitted_model."""
    torch.manual_seed(99)
    return {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }


@pytest.fixture
def X_batch():
    g = torch.Generator().manual_seed(7)
    return {"x1": torch.randn(BATCH, IN, generator=g)}


# ── 1. save creates a file (tracer bullet) — AC1 ─────────────────────────────


def test_save_creates_file(fitted_model, tmp_path):
    """save() writes a file to the given path."""
    path = tmp_path / "model.pt"
    fitted_model.save(path)
    assert path.exists()


# ── 2. round-trip: max|Δw| = 0 — AC1 ─────────────────────────────────────────


def test_load_restores_exact_weights(fitted_model, formula_fresh, family, tmp_path):
    """save() then load() gives max|Δw| = 0 (exact equality, not approx).

    Variational weights (loc and rho) must be exactly preserved — floating-point
    rounding during serialization would violate the round-trip guarantee.
    """
    path = tmp_path / "model.pt"
    fitted_model.save(path)
    loaded = BayesianNAMLSS.load(path, formula=formula_fresh, family=family)

    original_sd = fitted_model.state_dict()
    loaded_sd = loaded.state_dict()

    assert set(original_sd.keys()) == set(loaded_sd.keys()), "key sets must match"
    max_delta = max(
        (original_sd[k] - loaded_sd[k]).abs().max().item() for k in original_sd
    )
    assert max_delta == 0.0, f"max|Δw| = {max_delta} after round-trip; expected 0"


# ── 3. loaded model runs a forward pass — AC4 ────────────────────────────────


def test_loaded_model_forward_shape(
    fitted_model, formula_fresh, family, tmp_path, X_batch
):
    """Loaded model forward() returns a distribution with the correct batch shape."""
    path = tmp_path / "model.pt"
    fitted_model.save(path)
    loaded = BayesianNAMLSS.load(path, formula=formula_fresh, family=family)
    loaded.eval()

    dist = loaded(X_batch)
    assert isinstance(dist, torch.distributions.Distribution)
    assert dist.batch_shape == (BATCH,)


# ── 4. architecture/hyperparameters preserved — AC4 ──────────────────────────


def test_load_preserves_n_obs(fitted_model, formula_fresh, family, tmp_path):
    """n_obs is preserved across the round-trip."""
    path = tmp_path / "model.pt"
    fitted_model.save(path)
    loaded = BayesianNAMLSS.load(path, formula=formula_fresh, family=family)
    assert loaded.n_obs == fitted_model.n_obs


def test_load_preserves_feature_names(fitted_model, formula_fresh, family, tmp_path):
    """feature_names are preserved across the round-trip."""
    path = tmp_path / "model.pt"
    fitted_model.save(path)
    loaded = BayesianNAMLSS.load(path, formula=formula_fresh, family=family)
    assert loaded.feature_names == fitted_model.feature_names


def test_load_preserves_intercept_mode(family, tmp_path):
    """intercept_mode survives the round-trip — a point-mode model must not come
    back variational (the state_dicts are not even compatible: no rho)."""
    torch.manual_seed(0)
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    model = BayesianNAMLSS(
        formula=formula, family=family, n_obs=N_OBS, intercept_mode="point"
    )
    formula_fresh = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = BayesianNAMLSS.load(path, formula=formula_fresh, family=family)
    assert loaded.intercept.mode == "point"


def test_load_preserves_feature_dropout(family, tmp_path):
    """feature_dropout is preserved across the round-trip."""
    torch.manual_seed(0)
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    model = BayesianNAMLSS(
        formula=formula, family=family, n_obs=N_OBS, feature_dropout=0.2
    )
    formula_fresh = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = BayesianNAMLSS.load(path, formula=formula_fresh, family=family)
    assert loaded.feature_dropout == pytest.approx(0.2)


def test_load_rejects_mismatched_feature_names(fitted_model, family, tmp_path):
    """load() with a formula whose feature names don't match the checkpoint
    raises ValueError naming both sides — not a cryptic load_state_dict error."""
    path = tmp_path / "model.pt"
    fitted_model.save(path)
    torch.manual_seed(99)
    formula_wrong = {
        "x2": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    with pytest.raises(ValueError, match=r"x2.*x1|feature names"):
        BayesianNAMLSS.load(path, formula=formula_wrong, family=family)


# ── 5. full-model pickle round-trip — AC2 ────────────────────────────────────


def test_full_model_pickle_round_trip(fitted_model, tmp_path, X_batch):
    """torch.save(model) / torch.load() also gives max|Δw| = 0.

    This is the 'SavedModel' equivalent: no architecture reconstruction needed,
    but the file is tied to the Python/class definitions.
    """
    path = tmp_path / "model_full.pt"
    torch.save(fitted_model, path)
    loaded = torch.load(path, weights_only=False)

    original_sd = fitted_model.state_dict()
    loaded_sd = loaded.state_dict()

    max_delta = max(
        (original_sd[k] - loaded_sd[k]).abs().max().item() for k in original_sd
    )
    assert max_delta == 0.0, (
        f"max|Δw| = {max_delta} after full-model pickle; expected 0"
    )


def test_full_model_pickle_forward_shape(fitted_model, tmp_path, X_batch):
    """Full-model pickle round-trip: loaded model forward() has correct shape."""
    path = tmp_path / "model_full.pt"
    torch.save(fitted_model, path)
    loaded = torch.load(path, weights_only=False)
    loaded.eval()

    dist = loaded(X_batch)
    assert dist.batch_shape == (BATCH,)


# ── 6. H5 extension raises a clear error — AC3 ───────────────────────────────


def test_save_h5_raises_clear_error(fitted_model, tmp_path):
    """save() with a .h5 path raises ValueError with a clear message.

    H5 is not supported: HDF5 weight-name collisions across variational layers
    cause silent corruption (spike-confirmed on the TF/Keras stack; the PyTorch
    equivalent is not implemented and the format is explicitly unsupported).
    """
    path = tmp_path / "model.h5"
    with pytest.raises(ValueError, match="[Hh]5"):
        fitted_model.save(path)
