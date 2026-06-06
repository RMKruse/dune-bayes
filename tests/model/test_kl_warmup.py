"""Tests for KL warm-up auto-injection in BayesianNAMLSS.fit() (issue 0004 / GitHub #5).

Acceptance criteria tested here:
  AC1 — fit() injects warm-up with no extra user setup.
  AC2 — kl_beta follows β: 0→1 over warmup_epochs on every VariationalDense.
  AC3 — warm-up length is configurable; warmup_epochs=0 disables it.
  AC4 — user-supplied callbacks list is preserved alongside the injected one.
"""

import pytest
import torch

from neural_bamlss.families import NormalFamily
from neural_bamlss.layers.variational_dense import VariationalDense
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
IN = 1

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily()


@pytest.fixture
def model(family):
    torch.manual_seed(42)
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def X_y():
    g = torch.Generator().manual_seed(0)
    X = {"x1": torch.randn(N_OBS, IN, generator=g)}
    y = 2.0 * X["x1"].squeeze(-1) + 0.1 * torch.randn(N_OBS, generator=g)
    return X, y


# ── helpers ───────────────────────────────────────────────────────────────────


def _full_kl_reference(model, X):
    """Return the KL produced by a single forward pass with β=1 (no warm-up scaling)."""
    from neural_bamlss.layers import collect_kl, set_kl_beta

    set_kl_beta(model, 1.0)
    model(X)
    return collect_kl(model).item()


# ── AC1: auto-injection ───────────────────────────────────────────────────────


def test_warmup_auto_injected(model, X_y):
    """AC1: fit() injects warm-up with no extra user setup.

    After training for warmup_epochs epochs all VariationalDense layers must
    have kl_beta == 1.0 (schedule completed, no β argument passed by the user).
    """
    X, y = X_y
    # β = min(1, epoch / warmup_epochs) → first hits 1.0 at epoch index
    # == warmup_epochs.
    # Train warmup_epochs + 1 steps so the final epoch starts with β = 1.0.
    warmup = 4
    model.fit(X, y, epochs=warmup + 1, lr=1e-2, warmup_epochs=warmup)
    vd_layers = [m for m in model.modules() if isinstance(m, VariationalDense)]
    assert vd_layers, "fixture must contain at least one VariationalDense"
    for layer in vd_layers:
        assert layer.kl_beta.item() == pytest.approx(1.0)


# ── AC2a: β = 0 at epoch 0 → KL ≈ 0 ─────────────────────────────────────────


def test_kl_zero_at_first_epoch(model, X_y):
    """AC2: β = min(1, 0/warmup_epochs) = 0 so KL recorded at epoch 0 is exactly 0.

    kl_beta=0 gates the KL accumulation in VariationalDense.forward() entirely.
    Tolerance abs=1e-7 to catch any float bleed-through.
    """
    X, y = X_y
    history = model.fit(X, y, epochs=10, lr=1e-2, warmup_epochs=10)
    assert history["kl"][0] == pytest.approx(0.0, abs=1e-7)


# ── AC2b: KL increases over the warm-up window ────────────────────────────────


def test_kl_increases_over_warmup(model, X_y):
    """AC2: KL in the history grows monotonically (on average) during warm-up.

    We compare the mean KL in the first half of warm-up against the second half;
    the second half must be strictly larger because β is higher.
    This is robust to per-step MC noise (mean comparison, not step-by-step).
    """
    X, y = X_y
    warmup = 10
    history = model.fit(X, y, epochs=warmup + 1, lr=1e-2, warmup_epochs=warmup)
    kl = history["kl"]
    first_half_mean = sum(kl[: warmup // 2]) / (warmup // 2)
    second_half_mean = sum(kl[warmup // 2 :]) / len(kl[warmup // 2 :])
    assert second_half_mean > first_half_mean, (
        f"KL did not grow during warm-up: "
        f"first_half={first_half_mean:.6f}, second_half={second_half_mean:.6f}"
    )


# ── AC3a: warmup_epochs=0 disables warm-up ────────────────────────────────────


def test_warmup_disabled_when_warmup_epochs_zero(model, X_y):
    """AC3: warmup_epochs=0 skips the warm-up injection; KL is positive from epoch 0.

    With warm-up active, kl[0] == 0.  With warm-up disabled, kl[0] reflects the
    full (β=1) KL contribution — the signal that no gating happened.
    """
    X, y = X_y
    history = model.fit(X, y, epochs=5, lr=1e-2, warmup_epochs=0)
    assert history["kl"][0] > 0.0


# ── AC3b: warm-up length controls saturation point ────────────────────────────


def test_warmup_length_controls_kl_at_midpoint(family, X_y):
    """AC3: a shorter warmup_epochs means higher KL earlier in training.

    With warmup=2 the KL at epoch 2 is at full β (β=1); with warmup=20 it is
    still ramping (β=0.1).  The short-warmup model must have higher KL at epoch 2.
    """
    X, y = X_y
    torch.manual_seed(0)
    formula_short = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
    }
    m_short = BayesianNAMLSS(formula=formula_short, family=family, n_obs=N_OBS)

    torch.manual_seed(0)
    formula_long = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
    }
    m_long = BayesianNAMLSS(formula=formula_long, family=family, n_obs=N_OBS)

    # Train both for 5 epochs; observe KL at epoch index 2.
    hist_short = m_short.fit(X, y, epochs=5, lr=1e-2, warmup_epochs=2)
    hist_long = m_long.fit(X, y, epochs=5, lr=1e-2, warmup_epochs=20)

    # At epoch 2: β_short = min(1, 2/2) = 1.0; β_long = min(1, 2/20) = 0.1.
    assert hist_short["kl"][2] > hist_long["kl"][2], (
        f"Short warmup must have higher KL at epoch 2: "
        f"short={hist_short['kl'][2]:.6f}, long={hist_long['kl'][2]:.6f}"
    )


# ── AC4: user callbacks preserved ─────────────────────────────────────────────


def test_user_callback_called_each_epoch(model, X_y):
    """AC4: a user-supplied callback is called exactly once per epoch.

    Callback receives the 0-based epoch index so callers can implement their
    own schedules (e.g. LR annealing, logging).
    """
    X, y = X_y
    called: list[int] = []
    model.fit(X, y, epochs=5, lr=1e-2, callbacks=[lambda epoch: called.append(epoch)])
    assert called == list(range(5))


def test_user_callback_preserved_alongside_warmup(model, X_y):
    """AC4: user callbacks run alongside the auto-injected warm-up callback.

    Both effects must be observable: kl[0] == 0 (warm-up gating active) AND
    the user callback recorded all epoch indices (callback not displaced).
    """
    X, y = X_y
    called: list[int] = []
    history = model.fit(
        X,
        y,
        epochs=6,
        lr=1e-2,
        warmup_epochs=5,
        callbacks=[lambda epoch: called.append(epoch)],
    )
    # warm-up still active: KL gated to 0 at epoch 0
    assert history["kl"][0] == pytest.approx(0.0, abs=1e-7)
    # user callback received all epochs
    assert called == list(range(6))
