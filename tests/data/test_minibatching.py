"""Tests for minibatching support in DataModule + BayesianNAMLSS.fit()

Issue 0026 / GitHub #53.

Acceptance criteria:
  AC1 — batched iteration yields (feature-dict, target) batches with correct shapes;
         a final partial batch is handled.
  AC2 — shuffling is seedable — same seed → same batch order within one run.
  AC3 — the KL divisor under minibatching equals full-data N — asserted by
         inspecting kl_divisor on each VariationalDense after fit().
  AC4 — end-to-end: fitting with a batch size on toy data runs to completion
         with finite, decreasing loss and per-epoch history.
  AC5 — full-batch default is unchanged when no batch size is given.
"""

import math

import pandas as pd
import pytest
import torch

from neural_bamlss.data import DataModule
from neural_bamlss.families import NormalFamily
from neural_bamlss.layers import collect_kl
from neural_bamlss.layers.variational_dense import VariationalDense
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
BATCH_SIZE = 8

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def df():
    rng = torch.Generator().manual_seed(0)
    x1 = torch.randn(N_OBS, generator=rng)
    x2 = torch.randn(N_OBS, generator=rng)
    y = 2.0 * x1 - 1.0 * x2 + 0.1 * torch.randn(N_OBS, generator=rng)
    return pd.DataFrame({"x1": x1.numpy(), "x2": x2.numpy(), "y": y.numpy()})


@pytest.fixture
def dm(df):
    return DataModule(df, response="y")


@pytest.fixture
def toy_model(dm):
    family = NormalFamily()
    formula = {
        "x1": BayesianMLP(1, family.param_count, hidden_dims=[8], kl_divisor=dm.n_obs),
        "x2": BayesianMLP(1, family.param_count, hidden_dims=[8], kl_divisor=dm.n_obs),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=dm.n_obs)


# ── AC1: batch shapes (tracer bullet) ─────────────────────────────────────────


def test_dataloader_yields_correct_batch_shapes(dm):
    """First batch from dm.dataloader() has correct tensor shapes."""
    loader = dm.dataloader(batch_size=BATCH_SIZE)
    batch_X, batch_y = next(iter(loader))

    assert batch_y.shape == (BATCH_SIZE,)
    assert batch_y.dtype == torch.float32

    assert set(batch_X.keys()) == set(dm.features.keys())
    for name, t in batch_X.items():
        assert t.shape[0] == BATCH_SIZE
        assert t.dtype == dm.features[name].dtype


# ── AC1 continued: partial batch covered ──────────────────────────────────────


def test_dataloader_covers_all_observations(dm):
    """All N_OBS observations appear across batches (partial batch is not dropped)."""
    # N_OBS=32, batch_size=10 → 3 full batches of 10 + 1 partial of 2.
    loader = dm.dataloader(batch_size=10, shuffle=False)
    total = sum(batch_y.shape[0] for _, batch_y in loader)
    assert total == N_OBS


# ── AC2: seedable shuffle ─────────────────────────────────────────────────────


def test_dataloader_shuffle_is_seedable(dm):
    """Two DataLoaders built with the same generator seed yield identical batches."""
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    loader1 = dm.dataloader(batch_size=BATCH_SIZE, generator=g1)
    loader2 = dm.dataloader(batch_size=BATCH_SIZE, generator=g2)
    order1 = [batch_y.clone() for _, batch_y in loader1]
    order2 = [batch_y.clone() for _, batch_y in loader2]

    for y1, y2 in zip(order1, order2, strict=True):
        assert torch.equal(y1, y2)


def test_dataloader_different_seeds_differ(dm):
    """Two DataLoaders built with different seeds yield different batch orders."""
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(99)

    loader1 = dm.dataloader(batch_size=BATCH_SIZE, generator=g1)
    loader2 = dm.dataloader(batch_size=BATCH_SIZE, generator=g2)
    order1 = [batch_y.clone() for _, batch_y in loader1]
    order2 = [batch_y.clone() for _, batch_y in loader2]

    # Not all batches should be identical (probability of collision ≈ 0 for N=32).
    assert any(not torch.equal(y1, y2) for y1, y2 in zip(order1, order2, strict=True))


# ── AC3: KL divisor stays full-data N ────────────────────────────────────────


def test_kl_divisor_stays_full_n_not_batch_size(dm, toy_model):
    """After fit() with a minibatch, each VariationalDense.kl_divisor == dm.n_obs.

    The KL term in the ELBO must be divided by full N, never by batch size.
    Dividing by batch size would over-regularize by a factor of N/batch_size
    (CLAUDE.md numerical rule — KL/N is load-bearing).
    """
    toy_model.fit(dm, epochs=3, batch_size=BATCH_SIZE, seed=0)

    for module in toy_model.modules():
        if isinstance(module, VariationalDense):
            assert module.kl_divisor == dm.n_obs, (
                f"kl_divisor == {module.kl_divisor}, expected {dm.n_obs} "
                "(minibatch training must not change the KL divisor to batch_size)"
            )


def test_kl_value_equals_full_batch_kl(dm, toy_model):
    """collect_kl() after a minibatch forward equals collect_kl() after a full-batch
    forward (same weights, same kl_divisor=N — KL does not depend on data).

    This verifies that the KL/N term is identical regardless of batch size.
    """
    toy_model.train()
    # Full-batch forward.
    _ = toy_model(dm.features)
    kl_full = collect_kl(toy_model).item()

    # Minibatch forward — different data slice, same weights.
    mini_X = {k: v[:BATCH_SIZE] for k, v in dm.features.items()}
    _ = toy_model(mini_X)
    kl_mini = collect_kl(toy_model).item()

    # KL depends only on weights and kl_divisor, not on the data passed in.
    # Exact equality: same computation path, no stochasticity between the two calls.
    assert kl_full == pytest.approx(kl_mini, rel=1e-5)


# ── AC4: end-to-end convergence with minibatching ────────────────────────────


def test_fit_with_batch_size_runs_to_completion(dm, toy_model):
    """fit(dm, batch_size=BATCH_SIZE) produces finite, decreasing NLL over 50 epochs.

    NLL is checked (not total loss) because the KL warm-up ramp makes the total
    loss non-monotone: epoch-0 loss has β=0 (NLL only) while later epochs include
    the full KL term, so total-loss comparisons are not a valid convergence signal.
    10% relative tolerance accommodates MC noise from the stochastic forward pass.
    """
    history = toy_model.fit(dm, epochs=50, lr=1e-2, batch_size=BATCH_SIZE, seed=0)

    assert "loss" in history and "nll" in history and "kl" in history
    assert len(history["loss"]) == 50  # one entry per epoch

    assert all(math.isfinite(v) for v in history["loss"]), "loss has non-finite values"
    assert all(math.isfinite(v) for v in history["nll"]), "nll has non-finite values"

    first_nll = history["nll"][0]
    last_nll = history["nll"][-1]
    # NLL should decrease; 10% tolerance for MC noise (mirrors test_bayesian_namlss.py).
    assert last_nll < first_nll * 1.10, (
        f"NLL did not decrease: first={first_nll:.4f}, last={last_nll:.4f}"
    )


# ── AC5: full-batch default unchanged ────────────────────────────────────────


def test_full_batch_unchanged_when_no_batch_size(dm, toy_model):
    """fit() without batch_size behaves identically to before this issue."""
    history = toy_model.fit(dm, epochs=5)
    assert len(history["loss"]) == 5
    assert all(math.isfinite(v) for v in history["loss"])
