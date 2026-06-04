"""Tests for the DataModule-accepting model surfaces (issue 0022 / GitHub #49).

Boundary behavior: from_formula(data=...) wires KL/N from the data with no
explicit n_obs argument, and fit() accepts a DataModule in place of (X, y).
"""

import pandas as pd
import pytest
import torch

from neural_bamlss.data import DataModule
from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 64

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dm():
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(N_OBS, generator=g)
    x2 = torch.randn(N_OBS, generator=g)
    y = 2.0 * x1 - 1.0 * x2 + 0.1 * torch.randn(N_OBS, generator=g)
    df = pd.DataFrame({"x1": x1.numpy(), "x2": x2.numpy(), "y": y.numpy()})
    return DataModule(df, response="y")


# ── from_formula(data=...) wires KL/N from the data ───────────────────────────


def test_from_formula_with_datamodule_auto_wires_n_obs(dm):
    model = BayesianNAMLSS.from_formula(
        "y ~ BayesianMLP(x1, hidden_dims=(8,)) + BayesianMLP(x2, hidden_dims=(8,))",
        family=NormalFamily(),
        data=dm,
    )
    # No explicit n_obs argument anywhere — N comes from the data.
    assert model.n_obs == N_OBS
    # kl_divisor is the public per-term config (it round-trips via get_config);
    # KL/N wired into every Bayesian term is the documented objective.
    for name in ("x1", "x2"):
        assert model.nets[name].kl_divisor == float(N_OBS)


# ── fit accepts a DataModule in place of (X, y) ───────────────────────────────


def test_fit_with_datamodule_end_to_end(dm):
    torch.manual_seed(42)
    model = BayesianNAMLSS.from_formula(
        "y ~ BayesianMLP(x1, hidden_dims=(8,)) + BayesianMLP(x2, hidden_dims=(8,))",
        family=NormalFamily(),
        data=dm,
    )
    history = model.fit(dm, epochs=60, lr=1e-2)

    assert all(torch.isfinite(torch.tensor(history["loss"])))
    # The KL term reflects N: post-warm-up KL is strictly positive (never
    # silently dropped) and finite under the KL/N scaling.
    assert history["kl"][-1] > 0.0
    # Mean over first/last 5 epochs smooths single-pass MC noise in the ELBO;
    # 60 epochs at lr=1e-2 reliably cuts the loss on this near-linear toy.
    first = sum(history["loss"][:5]) / 5
    last = sum(history["loss"][-5:]) / 5
    assert last < first
