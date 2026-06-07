"""Re-seed protocol determinism (GitHub #90).

Settles by experiment the contradiction between the hardening brief ("two
runs with the same seed produce identical ELBO trajectories") and the old
CLAUDE.md caveat ("reproducibility never holds across two freshly-built model
objects"): the full re-seed protocol — ``seed_everything → build → fit`` on
CPU — run twice must give bit-identical per-epoch loss trajectories, because
re-seeding replays the same global-RNG stream through init, reparameterization
noise, and shuffling alike.

Reference-test archetype (CLAUDE.md): MC-convergence/reproducibility at the
model boundary — exact equality, no tolerance (same RNG stream ⇒ same bits).
"""

import pandas as pd
import pytest
import torch

from dune_bayes.data import DataModule
from dune_bayes.families import NormalFamily
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.shapes import BayesianMLP
from dune_bayes.utils import seed_everything

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 32
SEED = 1234
EPOCHS = 5

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_deterministic_mode():
    """deterministic=True flips a process-global torch switch — restore it."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(was_enabled)


@pytest.fixture
def data():
    # Data is generated once with an explicit Generator, outside the re-seed
    # protocol under test — only build + fit may touch the global RNG stream.
    g = torch.Generator().manual_seed(7)
    x1 = torch.randn(N_OBS, 1, generator=g)
    y = (2.0 * x1 + 0.1 * torch.randn(N_OBS, 1, generator=g)).squeeze(-1)
    return {"x1": x1}, y


# ── helpers ───────────────────────────────────────────────────────────────────


def _run_protocol(data) -> dict[str, list[float]]:
    """One full re-seed protocol pass: seed_everything → build → fit."""
    X, y = data
    seed_everything(SEED, deterministic=True)
    family = NormalFamily()
    formula = {
        "x1": BayesianMLP(1, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    model = BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)
    return model.fit(X, y, epochs=EPOCHS, lr=1e-2, warmup_epochs=2)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_reseed_protocol_replays_identical_loss_trajectory(data):
    """Two seed → build → fit runs give bit-identical per-epoch histories."""
    first = _run_protocol(data)
    second = _run_protocol(data)
    # Same RNG stream from the same seed ⇒ identical init, identical
    # reparameterization draws, identical optimizer path — exact equality.
    assert first["loss"] == second["loss"]
    assert first["nll"] == second["nll"]
    assert first["kl"] == second["kl"]


def test_reseed_protocol_replays_identical_minibatch_trajectory(data):
    """The protocol also holds through the DataLoader shuffle path.

    fit(seed=None) makes the shuffle draw its sampler seed from the *global*
    torch RNG — the prime suspect for an uncontrolled source (GitHub #90) —
    so re-seeding must replay the identical batch order too.
    """
    X, y = data
    df = pd.DataFrame({"x1": X["x1"].squeeze(-1).numpy(), "y": y.numpy()})

    def run() -> dict[str, list[float]]:
        seed_everything(SEED, deterministic=True)
        dm = DataModule(df, response="y")
        family = NormalFamily()
        formula = {
            "x1": BayesianMLP(1, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
        }
        model = BayesianNAMLSS(formula=formula, family=family)
        return model.fit(dm, epochs=EPOCHS, lr=1e-2, warmup_epochs=2, batch_size=8)

    first = run()
    second = run()
    # Exact equality for the same reason as the full-batch protocol; any
    # mismatch would localize the leak to the dataloader/shuffle path.
    assert first["loss"] == second["loss"]
    assert first["nll"] == second["nll"]
    assert first["kl"] == second["kl"]
