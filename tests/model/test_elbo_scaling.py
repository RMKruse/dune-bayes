"""ELBO scaling and warm-up verification — hand-derived references (#87).

PRD 0002 (#84) hardening slice: verify the ELBO's composition on a toy model
small enough to derive by hand.  No behavior change expected — any RED here
is a real bug in the loss assembly, not a test problem.

  1. KL/N scaling — 1-feature / 1-weight toy with a hand-derived KL value;
     the divisor is the full training-set size N.
  2. Loss composition — training loss equals mean-NLL + KL/N end-to-end
     through the public loss(), with both terms hand-derived.
  3. Minibatching — the KL divisor stays full-data N, never batch size.
  4. β warm-up — β=0 reduces the objective to pure NLL; β=1 restores the
     full ELBO; the default schedule is monotone non-decreasing and reaches
     exactly 1.

All references are written out in comments (acceptance criterion).  All tests
run in the core (unskippable) suite — numerical correctness tests are never
skippable (CLAUDE.md).

Toy construction: param_count=1 family Normal(μ, 1) (fixed σ so the NLL is
hand-computable), a BayesianMLP with hidden_dims=[] (exactly ONE variational
weight, no bias on the output layer), and a point-mode intercept (zero KL,
level pinned at 0) — so μ_i = w·x_i and the model's total KL is the single
weight's Gaussian–Gaussian KL.
"""

import math

import pytest
import torch

from dune_bayes.model import BayesianNAMLSS
from dune_bayes.shapes import BayesianMLP

# ── toy model ─────────────────────────────────────────────────────────────────

N_OBS = 16

# softplus⁻¹(0.5) = ln(e^0.5 − 1): pins the posterior scale to exactly 0.5,
# the value the hand derivations in tests 1 and 3 assume.
_RHO_HALF = math.log(math.expm1(0.5))


class _ToyFamily:
    """Normal(μ, 1) — one distributional parameter, hand-computable NLL.

    With σ fixed at 1:  −log p(y | μ) = ½·ln(2π) + ½·(y − μ)².
    """

    param_count = 1

    def __call__(self, params: torch.Tensor) -> torch.distributions.Normal:
        # validate_args=True: test fixture, not the training hot path (rule 6).
        return torch.distributions.Normal(params.squeeze(-1), 1.0, validate_args=True)


def _toy_model(n_obs: int, loc: float, rho: float) -> BayesianNAMLSS:
    """Build the 1-feature / 1-weight toy with its posterior pinned to (loc, rho)."""
    formula = {
        "x1": BayesianMLP(
            in_features=1,
            param_count=1,
            hidden_dims=[],  # no hidden layers → single output VariationalDense
            local_reparam=False,  # vanilla draw: w = loc + softplus(rho)·ε
            kl_divisor=n_obs,
            validate_args=True,
        )
    }
    model = BayesianNAMLSS(
        formula=formula,
        family=_ToyFamily(),
        n_obs=n_obs,
        intercept_mode="point",  # zero KL; loc init 0 keeps μ_i = w·x_i
    )
    dense = model.nets["x1"].layers[0]
    with torch.no_grad():
        dense.kernel_loc.fill_(loc)
        dense.kernel_rho.fill_(rho)
    return model


def _toy_data(n_obs: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Fixed deterministic (X, y) — no seed dependence in the references."""
    x = torch.linspace(-1.0, 1.0, n_obs).unsqueeze(-1)
    y = 0.3 * x.squeeze(-1) + 0.1  # any fixed target; references recompute from it
    return {"x1": x}, y


# ── 1. KL/N scaling: hand-derived KL value ────────────────────────────────────


def test_toy_kl_matches_hand_derived_value_over_full_n():
    """The recorded KL equals the hand-derived weight-KL divided by full N.

    Posterior q(w) = N(0.5, 0.5²), prior p(w) = N(0, 1²).  By the closed form
    KL = ln(σ_p/σ_q) + (σ_q² + μ_q²)/(2σ_p²) − ½:

        KL = ln(1/0.5) + (0.25 + 0.25)/2 − ½
           = ln 2 + 0.25 − 0.5
           = ln 2 − ¼  ≈ 0.4431471806

    fit() for one epoch records collect_kl() in history["kl"]; with β=1
    (warmup_epochs=0) and kl_divisor = N_OBS = 16 the recorded value must be
    (ln 2 − ¼)/16.  lr=0 pins the parameters we set, making the claim exact.

    Tolerance: rel=1e-5 — the KL is analytic (no MC noise); the only error is
    float32 arithmetic plus the softplus(softplus_inv(0.5)) round-trip.
    """
    model = _toy_model(N_OBS, loc=0.5, rho=_RHO_HALF)
    X, y = _toy_data(N_OBS)

    history = model.fit(X, y, epochs=1, lr=0.0, warmup_epochs=0)

    expected = (math.log(2.0) - 0.25) / N_OBS
    assert history["kl"][0] == pytest.approx(expected, rel=1e-5)


# ── 2. loss composition: mean-NLL + KL/N end-to-end through loss() ────────────

# Degenerate posterior scale for the deterministic-forward trick:
# σ_q = softplus(−50) ≈ e⁻⁵⁰ ≈ 1.93e−22, so the weight draw
# w = 0.5 + σ_q·ε rounds to exactly 0.5 in float32 (eps at 0.5 ≈ 6e−8) —
# the forward pass is deterministic and the NLL hand-computable.
_RHO_DEGENERATE = -50.0


def _hand_loss_terms(loc: float) -> tuple[float, float]:
    """Hand-derived (mean-NLL, KL) for the deterministic toy at weight = loc.

    mean-NLL: with μ_i = loc·x_i and σ = 1,
        NLL = (1/N) Σ_i [ ½·ln(2π) + ½·(y_i − loc·x_i)² ]
    computed below in float64 directly from the formula — independent of the
    family's log_prob.

    KL: q(w) = N(loc, σ_q²) with σ_q = softplus(−50) ≈ e⁻⁵⁰, p(w) = N(0, 1):
        KL = ln(1/σ_q) + (σ_q² + loc²)/2 − ½
           ≈ 50 + loc²/2 − ½        (σ_q² ≈ 3.7e−44 is exactly 0 in float64)
    For loc = 0.5: KL ≈ 50 + 0.125 − 0.5 = 49.625.
    """
    X, y = _toy_data(N_OBS)
    x64 = X["x1"].squeeze(-1).double()
    y64 = y.double()
    nll = float((0.5 * math.log(2 * math.pi) + 0.5 * (y64 - loc * x64) ** 2).mean())
    kl = 50.0 + loc**2 / 2.0 - 0.5
    return nll, kl


def test_loss_composes_mean_nll_plus_kl_over_n():
    """model.loss(X, y) == hand-derived mean-NLL + hand-derived KL / N.

    The end-to-end composition claim of the issue: the training objective is
    mean-NLL + KL/N, verified entirely through the public loss() against
    references written out in _hand_loss_terms.

    Tolerance: rel=1e-5 — the forward pass is deterministic (degenerate σ_q)
    and the KL analytic, so the only error is float32 arithmetic; the ≈e⁻⁵⁰
    terms dropped from the hand derivation are below float32 resolution.
    """
    model = _toy_model(N_OBS, loc=0.5, rho=_RHO_DEGENERATE)
    X, y = _toy_data(N_OBS)

    nll, kl = _hand_loss_terms(loc=0.5)
    assert float(model.loss(X, y).detach()) == pytest.approx(nll + kl / N_OBS, rel=1e-5)


# ── 3. minibatching: the KL divisor stays full-data N ─────────────────────────


def test_minibatch_kl_divisor_stays_full_data_n():
    """Under minibatching the recorded KL is KL/N_full, never KL/batch_size.

    Same pinned posterior as test 1 (hand-derived KL = ln 2 − ¼), trained for
    one epoch in 4 batches of 4 via a DataModule.  lr=0 freezes the
    parameters, so every batch stashes the identical analytic KL and the
    epoch-mean recorded in history is exactly that per-batch value — making
    the divisor directly observable: (ln 2 − ¼)/16, NOT (ln 2 − ¼)/4.

    Tolerance: rel=1e-5, same float32-only budget as test 1.
    """
    import pandas as pd

    from dune_bayes.data import DataModule

    batch_size = 4
    X, y = _toy_data(N_OBS)
    df = pd.DataFrame({"x1": X["x1"].squeeze(-1).numpy(), "y": y.numpy()})
    dm = DataModule(df, response="y")

    model = _toy_model(N_OBS, loc=0.5, rho=_RHO_HALF)

    history = model.fit(
        dm, epochs=1, lr=0.0, warmup_epochs=0, batch_size=batch_size, seed=0
    )

    kl = math.log(2.0) - 0.25  # hand-derived in test 1
    assert history["kl"][0] == pytest.approx(kl / N_OBS, rel=1e-5)
    # The wrong divisor (batch size) is 4× larger — assert it is excluded.
    assert history["kl"][0] != pytest.approx(kl / batch_size, rel=1e-5)


# ── 4. β = 0 endpoint: the objective reduces to pure NLL ──────────────────────


def test_beta_zero_reduces_objective_to_pure_nll():
    """At β = 0 the training loss is exactly the mean-NLL — no KL trace.

    set_kl_beta(model, 0.0) gates the stashed KL to β·KL/N = 0 on the next
    forward pass, so the deterministic toy's loss() must equal the
    hand-derived mean-NLL alone (the warm-up's epoch-0 objective).

    Tolerance: rel=1e-5 for the NLL value (float32 arithmetic, deterministic
    forward); the dropped KL term would shift the loss by 49.625/16 ≈ 3.1 —
    five orders of magnitude above the tolerance, so a leaked KL cannot pass.
    """
    from dune_bayes.layers import set_kl_beta

    model = _toy_model(N_OBS, loc=0.5, rho=_RHO_DEGENERATE)
    X, y = _toy_data(N_OBS)
    nll, _ = _hand_loss_terms(loc=0.5)

    set_kl_beta(model, 0.0)
    assert float(model.loss(X, y).detach()) == pytest.approx(nll, rel=1e-5)


# ── 5. β = 1 endpoint: the full ELBO is restored ──────────────────────────────


def test_beta_one_restores_full_elbo():
    """At β = 1 the loss is the full negative ELBO: mean-NLL + KL/N.

    Driving β through 0 and back to 1 via set_kl_beta (the same entry point
    the warm-up callback uses) must restore exactly the test-2 composition —
    warm-up is a pure gate with no residual rescaling.

    Tolerance: rel=1e-5, same deterministic float32-only budget as test 2.
    """
    from dune_bayes.layers import set_kl_beta

    model = _toy_model(N_OBS, loc=0.5, rho=_RHO_DEGENERATE)
    X, y = _toy_data(N_OBS)
    nll, kl = _hand_loss_terms(loc=0.5)

    set_kl_beta(model, 0.0)  # leave the β=1 default first…
    set_kl_beta(model, 1.0)  # …then restore: the full ELBO must come back
    assert float(model.loss(X, y).detach()) == pytest.approx(nll + kl / N_OBS, rel=1e-5)


# ── 6. default schedule: monotone non-decreasing, reaches exactly 1 ───────────


def test_default_warmup_schedule_monotone_and_reaches_exactly_one():
    """The default warm-up schedule never decreases and saturates at exactly 1.0.

    A user callback runs after the auto-injected warm-up callback each epoch,
    so reading the kl_beta buffer there observes the β actually applied that
    epoch.  For warmup_epochs=5 over 8 epochs the documented schedule
    β = min(1, epoch/warmup_epochs) gives 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1:
    monotone non-decreasing, first hits 1 at epoch index 5, and stays there.

    Endpoint equality is exact (==, not approx): min(1.0, …) must yield the
    float 1.0, so the terminal ELBO is the full ELBO with no residual scaling.
    """
    from dune_bayes.layers.variational_dense import VariationalDense

    warmup, epochs = 5, 8
    model = _toy_model(N_OBS, loc=0.5, rho=_RHO_DEGENERATE)
    X, y = _toy_data(N_OBS)
    dense = next(m for m in model.modules() if isinstance(m, VariationalDense))

    betas: list[float] = []
    model.fit(
        X,
        y,
        epochs=epochs,
        lr=0.0,
        warmup_epochs=warmup,
        callbacks=[lambda epoch: betas.append(float(dense.kl_beta))],
    )

    assert len(betas) == epochs
    assert betas[0] == 0.0  # starts at pure NLL (test 4's endpoint)
    # strict=False: adjacent-pairs zip is shorter by one on purpose.
    assert all(b1 <= b2 for b1, b2 in zip(betas, betas[1:], strict=False))
    assert betas[warmup] == 1.0  # reaches exactly 1 when the ramp ends…
    assert all(b == 1.0 for b in betas[warmup:])  # …and stays there
