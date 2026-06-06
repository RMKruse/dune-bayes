"""Tests for BayesianNAMLSS walking skeleton (issue 0003 / GitHub #4).

Four reference-test archetypes (CLAUDE.md):
  - Shape:         forward() output is a distribution with the right batch shape.
  - Closed-form:   KL/N appears in the total loss (total > NLL alone).
  - Round-trip:    partial-Bayesian formula — deterministic term contributes zero KL.
  - MC-convergence: fit() reduces NLL over epochs (fixed seed, weak tolerance).
"""

import pytest
import torch
import torch.nn as nn

from neural_bamlss.families import NormalFamily
from neural_bamlss.layers import BayesianEmbedding, BayesianIntercept, collect_kl
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.shapes import BayesianMLP

# ── constants ─────────────────────────────────────────────────────────────────

N_OBS = 64  # toy training-set size
BATCH = 16
IN = 1  # one input feature per net (univariate shape function)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def family():
    return NormalFamily()


@pytest.fixture
def single_feature_model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def multi_feature_model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
        "x2": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


@pytest.fixture
def X_single():
    g = torch.Generator().manual_seed(7)
    return {"x1": torch.randn(BATCH, IN, generator=g)}


@pytest.fixture
def X_multi():
    g = torch.Generator().manual_seed(7)
    return {
        "x1": torch.randn(BATCH, IN, generator=g),
        "x2": torch.randn(BATCH, IN, generator=g),
    }


@pytest.fixture
def y():
    g = torch.Generator().manual_seed(13)
    return torch.randn(BATCH, generator=g)


# ── 1. build — AC1 ────────────────────────────────────────────────────────────


def test_build_single_feature(single_feature_model):
    """BayesianNAMLSS builds for a single-feature fully-Bayesian formula."""
    assert isinstance(single_feature_model, nn.Module)


def test_build_multi_feature(multi_feature_model):
    """BayesianNAMLSS builds for a multi-feature fully-Bayesian formula."""
    assert isinstance(multi_feature_model, nn.Module)


# ── 2. forward shape — AC1 ────────────────────────────────────────────────────


def test_forward_returns_distribution(single_feature_model, X_single):
    """forward() returns a torch.distributions.Distribution."""
    dist = single_feature_model(X_single)
    assert isinstance(dist, torch.distributions.Distribution)


def test_forward_batch_shape(single_feature_model, X_single):
    """forward() batch shape matches the input batch size."""
    dist = single_feature_model(X_single)
    assert dist.batch_shape == (BATCH,)


# ── 3. KL/N in loss — AC3 ─────────────────────────────────────────────────────


def test_kl_positive_after_forward(single_feature_model, X_single):
    """collect_kl reaches VariationalDense layers nested inside BayesianMLP."""
    single_feature_model(X_single)
    kl = collect_kl(single_feature_model)
    assert kl.item() > 0.0


def test_loss_exceeds_nll(single_feature_model, X_single, y):
    """Total loss = NLL + KL/N strictly exceeds the pure NLL.

    KL > 0 is the load-bearing claim from spike_kl_propagation.py [D].
    """
    dist = single_feature_model(X_single)
    nll = -dist.log_prob(y).mean().item()
    total = single_feature_model.loss(X_single, y).item()
    # total must exceed nll because KL/N > 0
    assert total > nll


def test_multi_feature_kl_exceeds_single(X_single, X_multi, y, family):
    """Two Bayesian nets produce more KL than one (both are in the walk)."""
    torch.manual_seed(0)
    m1 = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
        },
        family=family,
        n_obs=N_OBS,
    )
    m2 = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(
                IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
            ),
            "x2": BayesianMLP(
                IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
            ),
        },
        family=family,
        n_obs=N_OBS,
    )
    m1(X_single)
    m2(X_multi)
    assert collect_kl(m2).item() > collect_kl(m1).item()


# ── 4. model.loss callable — AC2 ─────────────────────────────────────────────


def test_loss_callable(single_feature_model):
    """model.loss is a callable."""
    assert callable(single_feature_model.loss)


def test_loss_is_scalar(single_feature_model, X_single, y):
    """model.loss returns a scalar tensor with gradient."""
    loss = single_feature_model.loss(X_single, y)
    assert loss.shape == ()
    assert loss.requires_grad


# ── 5. fit() trains to convergence — AC2 ─────────────────────────────────────


def test_fit_reduces_nll(family):
    """fit() on a toy Normal regression reduces NLL over 50 epochs (AC2).

    Fixed seed so the MC draw is reproducible within this model object
    (CLAUDE.md reproducibility caveat: not cross-model).
    Tolerance is loose (rel=0.10) to accommodate MC noise.
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    X = {"x1": torch.randn(N_OBS, IN, generator=g)}
    # y = 2*x + noise — a learnable signal
    y = 2.0 * X["x1"].squeeze(-1) + 0.1 * torch.randn(N_OBS, generator=g)

    model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
        },
        family=family,
        n_obs=N_OBS,
    )

    history = model.fit(X, y, epochs=50, lr=1e-2)

    first_nll = history["nll"][0]
    last_nll = history["nll"][-1]
    # NLL should decrease; allow 10% relative tolerance for MC noise
    assert last_nll < first_nll * 1.10, (
        f"NLL did not decrease: first={first_nll:.4f}, last={last_nll:.4f}"
    )


# ── 6. partial-Bayesian formula — AC4 ────────────────────────────────────────


def test_partial_bayesian_zero_kl_from_deterministic(family):
    """Deterministic term contributes zero KL; Bayesian term contributes > 0.

    Mirrors AC4: BayesianMLP(x1) + MLP(x2) trains, deterministic contributing 0.
    """

    class _DetMLP(nn.Module):
        """Minimal deterministic shape function — no VariationalDense, so zero KL."""

        def __init__(self):
            super().__init__()
            self.net = nn.Linear(IN, family.param_count, bias=False)

        def forward(self, x):
            return self.net(x)

    g = torch.Generator().manual_seed(5)
    X = {
        "x1": torch.randn(BATCH, IN, generator=g),
        "x2": torch.randn(BATCH, IN, generator=g),
    }

    bayesian_net = BayesianMLP(
        IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
    )
    det_net = _DetMLP()

    model = BayesianNAMLSS(
        formula={"x1": bayesian_net, "x2": det_net},
        family=family,
        n_obs=N_OBS,
    )
    model(X)

    kl_from_bayesian_net = collect_kl(bayesian_net)
    kl_total = collect_kl(model)

    # Full-model walk = Bayesian subnet + model intercept; the deterministic
    # term contributes exactly nothing.
    expected = kl_from_bayesian_net.item() + model.intercept.kl.item()
    assert kl_total.item() == pytest.approx(expected, rel=1e-5)
    assert kl_total.item() > 0.0


def test_partial_bayesian_trains(family):
    """Partial-Bayesian formula trains without error (AC4)."""

    class _DetMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(IN, family.param_count, bias=False)

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(1)
    g = torch.Generator().manual_seed(1)
    X = {
        "x1": torch.randn(N_OBS, IN, generator=g),
        "x2": torch.randn(N_OBS, IN, generator=g),
    }
    y = torch.randn(N_OBS, generator=g)

    model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(
                IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
            ),
            "x2": _DetMLP(),
        },
        family=family,
        n_obs=N_OBS,
    )
    history = model.fit(X, y, epochs=5, lr=1e-2)
    assert len(history["loss"]) == 5


# ── 7. feature_dropout — AC5 ──────────────────────────────────────────────────


def test_feature_dropout_defaults_to_zero_when_bayesian(family):
    """feature_dropout defaults to 0 when Bayesian nets are present (AC5)."""
    model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
        },
        family=family,
        n_obs=N_OBS,
    )
    assert model.feature_dropout == 0.0


def test_feature_dropout_overridable(family):
    """feature_dropout can be overridden even when Bayesian nets are present (AC5)."""
    model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
        },
        family=family,
        n_obs=N_OBS,
        feature_dropout=0.3,
    )
    assert model.feature_dropout == pytest.approx(0.3)


def test_feature_dropout_defaults_to_zero_when_embedding_only(family):
    """An embedding-only formula counts as Bayesian for the dropout default.

    BayesianEmbedding is a VariationalLayer but not a VariationalDense; the
    Bayesian-presence check must use the base class so the weight posterior —
    not feature noise — supplies the stochasticity (issue 0064 rationale).
    """
    model = BayesianNAMLSS(
        formula={
            "group": BayesianEmbedding(
                num_embeddings=4,
                embedding_dim=family.param_count,
                kl_divisor=N_OBS,
            )
        },
        family=family,
        n_obs=N_OBS,
    )
    assert model.feature_dropout == 0.0


def test_feature_dropout_defaults_to_nampy_rate_when_deterministic_only(family):
    """Deterministic-only formula defaults feature_dropout to 0.01 (issue 0021)."""
    det_net = nn.Linear(IN, family.param_count, bias=False)
    model = BayesianNAMLSS(
        formula={"x1": det_net},
        family=family,
        n_obs=N_OBS,
    )
    assert model.feature_dropout == pytest.approx(0.01)


def test_feature_dropout_explicit_override_on_deterministic(family):
    """Explicit feature_dropout overrides the NAMpy 0.01 default (issue 0021)."""
    det_net = nn.Linear(IN, family.param_count, bias=False)
    model = BayesianNAMLSS(
        formula={"x1": det_net},
        family=family,
        n_obs=N_OBS,
        feature_dropout=0.0,
    )
    assert model.feature_dropout == pytest.approx(0.0)


def _constant_contribution_model(family, feature_dropout):
    """Two deterministic nets with known contributions: 1.0 and 2.0 at x=1.

    With per-feature dropout (rescale F / #survivors) the train-mode loc can
    only take values in {0, 2, 3, 4}: both kept → 3, only x1 → 1·2/1 = 2,
    only x2 → 2·2/1 = 4, both dropped → 0.  The pre-fix bug (mask never
    applied to individual contributions) could only ever produce {0, 3}.

    Point-mode intercept (zero-init, deterministic) keeps the contribution
    arithmetic exact for the set-of-locs assertion; a variational intercept
    would add reparameterization noise to every draw.
    """
    net1 = nn.Linear(IN, family.param_count, bias=False)
    net2 = nn.Linear(IN, family.param_count, bias=False)
    with torch.no_grad():
        net1.weight.fill_(1.0)
        net2.weight.fill_(2.0)
    return BayesianNAMLSS(
        formula={"x1": net1, "x2": net2},
        family=family,
        n_obs=N_OBS,
        feature_dropout=feature_dropout,
        intercept_mode="point",
    )


def test_feature_dropout_drops_individual_contributions(family):
    """Train-mode dropout zeroes single feature contributions (AC5).

    Asserts the set of realized locs over many draws, not a single draw
    (testing rule: never assert one stochastic draw).  200 draws at p = 0.5
    miss a 1/4-probability outcome with prob (3/4)^200 ≈ 1e-25, so requiring
    all four outcomes is sound under the fixed seed.
    """
    model = _constant_contribution_model(family, feature_dropout=0.5)
    model.train()
    X = {"x1": torch.ones(1, IN), "x2": torch.ones(1, IN)}
    torch.manual_seed(0)
    locs = {round(model(X).mean.item(), 4) for _ in range(200)}
    # Exact values up to float32 rounding — the rescale arithmetic is exact
    # for these integer contributions.
    assert locs == {0.0, 2.0, 3.0, 4.0}


def test_feature_dropout_inactive_in_eval_mode(family):
    """eval() disables feature dropout: loc is the deterministic full sum."""
    model = _constant_contribution_model(family, feature_dropout=0.5)
    model.eval()
    X = {"x1": torch.ones(1, IN), "x2": torch.ones(1, IN)}
    locs = [model(X).mean.item() for _ in range(20)]
    assert locs == pytest.approx([3.0] * 20)


# ── 8. predict_params — single owner of predictor assembly (issue 0060) ───────


@pytest.fixture
def interaction_model(family):
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
        "x1:x2": BayesianMLP(
            2 * IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
        ),
    }
    return BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)


def test_predict_params_shape(multi_feature_model, X_multi, family):
    """predict_params returns the summed predictor, shape (batch, param_count)."""
    params = multi_feature_model.predict_params(X_multi)
    assert params.shape == (BATCH, family.param_count)


def test_predict_params_handles_interaction_key(interaction_model, X_multi, family):
    """predict_params accepts the same X dict as forward for "x1:x2" terms."""
    params = interaction_model.predict_params(X_multi)
    assert params.shape == (BATCH, family.param_count)
    assert torch.isfinite(params).all()


def test_forward_is_family_of_predict_params(multi_feature_model, X_multi, family):
    """forward(X) == family(predict_params(X)) under the same seed.

    Reproducibility holds within one model object (CLAUDE.md seeding rule):
    re-seeding before each call makes the reparameterization draws identical,
    so the two paths must produce the same distribution parameters exactly.
    """
    model = multi_feature_model
    model.eval()
    torch.manual_seed(11)
    dist_forward = model(X_multi)
    torch.manual_seed(11)
    dist_direct = family(model.predict_params(X_multi))
    assert torch.equal(dist_forward.mean, dist_direct.mean)
    assert torch.equal(dist_forward.stddev, dist_direct.stddev)


# ── 9. model-level intercept — the contract the shape functions rely on ───────
# Shape functions ship bias-free output layers ("intercept handled by
# BayesianIntercept"); these tests pin the model's side of that contract
# (issue 0010 wiring).


def test_model_owns_intercept_per_distributional_parameter(
    single_feature_model, family
):
    """The model owns a BayesianIntercept with one scalar per family parameter."""
    assert isinstance(single_feature_model.intercept, BayesianIntercept)
    assert single_feature_model.intercept.units == family.param_count


def test_intercept_shifts_predictor(single_feature_model, X_single):
    """Shifting the intercept location shifts the summed predictor by the same δ.

    Same seed → identical reparameterization draws, so the only change between
    the two passes is the intercept level.  atol=1e-5 covers float32
    reassociation of the additive sum (not MC noise — draws are identical).
    """
    model = single_feature_model
    model.eval()
    torch.manual_seed(3)
    before = model.predict_params(X_single)
    with torch.no_grad():
        model.intercept.loc += 5.0
    torch.manual_seed(3)
    after = model.predict_params(X_single)
    assert torch.allclose(after, before + 5.0, atol=1e-5)


def test_intercept_is_sole_kl_contributor_in_deterministic_formula(family):
    """A deterministic-only formula still carries the intercept's KL.

    The terms are degenerate zero-variance contributors (CONTEXT.md), so the
    model total must equal the intercept's stash exactly — same tensor, no
    float tolerance needed.
    """
    det_net = nn.Linear(IN, family.param_count, bias=False)
    model = BayesianNAMLSS(formula={"x1": det_net}, family=family, n_obs=N_OBS)
    model({"x1": torch.ones(BATCH, IN)})
    kl = collect_kl(model)
    assert kl.item() > 0.0
    assert kl.item() == model.intercept.kl.item()


def test_point_intercept_contributes_zero_kl(family):
    """intercept_mode='point' restores the fully-deterministic zero-KL model."""
    det_net = nn.Linear(IN, family.param_count, bias=False)
    model = BayesianNAMLSS(
        formula={"x1": det_net},
        family=family,
        n_obs=N_OBS,
        intercept_mode="point",
    )
    model({"x1": torch.ones(BATCH, IN)})
    assert collect_kl(model).item() == 0.0


def test_intercept_kl_divisor_wired_from_n_obs(single_feature_model):
    """n_obs given at construction becomes the intercept's KL/N divisor (ADR-0001)."""
    assert single_feature_model.intercept.kl_divisor == float(N_OBS)


def test_intercept_kl_divisor_updated_when_fit_infers_n_obs(family):
    """fit() inferring n_obs must also rewire the intercept's KL/N divisor.

    Shape functions own their kl_divisor at construction; the model-owned
    intercept is the one layer whose divisor the model itself must keep
    consistent when N arrives late.
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    X = {"x1": torch.randn(N_OBS, IN, generator=g)}
    y = torch.randn(N_OBS, generator=g)
    model = BayesianNAMLSS(
        formula={
            "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS)
        },
        family=family,  # n_obs deliberately omitted
    )
    assert model.intercept.kl_divisor == 1.0
    model.fit(X, y, epochs=1)
    assert model.intercept.kl_divisor == float(N_OBS)


# ── predict_params sample-dimension pass (issue 0027 / GitHub #80) ───────────


def test_predict_params_sample_dim_shape(family):
    """Expanded (S, n, in) inputs yield an (S, n, param_count) summed predictor,
    including interaction keys (concatenated along the last dim)."""
    formula = {
        "x1": BayesianMLP(IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS),
        "x1:x2": BayesianMLP(
            2 * IN, family.param_count, hidden_dims=[8], kl_divisor=N_OBS
        ),
    }
    model = BayesianNAMLSS(formula=formula, family=family, n_obs=N_OBS)
    model.eval()
    S = 4
    g = torch.Generator().manual_seed(11)
    X = {
        "x1": torch.randn(BATCH, IN, generator=g).expand(S, BATCH, IN),
        "x2": torch.randn(BATCH, IN, generator=g).expand(S, BATCH, IN),
    }
    with torch.no_grad():
        out = model.predict_params(X)
    assert out.shape == (S, BATCH, family.param_count)


def test_predict_params_sample_dim_intercept_is_fresh_per_draw(family):
    """The intercept is part of the posterior draw — fresh per sample slice.

    With deterministic shape functions, all cross-sample variation comes from
    the variational intercept alone: identical slices would mean one intercept
    draw was broadcast across the sample dimension (the issue-0027 violation).
    """
    from neural_bamlss.shapes import DeterministicMLP

    torch.manual_seed(12)
    formula = {"x1": DeterministicMLP(IN, family.param_count, hidden_dims=[8])}
    model = BayesianNAMLSS(
        formula=formula, family=family, n_obs=N_OBS, feature_dropout=0.0
    )
    model.eval()
    S = 4
    x = torch.randn(BATCH, IN).expand(S, BATCH, IN)
    with torch.no_grad():
        out = model.predict_params({"x1": x})
    assert out.shape == (S, BATCH, family.param_count)
    for s in range(1, S):
        assert not torch.equal(out[0], out[s]), (
            f"slice {s} equals slice 0 — intercept draw broadcast across samples"
        )
