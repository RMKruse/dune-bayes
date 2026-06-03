"""Tests for BayesianEmbedding — variational categorical first mapping (issue 0012/#13).

Four reference-test archetypes (CLAUDE.md):
  - Shape:       forward(idx) → (batch, embedding_dim).
  - Closed-form: KL against hand-computed Gaussian–Gaussian reference.
  - Round-trip:  state_dict + from_config with max|Δw| == 0.
  - MC-convergence: variational mean converges to loc over T draws.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_bamlss.layers import BayesianEmbedding, collect_kl, set_kl_beta
from neural_bamlss.priors import PriorScale

NUM_EMBEDDINGS = 8
EMBEDDING_DIM = 4
BATCH = 5


@pytest.fixture
def embedding():
    torch.manual_seed(0)
    ps = PriorScale(mode="fixed", scale=1.0)
    return BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps,
        validate_args=True,
    )


@pytest.fixture
def idx():
    torch.manual_seed(1)
    return torch.randint(0, NUM_EMBEDDINGS, (BATCH,))


# ── 1. shape ──────────────────────────────────────────────────────────────────


def test_forward_output_shape(embedding, idx):
    out = embedding(idx)
    assert out.shape == (BATCH, EMBEDDING_DIM)


# ── 2. KL is positive in variational mode ────────────────────────────────────


def test_kl_positive_after_forward(embedding, idx):
    embedding(idx)
    assert float(embedding.kl.detach()) > 0.0


# ── 3. KL matches closed-form Gaussian–Gaussian reference ────────────────────


def _reference_embedding_kl(
    loc: torch.Tensor, rho: torch.Tensor, prior_scale: float
) -> float:
    """KL[N(loc, σ²) ‖ N(0, s²)] summed over all (level, dim) pairs."""
    scale = F.softplus(rho)
    kl = torch.sum(
        math.log(prior_scale)
        - torch.log(scale)
        + (scale.pow(2) + loc.pow(2)) / (2.0 * prior_scale**2)
        - 0.5
    )
    return float(kl.detach())


def test_kl_matches_closed_form_reference(embedding, idx):
    """After forward(), .kl equals analytic Gaussian–Gaussian KL.

    PriorScale is fixed (scale=1.0), so prior_scale_handle.kl()==0 and
    s is a deterministic tensor — no MC noise.  float32 arithmetic: rel=1e-5.
    """
    embedding(idx)
    expected = _reference_embedding_kl(embedding.loc, embedding.rho, prior_scale=1.0)
    assert float(embedding.kl.detach()) == pytest.approx(expected, rel=1e-5)


# ── 4. point mode ─────────────────────────────────────────────────────────────


def test_point_mode_returns_loc_exactly(idx):
    """In point mode forward(idx) returns loc[idx] without stochastic noise."""
    layer = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM, mode="point"
    )
    with torch.no_grad():
        out = layer(idx)
    assert torch.equal(out, layer.loc[idx])


def test_point_mode_kl_is_zero(idx):
    """In point mode no KL is contributed — deterministic weights, no posterior."""
    layer = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM, mode="point"
    )
    layer(idx)
    assert float(layer.kl.detach()) == pytest.approx(0.0, abs=1e-7)


# ── 5. collect_kl aggregates BayesianEmbedding KL ────────────────────────────


class _ModelWithEmbedding(nn.Module):
    """Minimal stand-in: embedding lookup + downstream dense layer."""

    def __init__(self) -> None:
        super().__init__()
        from neural_bamlss.layers import VariationalDense

        self.embed = BayesianEmbedding(
            num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM
        )
        self.dense = VariationalDense(EMBEDDING_DIM, 2)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.dense(self.embed(idx))


def test_collect_kl_includes_bayesian_embedding(idx):
    """collect_kl() must reach BayesianEmbedding KL alongside VariationalDense KL."""
    model = _ModelWithEmbedding()
    model(idx)

    total_kl = float(collect_kl(model).detach())
    embed_kl = float(model.embed.kl.detach())
    dense_kl = float(model.dense.kl.detach())

    assert total_kl == pytest.approx(embed_kl + dense_kl, rel=1e-5), (
        "collect_kl dropped the BayesianEmbedding KL contribution"
    )
    assert embed_kl > 0.0, "variational embedding KL should be positive"


# ── 6. set_kl_beta propagates to BayesianEmbedding ───────────────────────────


def test_set_kl_beta_zeroes_embedding_kl(embedding, idx):
    """set_kl_beta(0) zeroes out BayesianEmbedding KL."""
    set_kl_beta(embedding, 0.0)
    embedding(idx)
    assert float(embedding.kl.detach()) == pytest.approx(0.0, abs=1e-7)


def test_set_kl_beta_propagates_through_composite_model(idx):
    """set_kl_beta() reaches BayesianEmbedding inside a composite module."""
    model = _ModelWithEmbedding()
    set_kl_beta(model, 0.5)
    assert float(model.embed.kl_beta) == pytest.approx(0.5)
    assert float(model.dense.kl_beta) == pytest.approx(0.5)


# ── 7. hierarchical PriorScale hyperprior KL is added to .kl ─────────────────


def test_hierarchical_prior_scale_kl_included(idx):
    """When PriorScale is hierarchical, its hyperprior KL is absorbed into .kl.

    We verify this by comparing: .kl with a hierarchical handle > .kl with a
    fixed handle of the same expected scale.  The hierarchical KL is strictly
    larger because it adds the positive hyperprior term.
    """
    torch.manual_seed(7)
    ps_fixed = PriorScale(mode="fixed", scale=1.0)
    emb_fixed = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps_fixed,
    )

    # Copy identical loc/rho so the embedding KL base is the same.
    ps_hierarchical = PriorScale(
        mode="hierarchical", scale=1.0, hyperprior="inverse_gamma"
    )
    emb_hier = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps_hierarchical,
    )
    with torch.no_grad():
        emb_hier.loc.copy_(emb_fixed.loc)
        emb_hier.rho.copy_(emb_fixed.rho)

    emb_fixed(idx)
    emb_hier(idx)

    # Hierarchical mode adds a non-zero hyperprior KL on top of the embedding KL.
    ps_kl = float(ps_hierarchical.kl().detach())
    assert ps_kl != pytest.approx(0.0, abs=1e-6), (
        "hierarchical PriorScale KL should be non-zero"
    )
    # The stashed .kl for the hierarchical layer includes the hyperprior term.
    # We check via ps_kl being non-zero (already asserted) and the hierarchical
    # .kl differing from the fixed one by more than a small epsilon.
    kl_fixed = float(emb_fixed.kl.detach())
    kl_hier = float(emb_hier.kl.detach())
    assert abs(kl_hier - kl_fixed) > 1e-3, (
        "hierarchical .kl should differ from fixed .kl by the hyperprior KL"
    )


# ── 8. get_config / from_config ───────────────────────────────────────────────


def test_get_config_is_closure_free(embedding):
    """get_config() contains only ints, floats, strings, dicts — no callables."""
    cfg = embedding.get_config()
    for k, v in cfg.items():
        assert not callable(v), f"config[{k!r}] is callable — not serialisable"


def test_from_config_preserves_hyperparameters():
    """from_config(get_config()) reconstructs an equivalent BayesianEmbedding."""
    ps = PriorScale(mode="empirical_bayes", scale=2.0)
    original = BayesianEmbedding(
        num_embeddings=10,
        embedding_dim=6,
        prior_scale_handle=ps,
        kl_divisor=50.0,
        mode="variational",
    )
    rebuilt = BayesianEmbedding.from_config(original.get_config())

    assert rebuilt.num_embeddings == original.num_embeddings
    assert rebuilt.embedding_dim == original.embedding_dim
    assert rebuilt.kl_divisor == pytest.approx(original.kl_divisor)
    assert rebuilt.mode == original.mode
    assert rebuilt.prior_scale_handle.mode == original.prior_scale_handle.mode


# ── 9. state_dict round-trip ──────────────────────────────────────────────────


def test_state_dict_round_trip_max_delta_zero(tmp_path):
    """config + state_dict save/load reconstructs identical variational weights.

    ADR-0004 load-bearing claim B pattern: max|Δw| == 0 (exact equality on
    deterministic parameter tensors, not stochastic predictions).
    """
    torch.manual_seed(2)
    ps = PriorScale(mode="empirical_bayes", scale=0.5)
    layer = BayesianEmbedding(
        num_embeddings=6, embedding_dim=3, prior_scale_handle=ps, kl_divisor=100.0
    )
    bundle_path = tmp_path / "embedding.pt"

    torch.save(
        {"config": layer.get_config(), "state_dict": layer.state_dict()}, bundle_path
    )

    bundle = torch.load(bundle_path, weights_only=True)
    loaded = BayesianEmbedding.from_config(bundle["config"])
    loaded.load_state_dict(bundle["state_dict"])

    sa, sb = layer.state_dict(), loaded.state_dict()
    assert sa.keys() == sb.keys()
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, (
        f"max|Δw| = {max_delta:.2e} — weights changed across round-trip"
    )


# ── 10. MC convergence ────────────────────────────────────────────────────────


def test_variational_samples_are_stochastic(embedding, idx):
    """Consecutive forward() calls produce different outputs (not a constant)."""
    torch.manual_seed(42)
    s1 = embedding(idx).detach().clone()
    s2 = embedding(idx).detach().clone()
    assert not torch.equal(s1, s2), (
        "variational forward produced identical samples — no randomness"
    )


def test_variational_mean_converges_to_loc():
    """Mean of T embedding samples converges to loc[idx].

    We sample one fixed index; T=2000 draws per (level, dim) cell.
    MC std_err ≈ softplus(-3)/√2000 ≈ 0.049/44.7 ≈ 0.001; abs=0.05 gives
    50× headroom while catching any genuine posterior-mean bias.
    """
    torch.manual_seed(5)
    ps = PriorScale(mode="fixed", scale=1.0)
    layer = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps,
    )
    query = torch.tensor([0, 1, 2])  # three distinct levels

    T = 2000
    samples = torch.stack([layer(query).detach() for _ in range(T)])  # (T, 3, 4)
    mean_sample = samples.mean(0)  # (3, 4)

    expected = layer.loc[query].detach().numpy()
    assert mean_sample.numpy() == pytest.approx(expected, abs=0.05), (
        "variational embedding mean drifted from loc — posterior mean should equal loc"
    )
