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

from dune_bayes.layers import BayesianEmbedding, collect_kl, set_kl_beta
from dune_bayes.priors import PriorScale

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

    PriorScale is fixed (scale=1.0), so its hyperprior KL is zero and
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
        from dune_bayes.layers import VariationalDense

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


# ── 7. hierarchical PriorScale hyperprior KL reaches collect_kl ──────────────


def test_hierarchical_hyperprior_kl_reaches_collect_kl(idx):
    """The hyperprior KL is stashed on the handle itself and found by collect_kl.

    Since issue #73 the handle is a VariationalLayer: the embedding stashes only
    its Gaussian–Gaussian KL, the handle stashes its own hyperprior KL, and
    collect_kl over the embedding (which owns the handle as a submodule) sums
    exactly the two.  rel=1e-5: float32 sum, no extra MC noise in the split.
    """
    torch.manual_seed(7)
    ps = PriorScale(mode="hierarchical", scale=1.0, hyperprior="inverse_gamma")
    emb = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps,
    )
    emb(idx)

    assert float(ps.kl.detach()) > 0.0, "hierarchical hyperprior KL should be > 0"
    total = float(collect_kl(emb).detach())
    assert total == pytest.approx(
        float(emb.kl.detach()) + float(ps.kl.detach()), rel=1e-5
    )


def test_embedding_kl_excludes_hyperprior_no_double_count(idx):
    """The embedding stash holds only the Gaussian–Gaussian KL — no hyperprior.

    Regression for issue #73: before, the embedding added the handle's
    hyperprior KL into its own stash; with the handle stashing it too that
    would double-count.  Construction makes the check exact:

      - rho_s = -20 → sigma_s = softplus(-20) ≈ 2e-9, loc_s = log(1.0) = 0,
        so the sampled s == 1.0 to float32 precision (deterministic), and the
        embedding KL has a closed-form reference at prior scale 1.0;
      - alpha0 = 200 makes the hyperprior KL huge (lgamma(200) ≈ 857), so a
        double-count would inflate the embedding stash catastrophically.

    rel=1e-4: float32 arithmetic plus the ~1e-9 jitter in the sampled s.
    """
    ps = PriorScale(
        mode="hierarchical", scale=1.0, hyperprior="inverse_gamma", alpha0=200.0
    )
    with torch.no_grad():
        ps.rho_s.fill_(-20.0)
    emb = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps,
    )
    emb(idx)

    expected = _reference_embedding_kl(emb.loc, emb.rho, prior_scale=1.0)
    assert float(emb.kl.detach()) == pytest.approx(expected, rel=1e-4), (
        "embedding .kl must hold only the Gaussian–Gaussian KL "
        "(hyperprior KL lives on the handle)"
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


# ── sample-dimension draws (issue 0027 / GitHub #80) ─────────────────────────


def test_sample_dim_slices_are_independent_draws(embedding):
    """An expanded (S, batch) idx yields S independent draws, not one broadcast.

    The vectorized sweeps expand the index tensor along a leading sample
    dimension; each slice must carry fresh embedding noise.  Identical idx
    rows with one shared table draw would make all slices equal — the
    independence violation issue 0027 forbids.
    """
    torch.manual_seed(7)
    S = 4
    idx = torch.randint(0, NUM_EMBEDDINGS, (BATCH,)).expand(S, BATCH)
    with torch.no_grad():
        out = embedding(idx)
    assert out.shape == (S, BATCH, EMBEDDING_DIM)
    for s in range(1, S):
        assert not torch.equal(out[0], out[s]), (
            f"slice {s} equals slice 0 — embedding draw broadcast across samples"
        )


def test_eval_mode_draws_are_coherent_across_repeated_levels(embedding):
    """In eval mode, repeated levels share one embedding draw (ADR-0007).

    A coherent posterior draw is one table realization W ~ q(W) gathered at
    every index: two rows with the same level must be identical within the
    draw. (Training keeps per-element local-reparam noise for gradient
    variance reduction — the embedding analog of VariationalDense's split.)
    Across separate calls the draws must differ, or the "draw" is just the
    posterior mean.
    """
    embedding.eval()
    idx = torch.tensor([2, 5, 2, 2, 5])
    with torch.no_grad():
        out = embedding(idx)
        out2 = embedding(idx)
    assert torch.equal(out[0], out[2]) and torch.equal(out[0], out[3]), (
        "repeated level 2 maps to different embeddings within one draw"
    )
    assert torch.equal(out[1], out[4]), (
        "repeated level 5 maps to different embeddings within one draw"
    )
    # With softplus(_RHO_INIT) ≈ 0.05 posterior scale, two independent draws
    # coinciding to float32 equality has probability ~0.
    assert not torch.equal(out, out2), "two eval draws identical — not stochastic"


def test_eval_mode_sample_dim_slices_are_coherent_independent_draws(embedding):
    """Eval + (S, batch) idx: one table draw per slice, fresh across slices.

    The sample-dimension contract (issue 0027) under the ADR-0007 split:
    each slice s is ONE coherent table realization (repeated levels equal
    within the slice) and slices carry independent draws (not one draw
    broadcast S ways).
    """
    embedding.eval()
    S = 4
    idx = torch.tensor([2, 5, 2, 2, 5]).expand(S, 5)
    with torch.no_grad():
        out = embedding(idx)  # (S, 5, EMBEDDING_DIM)
    assert out.shape == (S, 5, EMBEDDING_DIM)
    for s in range(S):
        assert torch.equal(out[s, 0], out[s, 2]), (
            f"slice {s}: repeated level differs within one draw"
        )
    for s in range(1, S):
        assert not torch.equal(out[0], out[s]), (
            f"slice {s} equals slice 0 — one table draw broadcast across samples"
        )


def test_sample_dim_marginal_matches_posterior():
    """Per-element draws follow N(loc[idx], softplus(rho)[idx]²) — MC convergence.

    Local reparameterization (per-element fresh noise, training path per
    ADR-0007) must leave each element's marginal posterior exact; ELBO, WAIC
    pointwise terms, and ribbon quantiles consume only these marginals.
    Tolerances (MC noise): S=4000 draws; mean std_err ≈ 0.049/√4000 ≈ 8e-4,
    abs=0.01 gives ~12× headroom.  Sample-std rel error ≈ 1/√(2S) ≈ 1.1%;
    rel=0.15 is stable under the fixed seed and catches a shared-draw bug
    (cross-sample std collapses to 0) or a wrong-scale bug.
    """
    torch.manual_seed(8)
    ps = PriorScale(mode="fixed", scale=1.0)
    layer = BayesianEmbedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        prior_scale_handle=ps,
        validate_args=True,
    )
    S = 4000
    query = torch.tensor([0, 1, 2]).expand(S, 3)
    with torch.no_grad():
        out = layer(query)  # (S, 3, EMBEDDING_DIM)
        loc_ref = layer.loc[query[0]]
        scale_ref = torch.nn.functional.softplus(layer.rho)[query[0]]

    assert out.mean(dim=0) == pytest.approx(loc_ref.numpy(), abs=0.01)
    assert out.std(dim=0) == pytest.approx(scale_ref.numpy(), rel=0.15)
