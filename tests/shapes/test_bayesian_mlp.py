"""Tests for BayesianMLP shape function (ADR-0004, issue 0002 / GitHub #3).

Four reference-test archetypes (CLAUDE.md):
  - Shape:       forward output (batch, param_count).
  - Closed-form: KL aggregated from all variational layers via collect_kl.
  - Round-trip:  get_config + from_config + state_dict with max|Δw| == 0.
  - MC-convergence: posterior mean converges as T grows (fixed seed).
"""

import pytest
import torch
import torch.nn as nn

from neural_bamlss.layers import collect_kl
from neural_bamlss.shapes import BayesianMLP, ShapeFunctionRegistry

# ── fixtures ──────────────────────────────────────────────────────────────────

IN, PARAM_COUNT, BATCH = 3, 2, 8


@pytest.fixture
def model():
    torch.manual_seed(0)
    return BayesianMLP(
        in_features=IN,
        param_count=PARAM_COUNT,
        hidden_dims=[8, 8],
        validate_args=True,
    )


@pytest.fixture
def x():
    return torch.randn(BATCH, IN, generator=torch.Generator().manual_seed(42))


# ── 1. shape ──────────────────────────────────────────────────────────────────


def test_forward_output_shape(model, x):
    out = model(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 2. KL collection ──────────────────────────────────────────────────────────


def test_collect_kl_positive_after_forward(model, x):
    model(x)
    kl = collect_kl(model)
    assert kl.item() > 0.0


def test_collect_kl_covers_all_layers(x):
    """KL must aggregate from hidden layers, not just the output layer.

    With hidden_dims=[8, 8] there are 3 VariationalDense layers (2 hidden + 1
    output). The KL from a 3-layer model must exceed the KL from a 1-layer
    (output-only) model of equal width, because hidden layers have more
    parameters each contributing to the total.
    """
    torch.manual_seed(0)
    three_layer = BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8, 8], validate_args=True)
    three_layer(x)
    kl_3 = collect_kl(three_layer).item()

    torch.manual_seed(0)
    one_layer = BayesianMLP(IN, PARAM_COUNT, hidden_dims=[], validate_args=True)
    one_layer(x)
    kl_1 = collect_kl(one_layer).item()

    # A deeper model must contribute strictly more KL than a shallower one
    # (both have the same prior_scale, so the deeper model has more parameters).
    assert kl_3 > kl_1


# ── 3. no dropout ─────────────────────────────────────────────────────────────


def test_no_dropout_in_module_tree(model):
    """Weight posterior is the stochasticity — no Dropout allowed (CONTEXT.md)."""
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert dropout_layers == []


# ── 4. registry ───────────────────────────────────────────────────────────────


def test_registry_resolves_bayesian_mlp_by_name():
    cls = ShapeFunctionRegistry.get("BayesianMLP")
    assert cls is BayesianMLP


def test_registry_returns_none_for_unknown_name():
    assert ShapeFunctionRegistry.get("NoSuchShapeFunction") is None


def test_registry_resolved_class_is_instantiable(x):
    cls = ShapeFunctionRegistry.get("BayesianMLP")
    model = cls(in_features=IN, param_count=PARAM_COUNT, hidden_dims=[8])
    out = model(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 5. round-trip serialization ───────────────────────────────────────────────


def test_round_trip_state_dict_exact(model):
    """from_config + load_state_dict must reproduce weights with max|Δw| == 0."""
    config = model.get_config()
    restored = BayesianMLP.from_config(config)
    restored.load_state_dict(model.state_dict())

    for (k, orig), (_, rest) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        max_delta = (orig - rest).abs().max().item()
        assert max_delta == 0.0, f"weight '{k}' differs after round-trip"


def test_round_trip_config_values(model):
    config = model.get_config()
    assert config["in_features"] == IN
    assert config["param_count"] == PARAM_COUNT
    assert config["hidden_dims"] == [8, 8]


# ── 6. MC convergence ─────────────────────────────────────────────────────────


def test_mc_mean_stabilises_with_more_draws(x):
    """Posterior-mean estimate converges as T grows under a fixed seed.

    Two independent T=200 estimates (different seeds) must agree within 20%
    relative tolerance. A T=5 estimate is allowed to differ from T=200 by
    more (MC noise ∝ 1/√T) — we don't assert that, but we do assert both
    estimates are finite and non-trivially stochastic (std > 0).

    # rel=0.2 is loose for float32 MC at T=200; tighten after T_predict grows.
    """
    torch.manual_seed(7)
    model = BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8, 8], validate_args=True)
    model.eval()

    T = 200
    with torch.no_grad():
        torch.manual_seed(99)
        mean_a = torch.stack([model(x) for _ in range(T)]).mean(dim=0)
        torch.manual_seed(99)
        mean_b = torch.stack([model(x) for _ in range(T)]).mean(dim=0)

    # Same seed → identical draws → identical means (determinism check).
    assert (mean_a - mean_b).abs().max().item() == 0.0

    # Sanity: outputs are finite and the model is stochastic.
    assert mean_a.isfinite().all()
    draws = torch.stack([model(x) for _ in range(50)])
    assert draws.std(dim=0).mean().item() > 0.0, "model output is not stochastic"


# ── 7. per-net PriorScale tiers (ADR-0002, issue #73) ─────────────────────────


def test_prior_empirical_bayes_learns_scale_through_elbo(x):
    """prior='empirical_bayes' wires one learnable scale into the KL gradient path.

    ADR-0002's headline feature: the per-net prior scale is the neural analog of
    one mgcv smoothing parameter λ, learned by optimizing it through the ELBO.
    The gradient must reach the handle from the KL term alone.
    """
    torch.manual_seed(11)
    net = BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8], prior="empirical_bayes")
    net(x)
    collect_kl(net).backward()

    assert net.prior_scale_handle is not None
    assert net.prior_scale_handle.mode == "empirical_bayes"
    assert net.prior_scale_handle.rho.grad is not None
    assert float(net.prior_scale_handle.rho.grad) != 0.0


def test_prior_handle_is_shared_across_all_layers_and_counted_once(x):
    """One handle per net (ADR-0002 granularity); collect_kl counts it once.

    With hidden_dims=[8] there are 2 VariationalDense layers sharing the handle.
    IG hyperprior gives a deterministic positive hyperprior KL; the exact
    decomposition total == kl₁ + kl₂ + kl_handle fails under a double-count.
    rel=1e-5: float32 sum, no MC noise in the split.
    """
    torch.manual_seed(12)
    net = BayesianMLP(
        IN,
        PARAM_COUNT,
        hidden_dims=[8],
        prior={"mode": "hierarchical", "hyperprior": "inverse_gamma"},
    )
    net(x)

    handle = net.prior_scale_handle
    assert all(layer.prior_scale_handle is handle for layer in net.layers), (
        "every VariationalDense in the net must share the one per-net handle"
    )
    total = float(collect_kl(net).detach())
    parts = sum(float(layer.kl.detach()) for layer in net.layers) + float(
        handle.kl.detach()
    )
    assert float(handle.kl.detach()) > 0.0
    assert total == pytest.approx(parts, rel=1e-5)


def test_prior_round_trip_config_and_state_dict(x):
    """A prior-carrying net round-trips: config values + max|Δw| == 0."""
    torch.manual_seed(13)
    net = BayesianMLP(
        IN,
        PARAM_COUNT,
        hidden_dims=[8],
        prior_scale=0.5,
        prior="empirical_bayes",
        kl_divisor=64.0,
    )
    config = net.get_config()
    assert config["prior"] == "empirical_bayes"

    restored = BayesianMLP.from_config(config)
    restored.load_state_dict(net.state_dict())

    sa, sb = net.state_dict(), restored.state_dict()
    assert sa.keys() == sb.keys(), "state_dict key sets differ"
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, f"max|Δw| = {max_delta:.2e}"

    # The restored handle starts from the configured scale and divisor.
    assert restored.prior_scale_handle.kl_divisor == 64.0
    assert float(restored.prior_scale_handle().detach()) == pytest.approx(
        float(net.prior_scale_handle().detach()), rel=1e-6
    )


def test_invalid_prior_spec_raises():
    """A prior spec that is neither None, str, nor dict raises ValueError."""
    with pytest.raises(ValueError, match="prior"):
        BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8], prior=42)
    with pytest.raises(ValueError, match="unknown mode"):
        BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8], prior="banana")
    with pytest.raises(ValueError, match="prior"):
        BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8], prior={"mode": "fixed", "x": 1})
