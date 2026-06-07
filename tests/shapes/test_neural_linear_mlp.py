"""Tests for NeuralLinearMLP shape function (ADR-0004, issue 0014 / GitHub #15).

Four reference-test archetypes (CLAUDE.md):
  - Shape:       forward output (batch, param_count).
  - Structural:  hidden layers deterministic (nn.Linear), single variational output.
  - Closed-form: KL from collect_kl equals KL from the single output layer only.
  - Round-trip:  get_config + from_config + state_dict with max|Δw| == 0.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dune_bayes.layers import VariationalDense, collect_kl
from dune_bayes.shapes import NeuralLinearMLP, ShapeFunctionRegistry

# ── fixtures ──────────────────────────────────────────────────────────────────

IN, PARAM_COUNT, BATCH = 3, 2, 8


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NeuralLinearMLP(
        in_features=IN,
        param_count=PARAM_COUNT,
        hidden_dims=[8, 8],
        validate_args=True,
    )


@pytest.fixture
def x():
    return torch.randn(BATCH, IN, generator=torch.Generator().manual_seed(42))


def test_local_reparam_default_on(model):
    """Bayesian shape functions default to local_reparam=True (ADR-0007)."""
    assert model.local_reparam is True
    assert model.get_config()["local_reparam"] is True


# ── 1. shape ──────────────────────────────────────────────────────────────────


def test_forward_output_shape(model, x):
    out = model(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 2. structural: deterministic hidden + single variational output ────────────


def test_hidden_layers_are_nn_linear(model):
    """Hidden layers must be deterministic nn.Linear — no VariationalDense (AC 2)."""
    for layer in model.hidden_layers:
        assert isinstance(layer, nn.Linear), f"hidden layer {layer} is not nn.Linear"
        assert not isinstance(layer, VariationalDense)


def test_exactly_one_variational_dense(model):
    """Only output layer is variational; hidden layers contribute zero KL (AC 2, 3)."""
    vd_layers = [m for m in model.modules() if isinstance(m, VariationalDense)]
    assert len(vd_layers) == 1
    assert vd_layers[0] is model.output_layer


def test_no_dropout_in_module_tree(model):
    """Weight posterior is the stochasticity — no Dropout allowed (CONTEXT.md)."""
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert dropout_layers == []


# ── 3. KL collection (output-layer-only) ──────────────────────────────────────


def test_collect_kl_positive_after_forward(model, x):
    """KL must be > 0 after a forward pass (variational output layer is active)."""
    model(x)
    kl = collect_kl(model)
    assert kl.item() > 0.0


def test_kl_equals_output_layer_kl(model, x):
    """collect_kl must equal the output layer's KL only — hidden layers contribute zero.

    # Exact equality: output_layer.kl is the single source of KL in the model.
    """
    model(x)
    total_kl = collect_kl(model).item()
    output_kl = model.output_layer.kl.item()
    assert total_kl == pytest.approx(output_kl, rel=1e-6)


# ── 4. registry ───────────────────────────────────────────────────────────────


def test_registry_resolves_neural_linear_mlp_by_name():
    cls = ShapeFunctionRegistry.get("NeuralLinearMLP")
    assert cls is NeuralLinearMLP


def test_registry_resolved_class_is_instantiable(x):
    cls = ShapeFunctionRegistry.get("NeuralLinearMLP")
    model = cls(in_features=IN, param_count=PARAM_COUNT, hidden_dims=[8])
    out = model(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 5. KL comparison vs fully variational ─────────────────────────────────────


def test_kl_strictly_less_than_bayesian_mlp(x):
    """NeuralLinearMLP KL < BayesianMLP KL of same depth (fewer variational params).

    BayesianMLP with hidden_dims=[8, 8] has 3 VariationalDense layers;
    NeuralLinearMLP has only 1 — so its total KL must be strictly smaller.
    """
    from dune_bayes.shapes import BayesianMLP

    torch.manual_seed(0)
    fully_var = BayesianMLP(IN, PARAM_COUNT, hidden_dims=[8, 8], validate_args=True)
    fully_var(x)
    kl_full = collect_kl(fully_var).item()

    torch.manual_seed(0)
    last_only = NeuralLinearMLP(IN, PARAM_COUNT, hidden_dims=[8, 8], validate_args=True)
    last_only(x)
    kl_last = collect_kl(last_only).item()

    assert kl_last < kl_full


# ── 6. round-trip serialization ───────────────────────────────────────────────


def test_round_trip_state_dict_exact(model):
    """from_config + load_state_dict must reproduce weights with max|Δw| == 0."""
    config = model.get_config()
    restored = NeuralLinearMLP.from_config(config)
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
    assert config["activation"] == "relu"


# ── 7. last-layer-only uncertainty ────────────────────────────────────────────


def test_output_is_stochastic(model, x):
    """NeuralLinearMLP must be stochastic overall (output layer is variational).

    Two consecutive forward calls with the same input must differ.
    """
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert not torch.allclose(out1, out2), "model output is not stochastic"


def test_hidden_basis_is_deterministic(model, x):
    """Deterministic hidden layers must produce identical output across calls (AC 4).

    Stochasticity is confined to the output VariationalDense — not the basis.
    """
    model.eval()
    act = F.relu  # activation matches the default "relu"

    def compute_basis(inp: torch.Tensor) -> torch.Tensor:
        h = inp
        for layer in model.hidden_layers:
            h = act(layer(h))
        return h

    with torch.no_grad():
        basis1 = compute_basis(x)
        basis2 = compute_basis(x)

    # Exact equality: deterministic nn.Linear → identical intermediate features.
    assert (basis1 - basis2).abs().max().item() == 0.0


# ── per-net PriorScale tiers (ADR-0002, issue #73) ────────────────────────────


def test_prior_empirical_bayes_learns_scale_through_elbo(x):
    """prior='empirical_bayes' wires the learnable scale into the output-layer KL.

    Only the output layer is variational here, so the EB gradient path runs
    through that single layer's KL (ADR-0002's REML analog).
    """
    torch.manual_seed(21)
    net = NeuralLinearMLP(IN, PARAM_COUNT, hidden_dims=[8], prior="empirical_bayes")
    net(x)
    collect_kl(net).backward()

    assert net.prior_scale_handle is not None
    assert net.output_layer.prior_scale_handle is net.prior_scale_handle
    assert net.prior_scale_handle.rho.grad is not None
    assert float(net.prior_scale_handle.rho.grad) != 0.0


def test_prior_round_trip_config_and_state_dict(x):
    """A prior-carrying net round-trips: config values + max|Δw| == 0."""
    torch.manual_seed(22)
    net = NeuralLinearMLP(
        IN,
        PARAM_COUNT,
        hidden_dims=[8],
        prior_scale=0.5,
        prior={"mode": "hierarchical", "hyperprior": "inverse_gamma"},
        kl_divisor=32.0,
    )
    config = net.get_config()
    assert config["prior"] == {"mode": "hierarchical", "hyperprior": "inverse_gamma"}

    restored = NeuralLinearMLP.from_config(config)
    restored.load_state_dict(net.state_dict())

    sa, sb = net.state_dict(), restored.state_dict()
    assert sa.keys() == sb.keys(), "state_dict key sets differ"
    max_delta = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    assert max_delta == 0.0, f"max|Δw| = {max_delta:.2e}"
