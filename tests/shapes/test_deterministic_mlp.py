"""Tests for DeterministicMLP shape function (issue 0020 / GitHub #39).

Four reference-test archetypes (CLAUDE.md):
  - Shape:      forward output (batch, param_count).
  - Structural: no VariationalDense — all layers are nn.Linear.
  - Zero-KL:    collect_kl returns exactly 0.0 (no variational parameters).
  - Determinism: same input → identical output on every call.
  - Round-trip: get_config + from_config + state_dict with max|Δw| == 0.
"""

import pytest
import torch
import torch.nn as nn

from neural_bamlss.layers import collect_kl
from neural_bamlss.layers.variational_dense import VariationalDense
from neural_bamlss.shapes import ShapeFunctionRegistry
from neural_bamlss.shapes.deterministic_mlp import DeterministicMLP

IN, PARAM_COUNT, BATCH = 3, 2, 8


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DeterministicMLP(in_features=IN, param_count=PARAM_COUNT, hidden_dims=[8, 8])


@pytest.fixture
def x():
    return torch.randn(BATCH, IN, generator=torch.Generator().manual_seed(42))


# ── 1. registry ───────────────────────────────────────────────────────────────


def test_registry_resolves_mlp_by_name():
    cls = ShapeFunctionRegistry.get("MLP")
    assert cls is DeterministicMLP


def test_registry_resolved_class_is_instantiable(x):
    cls = ShapeFunctionRegistry.get("MLP")
    m = cls(in_features=IN, param_count=PARAM_COUNT, hidden_dims=[8])
    out = m(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 2. output shape ───────────────────────────────────────────────────────────


def test_forward_output_shape(model, x):
    out = model(x)
    assert out.shape == (BATCH, PARAM_COUNT)


def test_forward_output_shape_no_hidden(x):
    """Zero hidden layers: direct linear projection in_features → param_count."""
    m = DeterministicMLP(in_features=IN, param_count=PARAM_COUNT, hidden_dims=[])
    out = m(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 3. structural: only nn.Linear — no VariationalDense ──────────────────────


def test_no_variational_dense_in_module_tree(model):
    """Deterministic MLP must contain no VariationalDense layers (zero-KL contract)."""
    vd_layers = [m for m in model.modules() if isinstance(m, VariationalDense)]
    assert vd_layers == []


def test_no_dropout_in_module_tree(model):
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert dropout_layers == []


# ── 4. zero-KL contract ───────────────────────────────────────────────────────


def test_collect_kl_is_zero_before_forward(model):
    """A deterministic model contributes nothing to collect_kl (no VariationalDense)."""
    kl = collect_kl(model)
    assert kl.item() == 0.0


def test_collect_kl_is_zero_after_forward(model, x):
    model(x)
    kl = collect_kl(model)
    assert kl.item() == 0.0


# ── 5. determinism ────────────────────────────────────────────────────────────


def test_output_is_deterministic(model, x):
    """Pure nn.Linear: same input must produce exactly identical output on every call."""
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert (out1 - out2).abs().max().item() == 0.0


def test_output_is_deterministic_in_train_mode(model, x):
    """Determinism must hold in train mode too — no stochastic layers."""
    out1 = model(x)
    out2 = model(x)
    assert (out1 - out2).abs().max().item() == 0.0


# ── 6. round-trip serialization ───────────────────────────────────────────────


def test_round_trip_state_dict_exact(model):
    """from_config + load_state_dict must reproduce weights with max|Δw| == 0."""
    config = model.get_config()
    restored = DeterministicMLP.from_config(config)
    restored.load_state_dict(model.state_dict())
    for (k, orig), (_, rest) in zip(
        model.state_dict().items(), restored.state_dict().items()
    ):
        assert (orig - rest).abs().max().item() == 0.0, f"weight '{k}' differs"


def test_round_trip_config_values(model):
    config = model.get_config()
    assert config["in_features"] == IN
    assert config["param_count"] == PARAM_COUNT
    assert config["hidden_dims"] == [8, 8]
    assert config["activation"] == "relu"
