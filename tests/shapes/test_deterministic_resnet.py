"""Tests for DeterministicResNet shape function (issue 0020 / GitHub #39).

Four reference-test archetypes (CLAUDE.md):
  - Shape:      forward output (batch, param_count).
  - Structural: residual blocks with skip connections; no VariationalDense.
  - Zero-KL:    collect_kl returns exactly 0.0 (no variational parameters).
  - Determinism: same input → identical output on every call.
  - Round-trip: get_config + from_config + state_dict with max|Δw| == 0.
"""

import pytest
import torch
import torch.nn as nn

from dune_bayes.layers import collect_kl
from dune_bayes.layers.variational_dense import VariationalDense
from dune_bayes.shapes import ShapeFunctionRegistry
from dune_bayes.shapes.deterministic_resnet import DeterministicResNet

IN, PARAM_COUNT, BATCH = 3, 2, 8


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DeterministicResNet(
        in_features=IN, param_count=PARAM_COUNT, hidden_dim=16, num_blocks=2
    )


@pytest.fixture
def x():
    return torch.randn(BATCH, IN, generator=torch.Generator().manual_seed(42))


# ── 1. registry ───────────────────────────────────────────────────────────────


def test_registry_resolves_resnet_by_name():
    cls = ShapeFunctionRegistry.get("ResNet")
    assert cls is DeterministicResNet


def test_registry_resolved_class_is_instantiable(x):
    cls = ShapeFunctionRegistry.get("ResNet")
    m = cls(in_features=IN, param_count=PARAM_COUNT, hidden_dim=16, num_blocks=1)
    out = m(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 2. output shape ───────────────────────────────────────────────────────────


def test_forward_output_shape(model, x):
    out = model(x)
    assert out.shape == (BATCH, PARAM_COUNT)


def test_forward_output_shape_single_block(x):
    m = DeterministicResNet(
        in_features=IN, param_count=PARAM_COUNT, hidden_dim=8, num_blocks=1
    )
    out = m(x)
    assert out.shape == (BATCH, PARAM_COUNT)


def test_forward_output_shape_deep(x):
    m = DeterministicResNet(
        in_features=IN, param_count=PARAM_COUNT, hidden_dim=32, num_blocks=4
    )
    out = m(x)
    assert out.shape == (BATCH, PARAM_COUNT)


# ── 3. structural: residual blocks, no VariationalDense ──────────────────────


def test_no_variational_dense_in_module_tree(model):
    """All deterministic — no VariationalDense anywhere in the tree."""
    vd_layers = [m for m in model.modules() if isinstance(m, VariationalDense)]
    assert vd_layers == []


def test_no_dropout_in_module_tree(model):
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert dropout_layers == []


def test_residual_blocks_exist(model):
    """ResNet must contain at least one residual block (structural contract)."""
    from dune_bayes.shapes.deterministic_resnet import ResBlock

    blocks = [m for m in model.modules() if isinstance(m, ResBlock)]
    assert len(blocks) == 2  # num_blocks=2 in fixture


# ── 4. zero-KL contract ───────────────────────────────────────────────────────


def test_collect_kl_is_zero_before_forward(model):
    kl = collect_kl(model)
    assert kl.item() == 0.0


def test_collect_kl_is_zero_after_forward(model, x):
    model(x)
    kl = collect_kl(model)
    assert kl.item() == 0.0


# ── 5. determinism ────────────────────────────────────────────────────────────


def test_output_is_deterministic(model, x):
    """Pure nn.Linear: same input must give exactly identical output every call."""
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert (out1 - out2).abs().max().item() == 0.0


def test_output_is_deterministic_in_train_mode(model, x):
    out1 = model(x)
    out2 = model(x)
    assert (out1 - out2).abs().max().item() == 0.0


# ── 6. round-trip serialization ───────────────────────────────────────────────


def test_round_trip_state_dict_exact(model):
    """from_config + load_state_dict must reproduce weights with max|Δw| == 0."""
    config = model.get_config()
    restored = DeterministicResNet.from_config(config)
    restored.load_state_dict(model.state_dict())
    for (k, orig), (_, rest) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert (orig - rest).abs().max().item() == 0.0, f"weight '{k}' differs"


def test_round_trip_config_values(model):
    config = model.get_config()
    assert config["in_features"] == IN
    assert config["param_count"] == PARAM_COUNT
    assert config["hidden_dim"] == 16
    assert config["num_blocks"] == 2
