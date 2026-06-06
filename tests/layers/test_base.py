"""Boundary tests for the variational layer base (GitHub #64).

Archetypes covered:
  - Closed-form: gaussian_kl against a hand-computed Gaussian–Gaussian value,
    for both the float-prior and the scalar-tensor-prior path.
  - Inheritance contract: a brand-new VariationalLayer subclass is reached by
    collect_kl / set_kl_beta with no walker edit — the growth-resistance
    regression for numerical rule 5 (KL is never silently dropped).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dune_bayes.layers import (
    BayesianEmbedding,
    BayesianIntercept,
    VariationalDense,
    VariationalLayer,
    collect_kl,
    gaussian_kl,
    set_kl_beta,
)

# ── 1. gaussian_kl closed-form reference ─────────────────────────────────────


def test_gaussian_kl_matches_hand_computed_value():
    """KL[N(loc, scale²) ‖ N(0, p²)] = Σ log(p/σ) + (σ²+μ²)/(2p²) − ½.

    Hand-computed for loc=(0.5, −1.0), scale=(0.3, 0.6), p=2.0:
      element 1: ln(2/0.3) + (0.09+0.25)/8 − 0.5 = 1.43961998…
      element 2: ln(2/0.6) + (0.36+1.00)/8 − 0.5 = 0.87397280…
      sum = 2.31359279…
    """
    loc = torch.tensor([0.5, -1.0])
    scale = torch.tensor([0.3, 0.6])
    got = gaussian_kl(loc, scale, 2.0)
    # rel=1e-6: pure float32 arithmetic error — no MC noise in a closed form.
    assert float(got) == pytest.approx(2.3135927892118174, rel=1e-6)


def test_gaussian_kl_float_and_tensor_prior_agree():
    """The float and scalar-tensor prior paths are the same formula."""
    loc = torch.tensor([0.5, -1.0])
    scale = torch.tensor([0.3, 0.6])
    kl_float = float(gaussian_kl(loc, scale, 2.0))
    kl_tensor = float(gaussian_kl(loc, scale, torch.tensor(2.0)))
    # rel=1e-6: both paths are float32 closed forms; only rounding may differ.
    assert kl_float == pytest.approx(kl_tensor, rel=1e-6)


def test_gaussian_kl_tensor_prior_carries_gradient():
    """A PriorScale-style tensor prior must keep its gradient path alive."""
    prior = torch.tensor(2.0, requires_grad=True)
    kl = gaussian_kl(torch.tensor([0.5]), torch.tensor([0.3]), prior)
    kl.backward()
    assert prior.grad is not None
    assert float(prior.grad) != 0.0


# ── 2. inheritance contract: subclassing registers with the KL machinery ─────


class _ToyVariational(VariationalLayer):
    """Minimal fourth variational layer — proves the walker needs no edit."""

    def __init__(self) -> None:
        super().__init__(kl_divisor=2.0)
        self.loc = nn.Parameter(torch.zeros(3))
        self.rho = nn.Parameter(torch.full((3,), -3.0))

    def forward(self) -> torch.Tensor:
        scale = F.softplus(self.rho)
        self._stash_kl(gaussian_kl(self.loc, scale, 1.0))
        return self.loc + scale * torch.randn_like(self.loc)


def test_shipped_layers_subclass_variational_layer():
    """All three shipped variational layers register via inheritance."""
    assert issubclass(VariationalDense, VariationalLayer)
    assert issubclass(BayesianIntercept, VariationalLayer)
    assert issubclass(BayesianEmbedding, VariationalLayer)


def test_new_subclass_is_reached_by_collect_kl():
    """collect_kl sees a new VariationalLayer subclass with no walker edit."""
    model = nn.ModuleDict({"toy": _ToyVariational()})
    model["toy"]()
    assert float(collect_kl(model).detach()) > 0.0, (
        "collect_kl dropped a VariationalLayer subclass — numerical rule 5"
    )


def test_new_subclass_is_reached_by_set_kl_beta():
    """set_kl_beta(0) gates a new subclass's KL to exactly zero (0 · KL/N)."""
    model = nn.ModuleDict({"toy": _ToyVariational()})
    set_kl_beta(model, 0.0)
    model["toy"]()
    assert float(collect_kl(model).detach()) == 0.0


def test_kl_beta_buffer_is_in_state_dict():
    """kl_beta is a buffer: serialized under its own key, not a parameter."""
    toy = _ToyVariational()
    assert "kl_beta" in toy.state_dict()
    assert "kl_beta" not in dict(toy.named_parameters())
