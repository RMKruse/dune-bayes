"""NaN-gradient gate on the full stochastic ELBO (GitHub #89).

PRD 0002 (#84) hardening slice, integration half: the deterministic atoms are
gradchecked elsewhere (tests/layers, tests/priors, tests/families); the full
ELBO is stochastic, so finite differences across calls would measure
reparameterization noise, not gradients.  What CAN be asserted about the full
graph is finiteness: one seeded backward pass on a deliberately
ill-conditioned batch must leave every parameter with a finite gradient.

The batch stacks the classic pathologies:
  - an exactly-zero feature column — all-zero ReLU rows drive the
    local-reparameterization variance to 0, the sqrt(0)-NaN regression of
    GitHub #85 that the EPS variance floor guards;
  - a near-constant feature column (variance ~1e-14);
  - an extreme-magnitude feature column (±1e4);
  - responses in the far tails / extreme outliers (±1e6, and 1e-6 next to
    1e6 for positive-support families), support-filtered per family.

The model exercises every variational atom in one graph: BayesianMLP with a
hierarchical inverse-gamma PriorScale, NeuralLinearMLP with hierarchical
half-Cauchy, a fixed-prior BayesianMLP, and the variational intercept — so a
non-finite gradient anywhere in the KL or NLL paths trips the gate.

Runs in the core (unskippable) suite: numerical correctness tests are never
skippable (CLAUDE.md).
"""

import math

import pytest
import torch

from dune_bayes.families import BaseFamily
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.shapes import BayesianMLP, NeuralLinearMLP
from dune_bayes.utils import seed_everything
from tests.support import concrete_families

N = 64  # batch size; small but enough to mix all pathologies in one pass


def _ill_conditioned_features(n: int) -> dict[str, torch.Tensor]:
    """Deterministic pathological design matrix — no RNG, fully reproducible."""
    x_const = torch.ones((n, 1))
    x_const[n // 2] += 1e-7  # near-constant: variance ~1e-14, not exactly 0
    return {
        "x_zero": torch.zeros((n, 1)),
        "x_const": x_const,
        "x_extreme": torch.linspace(-1e4, 1e4, n).unsqueeze(-1),
    }


# Far-tail / outlier responses; each family keeps the ones its support admits
# (Gamma drops the negatives but keeps the 1e-6-next-to-1e6 spread), so
# future families need no hand-edit here.
TAIL_CANDIDATES = [-1e6, -50.0, -1.0, 1e-6, 1.0, 50.0, 1e6]


def _far_tail_responses(family: BaseFamily, n: int) -> torch.Tensor:
    """Deterministic length-n response vector tiled from in-support tails."""
    probe = torch.zeros((1, family.param_count))
    dist = family(probe)
    kept = [
        y for y in torch.tensor(TAIL_CANDIDATES) if bool(dist.support.check(y).all())
    ]
    # If this trips for a new family, extend TAIL_CANDIDATES — never skip.
    assert kept, f"no tail candidate lies in {type(family).__name__} support"
    return torch.stack(kept).repeat(math.ceil(n / len(kept)))[:n]


@pytest.mark.parametrize("family_cls", concrete_families(), ids=lambda c: c.__name__)
def test_ill_conditioned_batch_gradients_finite(
    family_cls: type[BaseFamily],
) -> None:
    """Gate: one seeded ELBO backward pass leaves every param grad finite.

    Also asserts every parameter *received* a gradient — a None grad would
    mean a KL or NLL path silently detached (numerical rule 5 in gradient
    form), which finiteness alone would not catch.
    """
    seed_everything(0)  # weight init + reparameterization draws
    family = family_cls()  # validate_args=False — this gates the hot path
    formula = {
        "x_zero": BayesianMLP(
            in_features=1,
            param_count=family.param_count,
            hidden_dims=[8],
            prior={"mode": "hierarchical", "hyperprior": "inverse_gamma"},
            kl_divisor=float(N),
        ),
        "x_const": NeuralLinearMLP(
            in_features=1,
            param_count=family.param_count,
            hidden_dims=[8],
            prior={"mode": "hierarchical", "hyperprior": "half_cauchy"},
            kl_divisor=float(N),
        ),
        "x_extreme": BayesianMLP(  # fixed-prior tier
            in_features=1,
            param_count=family.param_count,
            hidden_dims=[8],
            kl_divisor=float(N),
        ),
    }
    model = BayesianNAMLSS(formula, family=family, n_obs=N)
    model.train()

    loss = model.loss(_ill_conditioned_features(N), _far_tail_responses(family, N))
    assert torch.isfinite(loss), f"non-finite ELBO loss: {loss.item()}"
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"
