"""Extreme pre-link finite-log_prob gate for all families (GitHub #88).

Bare ``softplus`` underflows to exactly 0.0 near pre-link −104 in float32,
producing a zero scale and a NaN ``log_prob`` (verified: numerical rule 1).
Every positivity link in the families must therefore be ``softplus(x) + EPS``.
This gate sweeps extreme pre-link values — ±1e4 and the −104 underflow
region — through every registered family and asserts finite ``log_prob``.

Registration = subclassing ``BaseFamily`` (its ``__init_subclass__`` enforces
the contract), so the parametrization walks ``__subclasses__`` recursively
(``tests.support.concrete_families``) and future families are auto-included
the moment they are defined.  This file owns the walk's coverage test.
"""

import itertools

import pytest
import torch

from dune_bayes.families import BaseFamily, GammaFamily, NormalFamily, StudentTFamily
from tests.support import concrete_families

# ±1e4 per the acceptance criteria; −104.0 is where float32 softplus
# underflows to exactly 0.0 (the failure mode this gate guards); −3.0/0.0/3.0
# cover the ordinary operating range.
PRE_LINK_VALUES = [-1e4, -104.0, -3.0, 0.0, 3.0, 1e4]

# Candidate responses spanning typical values and both tails; each family
# keeps only the ones inside its support (via the distribution's own
# ``support.check``), so future families need no hand-edit here.
CANDIDATE_RESPONSES = [-50.0, -1.0, 1e-4, 0.01, 1.0, 50.0, 1e6]


def test_subclass_walk_finds_all_registered_families() -> None:
    """The auto-discovery must cover every family the package exports."""
    found = set(concrete_families())
    assert {NormalFamily, GammaFamily, StudentTFamily} <= found


@pytest.mark.parametrize("family_cls", concrete_families(), ids=lambda c: c.__name__)
def test_extreme_pre_link_log_prob_is_finite(family_cls: type[BaseFamily]) -> None:
    """Gate: finite log_prob across the full cartesian extreme pre-link grid.

    validate_args=True per numerical rule 6 (test side) — it additionally
    catches a zero/negative scale at distribution construction time.
    """
    family = family_cls(validate_args=True)
    # Cartesian product over all param columns: every parameter is pushed to
    # every extreme simultaneously and in mixed combinations.
    grid = torch.tensor(
        list(itertools.product(PRE_LINK_VALUES, repeat=family.param_count)),
        dtype=torch.float32,
    )
    dist = family(grid)

    in_support = [
        y
        for y in torch.tensor(CANDIDATE_RESPONSES)
        if bool(dist.support.check(y).all())
    ]
    # If this trips for a new family, extend CANDIDATE_RESPONSES — never skip.
    assert in_support, f"no candidate response lies in {family_cls.__name__} support"

    for y in in_support:
        log_prob = family.log_prob(grid, y.expand(grid.shape[0]))
        assert torch.isfinite(log_prob).all(), (
            f"{family_cls.__name__}: non-finite log_prob at response y={y.item()} "
            f"for pre-link rows "
            f"{grid[~torch.isfinite(log_prob)].tolist()}"
        )
