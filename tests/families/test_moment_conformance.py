"""Family moment-conformance gate (issue 0091 / GitHub #91).

The variance decomposition (disentanglement) is computed generically from each
posterior draw's ``dist.mean`` / ``dist.variance``, so the family contract
requires every registered family to produce distributions where both moments
are **defined, or documented-infinite** (CONTEXT.md glossary). Concretely:

  - accessing ``mean`` / ``variance`` must not raise (defined),
  - neither moment may ever be NaN (undefined),
  - ``variance`` may be ``+inf`` only as an honest documented-infinity
    (e.g. StudentT with df ≤ 2) — never negative, never -inf.

Same auto-discovery as the link-floor gate: the recursive subclass walk
(``tests.support.concrete_families``) includes future families the moment they
are defined; the walk's coverage is asserted in ``test_link_floor_gate.py``.
"""

import itertools

import pytest
import torch

from dune_bayes.families import BaseFamily
from tests.support import concrete_families

# Same grid as the link-floor gate: ±1e4 extremes, the −104 float32 softplus
# underflow region, and the ordinary operating range — moments must conform
# everywhere log_prob does.
PRE_LINK_VALUES = [-1e4, -104.0, -3.0, 0.0, 3.0, 1e4]


@pytest.mark.parametrize("family_cls", concrete_families(), ids=lambda c: c.__name__)
def test_mean_and_variance_are_defined_everywhere(
    family_cls: type[BaseFamily],
) -> None:
    """Gate: defined, non-NaN moments across the extreme pre-link grid.

    validate_args=True per numerical rule 6 (test side).
    """
    family = family_cls(validate_args=True)
    grid = torch.tensor(
        list(itertools.product(PRE_LINK_VALUES, repeat=family.param_count)),
        dtype=torch.float32,
    )
    dist = family(grid)

    mean, variance = dist.mean, dist.variance  # must not raise

    # Every current family has a finite mean over its whole pre-link range
    # (StudentT's default df_min=1.0 keeps df > 1, out of the Cauchy regime).
    # A future family documenting an infinite-mean region extends this gate.
    assert torch.isfinite(mean).all(), (
        f"{family_cls.__name__}: non-finite mean at pre-link rows "
        f"{grid[~torch.isfinite(mean)].tolist()}"
    )
    # Variance: finite or honestly +inf (documented-infinite); NaN or negative
    # values would silently poison the variance decomposition.
    conforming = torch.isfinite(variance) | torch.isposinf(variance)
    assert conforming.all(), (
        f"{family_cls.__name__}: NaN/-inf variance at pre-link rows "
        f"{grid[~conforming].tolist()}"
    )
    assert (variance[torch.isfinite(variance)] >= 0).all()
