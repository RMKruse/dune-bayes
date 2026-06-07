"""Gradcheck of every registered family's log_prob w.r.t. pre-link params (#89).

PRD 0002 (#84) hardening slice: ``torch.autograd.gradcheck`` (float64,
``validate_args=True``) on ``family.log_prob(params, y)`` differentiated
w.r.t. the *pre-link* parameter tensor — links included, so the
``softplus(x) + EPS`` path of GitHub #88 is part of the checked graph.

Registration = subclassing ``BaseFamily``, so the parametrization walks
``__subclasses__`` recursively (same auto-discovery as the #88 link-floor
gate) and future families are included the moment they are defined.

No behavior change expected — any RED here is a real bug in the link or
log_prob math.  Runs in the core (unskippable) suite: numerical correctness
tests are never skippable (CLAUDE.md).
"""

import pytest
import torch
from torch.autograd import gradcheck

from dune_bayes.families import BaseFamily
from tests.support import concrete_families

# Moderate, strictly positive responses: inside the support of every current
# family (positive reals ⊂ all reals), and close enough to the implied
# location that no log_prob plateaus at FD-unresolvable magnitudes.  Each
# family keeps only the ones its own support admits, so future families with
# narrower supports need no hand-edit here.  The integer values exist for the
# count families (NegativeBinomial, #95): discrete in y, but log_prob is
# still smooth in the pre-link params, so gradcheck applies unchanged.
CANDIDATE_RESPONSES = [0.3, 0.7, 1.0, 1.3, 2.5, 3.0]


def _in_support_responses(family: BaseFamily, params: torch.Tensor) -> torch.Tensor:
    """The CANDIDATE_RESPONSES the family's own support admits (never empty)."""
    dist = family(params)
    kept = [
        y
        for y in torch.tensor(CANDIDATE_RESPONSES, dtype=torch.float64)
        if bool(dist.support.check(y).all())
    ]
    # If this trips for a new family, extend CANDIDATE_RESPONSES — never skip.
    assert kept, f"no candidate response lies in {type(family).__name__} support"
    return torch.stack(kept)


@pytest.mark.parametrize("family_cls", concrete_families(), ids=lambda c: c.__name__)
def test_log_prob_gradcheck_interior(family_cls: type[BaseFamily]) -> None:
    """Analytic ∂log_prob/∂pre-link params match finite differences (interior).

    float64 + gradcheck defaults (eps=1e-6, atol=1e-5): log_prob composed of
    smooth links and torch.distributions densities is FD-accurate to ~1e-10
    at interior points, so the default tolerance has ample headroom.
    """
    family = family_cls(validate_args=True)
    g = torch.Generator().manual_seed(0)
    probe = torch.randn((1, family.param_count), generator=g, dtype=torch.float64)
    y = _in_support_responses(family, probe)
    # One batch row per admitted response, fresh interior pre-link draws.
    params = torch.randn(
        (y.shape[0], family.param_count), generator=g, dtype=torch.float64
    ).requires_grad_()

    def fn(p: torch.Tensor) -> torch.Tensor:
        return family.log_prob(p, y)

    assert gradcheck(fn, (params,))


# Boundary pre-link values, applied to one parameter column at a time:
#   −1e4 — deep in the softplus-underflow regime: the link saturates at its
#          EPS floor (df at its +1 floor), so the analytic gradient through
#          sigmoid(−1e4) is exactly 0 and FD must agree;
#   −104 — the float32 softplus-underflow point of GitHub #88, here checked
#          for gradient (not just value) correctness in float64;
#   −10  — tiny-but-live scale (softplus ≈ 4.5e−5): the steepest stretch of
#          −log(scale), where a wrong link derivative would surface;
#   +10 / +1e4 — large-parameter side, links effectively identity.
# One column at a time keeps every log_prob magnitude FD-resolvable: pushing
# e.g. an identity-linked loc AND the scale to extremes simultaneously drives
# |log_prob| past the point where its float64 ulp rivals the eps=1e-6 FD
# signal — a resolution artifact, not a gradient property.  (Value-correctness
# at the full cartesian extremes is #88's gate.)
BOUNDARY_PRE_LINK_VALUES = [-1e4, -104.0, -10.0, 10.0, 1e4]
INTERIOR_PRE_LINK = 0.5


@pytest.mark.parametrize("family_cls", concrete_families(), ids=lambda c: c.__name__)
def test_log_prob_gradcheck_near_boundary(family_cls: type[BaseFamily]) -> None:
    """Gradcheck with each pre-link parameter pushed to its boundary in turn.

    Covers the acceptance criterion that near-boundary points are part of the
    gradcheck sweep: tiny scale, df just above its floor (StudentT column 2
    at −1e4 → df = 1 + EPS-ish), and extreme ±1e4 pre-links — all through the
    softplus+EPS links of #88.
    """
    family = family_cls(validate_args=True)
    for col in range(family.param_count):
        for boundary in BOUNDARY_PRE_LINK_VALUES:
            base = torch.full(
                (1, family.param_count), INTERIOR_PRE_LINK, dtype=torch.float64
            )
            base[0, col] = boundary
            y = _in_support_responses(family, base)
            params = base.repeat(y.shape[0], 1).requires_grad_()

            def fn(p: torch.Tensor) -> torch.Tensor:
                return family.log_prob(p, y)  # noqa: B023 — consumed immediately

            assert gradcheck(fn, (params,)), (
                f"{family_cls.__name__}: gradcheck failed with column {col} "
                f"at pre-link {boundary}"
            )
