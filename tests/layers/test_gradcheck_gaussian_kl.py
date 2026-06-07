"""Gradcheck of the gaussian_kl atom (GitHub #89).

PRD 0002 (#84) hardening slice: ``torch.autograd.gradcheck`` (float64) on the
deterministic closed-form Gaussian–Gaussian KL, differentiated through the
``scale = softplus(rho)`` link exactly as ``VariationalDense`` consumes it —
so the verified gradient path is the one training actually uses.

No behavior change expected — any RED here is a real bug in the KL math.
Runs in the core (unskippable) suite: numerical correctness tests are never
skippable (CLAUDE.md).
"""

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from dune_bayes.layers import gaussian_kl


def _kl_via_rho(prior_scale: float) -> object:
    """gaussian_kl as a function of (loc, rho) — the VariationalDense path."""

    def fn(loc: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return gaussian_kl(loc, F.softplus(rho), prior_scale)

    return fn


def test_gaussian_kl_gradcheck_interior() -> None:
    """Analytic grads w.r.t. loc and rho match finite differences (interior).

    float64 + gradcheck defaults (eps=1e-6, atol=1e-5): central differences
    of a smooth closed form are accurate to ~1e-10 in float64, so the default
    tolerance has 5 orders of magnitude of headroom — failures are real.
    """
    g = torch.Generator().manual_seed(0)
    loc = torch.randn((3, 4), generator=g, dtype=torch.float64).requires_grad_()
    # rho ~ N(0,1): scales in softplus's ordinary operating range (~0.3–1.3).
    rho = torch.randn((3, 4), generator=g, dtype=torch.float64).requires_grad_()
    assert gradcheck(_kl_via_rho(prior_scale=1.0), (loc, rho))


def test_gaussian_kl_gradcheck_near_boundary() -> None:
    """Gradcheck at the tiny-scale boundary and with extreme loc / prior_scale.

    rho = −30 → scale ≈ 9.4e−14: the −log(scale) term is steepest here, the
    spot where a wrong derivative through softplus would surface.  In rho-space
    the composition log(softplus(rho)) ≈ rho stays smooth, so gradcheck's
    default float64 tolerances still have ample headroom.

    One gradcheck per grid point (scalar tensors): gaussian_kl sums over all
    elements, so batching points of wildly different magnitude into one tensor
    would sink the finite-difference signal of the small entries below the
    float64 rounding noise of the large ones — a resolution artifact, not a
    gradient property.
    """
    rho_vals = (-30.0, -10.0, 0.0, 10.0)
    loc_vals = (-100.0, 0.0, 1e-8, 100.0)
    priors = (1e-3, 1.0, 1e3)  # tight, unit, and diffuse priors
    checked = 0
    for prior_scale in priors:
        for rho_val in rho_vals:
            for loc_val in loc_vals:
                # FD-resolvability guard: when the (scale²+loc²)/(2·ps²) KL
                # offset exceeds ~1e6, its float64 ulp (≈1e-10·offset) rivals
                # the eps=1e-6 finite-difference signal of the small gradients
                # — central differences cannot resolve an O(1) slope on a 5e9
                # plateau.  That is an FD limitation, not a gradient property;
                # extreme loc and large scale are still checked at ps ≥ 1.
                scale_val = float(F.softplus(torch.tensor(rho_val)))
                offset = (scale_val**2 + loc_val**2) / (2.0 * prior_scale**2)
                if offset > 1e6:
                    continue
                loc = torch.tensor([loc_val], dtype=torch.float64).requires_grad_()
                rho = torch.tensor([rho_val], dtype=torch.float64).requires_grad_()
                assert gradcheck(_kl_via_rho(prior_scale=prior_scale), (loc, rho))
                checked += 1
    # Guard against the guard: the sweep must not silently degenerate.
    # 48-cell grid minus the 10 FD-unresolvable tight-prior cells.
    assert checked == 38


def test_gaussian_kl_gradcheck_tensor_prior_scale() -> None:
    """The tensor-prior_scale branch carries a correct gradient.

    The hierarchical tier feeds a PriorScale *sample* into gaussian_kl as a
    scalar tensor precisely so the hyperprior keeps its gradient path — so
    that path's derivative (through torch.log and the /2ps² term) must
    gradcheck too, not just loc/rho.
    """
    g = torch.Generator().manual_seed(1)
    loc = torch.randn((3, 4), generator=g, dtype=torch.float64).requires_grad_()
    rho = torch.randn((3, 4), generator=g, dtype=torch.float64).requires_grad_()

    def fn(
        loc: torch.Tensor, rho: torch.Tensor, prior_scale: torch.Tensor
    ) -> torch.Tensor:
        return gaussian_kl(loc, F.softplus(rho), prior_scale)

    for ps0 in (0.05, 1.0, 20.0):  # tight, unit, diffuse — all FD-resolvable
        prior = torch.tensor(ps0, dtype=torch.float64).requires_grad_()
        assert gradcheck(fn, (loc, rho, prior))
