"""Gradcheck of the hierarchical-IG closed-form scale-KL (GitHub #89).

PRD 0002 (#84) hardening slice: ``torch.autograd.gradcheck`` (float64) on
``PriorScale.hyperprior_kl()`` for the inverse-gamma hyperprior — the one
closed-form KL in the prior tiers (ADR-0002; its *value* was verified against
an MC reference in #86, this verifies its *derivative* w.r.t. the variational
params loc_s / rho_s).

The half-Cauchy tier is deliberately absent: its KL is a single-sample MC
estimate, and finite differences across stochastic calls measure noise, not
gradients (the same re-scoping that moved this issue off the full-ELBO
gradcheck).

No behavior change expected — any RED here is a real bug in the KL math.
Runs in the core (unskippable) suite: numerical correctness tests are never
skippable (CLAUDE.md).
"""

import torch
import torch.nn as nn
from torch.autograd import gradcheck
from torch.func import functional_call

from dune_bayes.priors import PriorScale


class _IGScaleKL(nn.Module):
    """Expose hyperprior_kl() as forward() so functional_call can swap params."""

    def __init__(self, ps: PriorScale) -> None:
        super().__init__()
        self.ps = ps

    def forward(self) -> torch.Tensor:
        return self.ps.hyperprior_kl()


def _ig_kl_fn(alpha0: float, beta0: float) -> object:
    """The IG scale-KL as a pure function of float64 (loc_s, rho_s) tensors."""
    ps = PriorScale(
        mode="hierarchical",
        hyperprior="inverse_gamma",
        alpha0=alpha0,
        beta0=beta0,
        validate_args=True,
    ).double()
    wrapper = _IGScaleKL(ps)

    def fn(loc_s: torch.Tensor, rho_s: torch.Tensor) -> torch.Tensor:
        return functional_call(wrapper, {"ps.loc_s": loc_s, "ps.rho_s": rho_s})

    return fn


def test_ig_scale_kl_gradcheck_interior() -> None:
    """Analytic grads w.r.t. loc_s and rho_s match finite differences.

    float64 + gradcheck defaults (eps=1e-6, atol=1e-5): the KL is a smooth
    closed form (exp, log, lgamma of constants), so central differences are
    accurate to ~1e-10 — default tolerances have ample headroom.
    """
    fn = _ig_kl_fn(alpha0=1.0, beta0=1.0)
    loc_s = torch.tensor(0.3, dtype=torch.float64).requires_grad_()
    rho_s = torch.tensor(-1.0, dtype=torch.float64).requires_grad_()
    assert gradcheck(fn, (loc_s, rho_s))


def test_ig_scale_kl_gradcheck_near_boundary() -> None:
    """Gradcheck at the tiny-σ_s boundary and off-center loc_s, varied (α₀, β₀).

    rho_s = −30 → σ_s ≈ 9.4e−14: the −½·log(2π·σ_s²) entropy term is steepest
    here — the spot a wrong softplus derivative would surface.  In rho-space
    log(softplus(rho_s)) ≈ rho_s stays smooth, so default tolerances hold.

    |loc_s| is capped at 3 (scale s ∈ [e⁻³, e³]): beyond that the
    β₀·exp(−2·loc_s) term exceeds ~e⁶ and its float64 ulp rivals the eps=1e-6
    FD signal of the O(1) gradients — an FD-resolution limit, not a gradient
    property.  e±3 already brackets any scale a sane prior would visit.
    """
    for alpha0, beta0 in ((0.5, 0.5), (1.0, 1.0), (3.0, 2.0)):
        fn = _ig_kl_fn(alpha0=alpha0, beta0=beta0)
        for loc_val in (-3.0, 0.0, 3.0):
            for rho_val in (-30.0, -10.0, 0.0, 5.0):
                loc_s = torch.tensor(loc_val, dtype=torch.float64).requires_grad_()
                rho_s = torch.tensor(rho_val, dtype=torch.float64).requires_grad_()
                assert gradcheck(fn, (loc_s, rho_s))
