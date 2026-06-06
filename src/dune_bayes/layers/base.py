"""Variational layer base — shared KL machinery (ADR-0004, GitHub #64).

Everything a variational layer must share lives here:
  - gaussian_kl: the one closed-form Gaussian–Gaussian KL (numerical rule 4),
    accepting a float or scalar-tensor prior scale so fixed priors and
    PriorScale handles go through the same formula.
  - _RHO_INIT: the common posterior-scale initialisation.
  - VariationalLayer: base class owning the kl_beta buffer, the .kl stash,
    and the β·KL/kl_divisor scaling. Subclassing is what registers a layer
    with collect_kl()/set_kl_beta(), so the KL-is-never-dropped invariant
    (numerical rule 5) is enforced by inheritance rather than by a
    hand-maintained isinstance list.
"""

import math

import torch
import torch.nn as nn

# softplus(-3) ≈ 0.049 — tight-ish initial posterior scale.
_RHO_INIT = -3.0


def gaussian_kl(
    loc: torch.Tensor,
    scale: torch.Tensor,
    prior_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL[ N(loc, scale²) ‖ N(0, prior_scale²) ], summed over all dims.

    KL = log(prior_scale/scale) + (scale² + loc²)/(2·prior_scale²) − ½

    Args:
        loc: Posterior means, any shape.
        scale: Posterior stds, same shape as loc, strictly positive.
        prior_scale: Prior std — a plain float for fixed priors, or a positive
            scalar tensor (e.g. a PriorScale sample) so the hyperprior keeps
            its gradient path.

    Returns:
        Scalar tensor: the KL summed over all elements of loc/scale.
    """
    log_prior = (
        torch.log(prior_scale)
        if torch.is_tensor(prior_scale)
        else math.log(prior_scale)
    )
    return torch.sum(
        log_prior
        - torch.log(scale)
        + (scale.pow(2) + loc.pow(2)) / (2.0 * prior_scale**2)
        - 0.5
    )


class VariationalLayer(nn.Module):
    """Base class for layers carrying a variational posterior.

    Owns the KL plumbing every variational layer needs:
      - ``kl_divisor`` — the KL/N denominator (ADR-0001),
      - ``kl_beta`` — non-trainable warm-up factor β ∈ [0, 1]; a buffer so it
        moves with ``.to(device)`` and is saved in state_dict; driven by
        :func:`set_kl_beta`,
      - ``.kl`` — live autograd stash, a scalar zero before the first forward
        pass so :func:`collect_kl` is always well-defined,
      - :meth:`_stash_kl` — the single place where β·KL/kl_divisor lives.

    Subclasses call ``_stash_kl(kl)`` at the end of each variational forward.
    Point-mode forwards reset ``self.kl = torch.zeros(())`` directly —
    β-scaling a structural zero is theater, not annealing.

    Args:
        kl_divisor: KL term denominator — set to N (training-set size) for
            the KL/N weighting prescribed by ADR-0001.
    """

    # register_buffer types as Tensor | Module; pin it down for mypy.
    kl_beta: torch.Tensor

    def __init__(self, kl_divisor: float = 1.0) -> None:
        super().__init__()
        self.kl_divisor = float(kl_divisor)
        self.register_buffer("kl_beta", torch.tensor(1.0))
        self.kl: torch.Tensor = torch.zeros(())

    def _stash_kl(self, kl: torch.Tensor) -> None:
        """Stash β·KL/kl_divisor for collect_kl() — the one place this scaling lives."""
        self.kl = self.kl_beta * kl / self.kl_divisor


# ── module-walk utilities ─────────────────────────────────────────────────────


def _iter_variational_layers(model: nn.Module):
    """Yield every VariationalLayer in the module tree."""
    for module in model.modules():
        if isinstance(module, VariationalLayer):
            yield module


def collect_kl(model: nn.Module) -> torch.Tensor:
    """Sum the most-recent KL of every VariationalLayer in the module tree.

    Call after a forward pass. The explicit module-walk replaces Keras
    add_loss auto-propagation (ADR-0004 / ADR-0006): model.modules() recurses
    through all sub-modules, so KL crosses nested-module boundaries trivially.

    Returns:
        Scalar tensor (with autograd) equal to sum of all stashed .kl values.
    """
    total = torch.zeros(())
    for layer in _iter_variational_layers(model):
        total = total + layer.kl
    return total


def set_kl_beta(model: nn.Module, beta: float) -> None:
    """Set the warm-up annealing factor on every VariationalLayer in a model.

    Args:
        model: Any nn.Module containing VariationalLayer sub-modules.
        beta: New value for kl_beta, typically in [0, 1].
    """
    for layer in _iter_variational_layers(model):
        layer.kl_beta.fill_(float(beta))
