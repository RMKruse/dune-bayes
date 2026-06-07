"""Variance decomposition (disentanglement) of the posterior predictive.

Issue 0091 / GitHub #91 — the paper's core claim in code form (CONTEXT.md
glossary: "Variance decomposition (disentanglement)"): the law-of-total-variance
split of the posterior predictive,

    aleatoric  = E_θ[Var(y|θ)]   (mean over posterior draws of the family variance)
    epistemic  = Var_θ[E(y|θ)]   (variance over posterior draws of the family mean)

computed generically from each draw's ``dist.mean`` / ``dist.variance`` for any
registered family — no family-specific branches.  The split is valid only for
coherent posterior draws (ADR-0007, issue #85): ``draw_predictive`` guarantees
each of the T components is ONE global weight realization.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from dune_bayes.model import BayesianNAMLSS


@dataclass
class VarianceDecomposition:
    """The law-of-total-variance split of the posterior predictive.

    Attributes:
        aleatoric: E_θ[Var(y|θ)], shape (n,) — irreducible response noise,
            captured by the family.
        epistemic: Var_θ[E(y|θ)], shape (n,) — uncertainty about the learned
            effects, captured by the posterior over weights.
        total: aleatoric + epistemic, shape (n,) — the predictive variance
            (equals ``MixtureSameFamily.variance`` of the same draws).
    """

    aleatoric: torch.Tensor
    epistemic: torch.Tensor
    total: torch.Tensor


def variance_decomposition(
    model: BayesianNAMLSS,
    summed_samples: torch.Tensor,
) -> VarianceDecomposition:
    """Decompose predictive variance into aleatoric + epistemic components.

    Consumes the ``summed_samples`` produced by ``draw_predictive`` (the
    draw-once-score-once pattern of the split workhorse, GitHub #68), so a
    single set of posterior draws serves WAIC/LOO and the decomposition alike.

    Args:
        model: BayesianNAMLSS instance — only its family is used here.
        summed_samples: Summed-predictor draws, shape (T, n, param_count),
            as returned by ``draw_predictive``.

    Returns:
        VarianceDecomposition with aleatoric / epistemic / total, each (n,).
    """
    with torch.no_grad():
        dist = model.family(summed_samples)  # batch_shape (T, n)
        # Population variance over draws (correction=0): the T draws form a
        # uniform mixture, and the law of total variance for a mixture uses
        # the ÷T moment — this makes aleatoric + epistemic IDENTICALLY equal
        # to MixtureSameFamily.variance, not merely asymptotically.
        per_draw_var = dist.variance  # (T, n)
        # Honest-inf path (#91): a draw whose family variance is truly
        # infinite (e.g. StudentT with df ≤ 2) makes the aleatoric mean inf
        # for that observation. Surface it with a cause-naming warning —
        # NEVER clamp; clamping would silently fake a finite uncertainty.
        nonfinite = ~torch.isfinite(per_draw_var)
        if bool(nonfinite.any()):
            n_draws_total, n_obs = per_draw_var.shape
            n_draws_bad = int(nonfinite.any(dim=1).sum())
            n_obs_bad = int(nonfinite.any(dim=0).sum())
            warnings.warn(
                f"Aleatoric variance is non-finite for {n_obs_bad} of {n_obs} "
                f"observations: {n_draws_bad} of {n_draws_total} posterior "
                f"draws have infinite {type(model.family).__name__} variance "
                f"(StudentT: df ≤ 2 has infinite variance; "
                f"df_min=2.0 pins it finite). Reported honestly as inf, "
                f"never clamped.",
                RuntimeWarning,
                stacklevel=2,
            )
        aleatoric = per_draw_var.mean(dim=0)  # (n,)
        epistemic = dist.mean.var(dim=0, correction=0)  # (n,)
        return VarianceDecomposition(
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=aleatoric + epistemic,
        )
