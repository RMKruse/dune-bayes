"""BayesianEmbedding — variational categorical first mapping (issue 0012 / GitHub #13).



Each category level gets a mean-field Gaussian weight posterior under a shared
per-feature PriorScale — the neural analog of a BAMLSS random effect.  All
levels share one PriorScale, so rare levels shrink toward the prior (partial
pooling) while common levels stay data-driven.

A point-embedding fallback (mode="point") is available but not the default;
it removes the per-level credible interval and the shrinkage behavior.

Estimator split (ADR-0007, issue #85), mirroring VariationalDense:
  - **Training** uses local reparameterization: each gathered row draws fresh
    per-element noise from its marginal posterior. Exact per-element
    marginals, same expectation, lower gradient variance.
  - **Eval (posterior sampling)** draws coherent table realizations: one full
    W ~ q(W) per draw, gathered at every index, so repeated levels share an
    embedding within a draw. An index tensor expanded along a leading sample
    dimension yields one independent coherent table draw per slice — the
    contract the vectorized T-sweeps (issue 0027 / GitHub #80) rely on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from dune_bayes.layers.base import _RHO_INIT, VariationalLayer, gaussian_kl

if TYPE_CHECKING:
    from dune_bayes.priors.prior_scale import PriorScale

# PriorScale imports stay local (TYPE_CHECKING + function bodies): PriorScale
# subclasses VariationalLayer, so priors → layers/__init__ → this module is a
# real cycle if PriorScale is needed at module-exec time (issue #73).

# Small initialisation scale so the embedding table starts near zero.
_LOC_INIT_STD = 0.01


class BayesianEmbedding(VariationalLayer):
    """Variational embedding table — Bayesian categorical first mapping.

    Embedding weights W[level, dim] carry a mean-field Gaussian posterior;
    the shared PriorScale s acts as the random-effect variance component
    (analogous to the IG variance component in BAMLSS).

    Args:
        num_embeddings: Vocabulary size (number of distinct categories).
        embedding_dim: Output width per category.
        prior_scale_handle: Per-feature PriorScale module.  Defaults to a
            fixed PriorScale(scale=1.0) when not supplied.
        kl_divisor: KL denominator — set to N for KL/N weighting (ADR-0001).
        mode: ``"variational"`` (default) — mean-field Gaussian posterior, KL
            accumulated in ``.kl``.  ``"point"`` — deterministic weights, zero KL.
        validate_args: Passed to torch.distributions.  False in the training
            hot path; True in test fixtures (numerical rule 6).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        prior_scale_handle: PriorScale | None = None,
        kl_divisor: float = 1.0,
        mode: str = "variational",
        validate_args: bool = False,
    ) -> None:
        super().__init__(kl_divisor=kl_divisor)
        if mode not in ("variational", "point"):
            raise ValueError(f"unknown mode {mode!r}; choose 'variational' or 'point'")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        if prior_scale_handle is None:
            from dune_bayes.priors.prior_scale import PriorScale

            prior_scale_handle = PriorScale(mode="fixed", scale=1.0)
        self.prior_scale_handle = prior_scale_handle
        self.mode = mode
        self.validate_args = bool(validate_args)

        self.loc = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim).normal_(
                std=_LOC_INIT_STD
            )
        )
        if self.mode == "variational":
            self.rho = nn.Parameter(
                torch.full((self.num_embeddings, self.embedding_dim), _RHO_INIT)
            )
        else:
            self.register_parameter("rho", None)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return variational embedding for each index in idx.

        Args:
            idx: LongTensor of shape (batch,) or (...,) of category indices.

        Returns:
            Tensor of shape (*idx.shape, embedding_dim) — a reparameterized
            sample from the posterior over the requested embedding rows.
        """
        if self.mode == "variational":
            scale = F.softplus(self.rho)  # (num_embeddings, embedding_dim)
            if self.training:
                # Local reparameterization — TRAINING only (ADR-0007): sample
                # each gathered row from its marginal N(loc[idx], scale[idx]²)
                # with fresh per-element noise.  Per-element marginals are
                # exact, so the ELBO is unchanged; within-draw correlation
                # across repeated levels is dropped (lower gradient variance,
                # same expectation) — irrelevant to the per-row loss, fatal to
                # coherent function draws, hence the eval branch below.
                loc_g = self.loc[idx]
                out = loc_g + scale[idx] * torch.randn_like(loc_g)
            elif idx.dim() == 1:
                # Coherent posterior draw (ADR-0007): ONE table realization
                # W ~ q(W) gathered at every index, so repeated levels share
                # their embedding within the draw.
                table = self.loc + scale * torch.randn_like(self.loc)
                out = table[idx]
            else:
                # Sample-dimension pass (issue 0027): the leading dim of idx
                # indexes posterior draws — one fresh coherent table per
                # slice, never one draw broadcast S ways.
                S = idx.shape[0]
                noise = torch.randn(
                    S,
                    self.num_embeddings,
                    self.embedding_dim,
                    device=self.loc.device,
                    dtype=self.loc.dtype,
                )
                tables = self.loc + scale * noise  # (S, E, D)
                flat = idx.reshape(S, -1)  # (S, B)
                gathered = torch.gather(
                    tables,
                    1,
                    flat.unsqueeze(-1).expand(S, flat.shape[1], self.embedding_dim),
                )
                out = gathered.reshape(*idx.shape, self.embedding_dim)

            # Shared prior scale s from the per-feature PriorScale handle —
            # passed as a tensor so its gradient path stays alive.  The handle
            # stashes its own hyperprior KL on this call (issue #73); adding it
            # here too would double-count it in collect_kl.
            s = self.prior_scale_handle()  # positive scalar tensor

            # KL[N(loc, scale²) ‖ N(0, s²)], summed over all (level, dim) pairs.
            self._stash_kl(gaussian_kl(self.loc, scale, s))
        else:
            out = self.loc[idx]
            self.kl = torch.zeros(())

        return out

    # ── serialisation ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a closure-free, JSON-serialisable config dict."""
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "prior_scale_config": self.prior_scale_handle.get_config(),
            "kl_divisor": self.kl_divisor,
            "mode": self.mode,
        }

    @classmethod
    def from_config(cls, config: dict) -> BayesianEmbedding:
        from dune_bayes.priors.prior_scale import PriorScale

        cfg = dict(config)
        ps = PriorScale.from_config(cfg.pop("prior_scale_config"))
        return cls(prior_scale_handle=ps, **cfg)
