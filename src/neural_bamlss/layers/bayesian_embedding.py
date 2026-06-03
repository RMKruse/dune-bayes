"""BayesianEmbedding — variational categorical first mapping (issue 0012 / GitHub #13).



Each category level gets a mean-field Gaussian weight posterior under a shared
per-feature PriorScale — the neural analog of a BAMLSS random effect.  All
levels share one PriorScale, so rare levels shrink toward the prior (partial
pooling) while common levels stay data-driven.

A point-embedding fallback (mode="point") is available but not the default;
it removes the per-level credible interval and the shrinkage behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_bamlss.layers.variational_dense import _RHO_INIT
from neural_bamlss.priors.prior_scale import PriorScale

# Small initialisation scale so the embedding table starts near zero.
_LOC_INIT_STD = 0.01


class BayesianEmbedding(nn.Module):
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
        super().__init__()
        if mode not in ("variational", "point"):
            raise ValueError(f"unknown mode {mode!r}; choose 'variational' or 'point'")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        if prior_scale_handle is None:
            prior_scale_handle = PriorScale(mode="fixed", scale=1.0)
        self.prior_scale_handle = prior_scale_handle
        self.kl_divisor = float(kl_divisor)
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

        # Non-trainable warm-up factor β; driven by set_kl_beta().
        self.register_buffer("kl_beta", torch.tensor(1.0))
        self.kl: torch.Tensor = torch.zeros(())

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
            sampled = self.loc + scale * torch.randn_like(self.loc)
            out = sampled[idx]

            # Shared prior scale s from the per-feature PriorScale handle.
            s = self.prior_scale_handle()  # positive scalar tensor

            # KL[N(loc, scale²) ‖ N(0, s²)], summed over all (level, dim) pairs.
            embedding_kl = torch.sum(
                torch.log(s)
                - torch.log(scale)
                + (scale.pow(2) + self.loc.pow(2)) / (2.0 * s.pow(2))
                - 0.5
            )

            # Hyperprior KL from PriorScale (zero unless hierarchical mode).
            ps_kl = self.prior_scale_handle.kl()

            self.kl = self.kl_beta * (embedding_kl + ps_kl) / self.kl_divisor
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
    def from_config(cls, config: dict) -> "BayesianEmbedding":
        cfg = dict(config)
        ps = PriorScale.from_config(cfg.pop("prior_scale_config"))
        return cls(prior_scale_handle=ps, **cfg)
