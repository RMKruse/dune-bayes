"""BayesianNAMLSS — walking skeleton with KL warm-up.

ADR-0001/0003/0004, issue 0003/0004/0007 / GitHub #4/#5/#8.

Additive Bayesian model: per-feature shape functions → sum → family distribution.
Training objective: mean-NLL + KL/N (negative ELBO).

Design:
  - formula: dict[str, nn.Module] mapping feature name → shape function instance.
    Shape functions may be Bayesian (BayesianMLP) or deterministic; the latter
    contribute zero KL (degenerate zero-variance contributors, CONTEXT.md).
  - family: callable with .param_count that maps (batch, param_count) → Distribution.
  - n_obs: training-set size N for the KL/N weighting. May be provided at
    construction or inferred from the target during fit().
  - feature_dropout: rate at which feature contributions are randomly zeroed.
    Defaults to 0 when any Bayesian net is present (Bayesian weights supply
    the stochasticity — no need for additional feature noise).
  - forward() is a single stochastic pass (call()-style, ADR-0003).
  - model.Loss: callable (X_dict, y) → scalar ELBO loss (NLL + KL/N).
  - fit(): lightweight training loop; returns a history dict.
  - sample_posterior_predictive(X, T): MixtureSameFamily posterior predictive
    backed by LogLikSampler (issue 0007, ADR-0003).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from neural_bamlss.layers import collect_kl
from neural_bamlss.layers.variational_dense import VariationalDense, set_kl_beta


def _has_bayesian_nets(formula: dict[str, nn.Module]) -> bool:
    """Return True if any shape function contains at least one VariationalDense."""
    for net in formula.values():
        for m in net.modules():
            if isinstance(m, VariationalDense):
                return True
    return False


class BayesianNAMLSS(nn.Module):
    """Bayesian additive model for location, scale, and shape.

    Args:
        formula: Mapping of feature name → shape function (nn.Module).
            Bayesian nets (BayesianMLP) contribute KL; deterministic nets
            contribute zero KL, enabling partial-Bayesian formulas.
        family: Family object with .param_count (int) and __call__(params)
            returning a torch.distributions.Distribution.
        n_obs: Training-set size N used for KL/N weighting. If None, must be
            provided to fit() or set via n_obs attribute before training.
        feature_dropout: Fraction of feature contributions randomly zeroed
            per forward pass. Defaults to 0.0 when any Bayesian net is present
            (weight posterior supplies the required stochasticity).
    """

    def __init__(
        self,
        formula: dict[str, nn.Module],
        family: Any,
        n_obs: int | None = None,
        feature_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.feature_names: list[str] = list(formula.keys())
        # Register all shape functions as sub-modules so collect_kl() reaches them.
        self.nets = nn.ModuleDict(formula)
        self.family = family
        self.n_obs: int | None = int(n_obs) if n_obs is not None else None

        if feature_dropout is None:
            # 0.01 is NAMpy's default; 0.0 when weight posterior supplies stochasticity.
            self.feature_dropout = 0.0 if _has_bayesian_nets(formula) else 0.01
        else:
            self.feature_dropout = float(feature_dropout)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, X: dict[str, torch.Tensor]) -> torch.distributions.Distribution:
        """Single stochastic forward pass.

        Args:
            X: Dict mapping feature name → tensor of shape (batch, in_features).

        Returns:
            A torch.distributions.Distribution with batch_shape (batch,).
        """
        contribs = [self.nets[name](X[name]) for name in self.feature_names]
        # Sum contributions (additive model); stack → sum over feature dim.
        summed = torch.stack(contribs, dim=0).sum(dim=0)  # (batch, param_count)
        if self.training and self.feature_dropout > 0.0:
            # Feature-level dropout: zero entire feature contribution randomly.
            mask = torch.bernoulli(
                torch.full((len(contribs),), 1.0 - self.feature_dropout)
            )
            summed = summed * mask.sum() / max(mask.sum().item(), 1.0)
        return self.family(summed)

    # ── loss ──────────────────────────────────────────────────────────────────

    @property
    def Loss(self):
        """ELBO loss callable: (X_dict, y) → scalar tensor.

        Returns mean-NLL + KL/N.  KL/N is already scaled by kl_divisor inside
        each VariationalDense; collect_kl sums the scaled values.
        """

        def _loss(X: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
            dist = self(X)
            nll = -dist.log_prob(y).mean()
            kl = collect_kl(self)
            return nll + kl

        return _loss

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        callbacks: list | None = None,
    ) -> dict[str, list[float]]:
        """Train the model on (X, y) using the ELBO loss.

        KL warm-up (ADR-0001, issue 0004) is auto-injected: β ramps from 0→1
        over the first warmup_epochs epochs, guarding against posterior collapse.
        Set warmup_epochs=0 to disable.

        Args:
            X: Feature dict; each value is (n_obs, in_features).
            y: Target tensor of shape (n_obs,).
            epochs: Number of full-data gradient steps.
            lr: Adam learning rate.
            warmup_epochs: Epochs over which β ramps 0→1. 0 disables warm-up.
            callbacks: Optional list of callables with signature (epoch: int) → None,
                called at the start of each epoch alongside the warm-up callback.

        Returns:
            History dict with keys 'loss', 'nll', 'kl' — one value per epoch.
        """
        if self.n_obs is None:
            self.n_obs = int(y.shape[0])

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history: dict[str, list[float]] = {"loss": [], "nll": [], "kl": []}

        # Build the epoch-start callback list: warm-up is always first when active.
        _callbacks: list = []
        if warmup_epochs > 0:
            _callbacks.append(
                lambda epoch: set_kl_beta(self, min(1.0, epoch / warmup_epochs))
            )
        if callbacks:
            _callbacks.extend(callbacks)

        self.train()
        for epoch in range(epochs):
            for cb in _callbacks:
                cb(epoch)

            opt.zero_grad()
            dist = self(X)
            nll = -dist.log_prob(y).mean()
            kl = collect_kl(self)
            loss = nll + kl
            loss.backward()
            opt.step()

            history["loss"].append(float(loss.detach()))
            history["nll"].append(float(nll.detach()))
            history["kl"].append(float(kl.detach()))

        self.eval()
        return history

    # ── posterior predictive ──────────────────────────────────────────────────

    def sample_posterior_predictive(
        self,
        X: dict[str, torch.Tensor],
        T: int = 200,
    ) -> torch.distributions.MixtureSameFamily:
        """Build the T-draw posterior predictive as a MixtureSameFamily.

        Runs T stochastic forward passes and assembles a uniform mixture over
        the resulting family distributions (ADR-0003).  Spread across components
        = epistemic uncertainty; within each component = family aleatoric.

        Args:
            X: Feature dict {name: Tensor[n, in_features]}.
            T: Number of posterior weight draws. Defaults to 200 (T_predict).

        Returns:
            MixtureSameFamily with batch_shape (n,), backed by LogLikSampler.
        """
        from neural_bamlss.sampling.log_lik_sampler import LogLikSampler

        # Dummy y of zeros — only summed_samples and predictive are used here.
        # pointwise_loglik requires y; call LogLikSampler directly for WAIC/LOO.
        n = next(iter(X.values())).shape[0]
        y_dummy = torch.zeros(n)
        result = LogLikSampler()(self, X, y_dummy, T=T)
        return result.predictive

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and config to a checkpoint file.

        Uses torch.save to write a dict containing the state_dict and the
        hyperparameters needed to reconstruct the model (n_obs, feature_dropout,
        feature_names).  Architecture must be supplied by the caller on load.

        H5 format is not supported: HDF5 weight-name collisions across variational
        layers cause silent corruption (spike-confirmed on the TF/Keras stack);
        use .pt instead.

        Args:
            path: Destination file path.  Must not end in .h5.

        Raises:
            ValueError: If path ends with .h5.
        """
        path = Path(path)
        if path.suffix.lower() == ".h5":
            raise ValueError(
                f"H5 format is not supported for BayesianNAMLSS ({path}). "
                "HDF5 weight-name collisions across variational layers cause "
                "silent corruption. Use .pt format with save()/load() instead."
            )
        checkpoint = {
            "state_dict": self.state_dict(),
            "n_obs": self.n_obs,
            "feature_dropout": self.feature_dropout,
            "feature_names": self.feature_names,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        formula: dict[str, nn.Module],
        family: Any,
    ) -> "BayesianNAMLSS":
        """Load a BayesianNAMLSS from a checkpoint written by save().

        The caller must supply a formula with the same architecture as the
        saved model — this is the standard PyTorch pattern of separating
        architecture construction from weight restoration.

        Args:
            path: Path to the checkpoint file written by save().
            formula: Mapping of feature name → shape function with the same
                architecture as the saved model.
            family: Family object matching the saved model.

        Returns:
            BayesianNAMLSS with weights restored from the checkpoint.
        """
        checkpoint = torch.load(path, weights_only=True)
        model = cls(
            formula=formula,
            family=family,
            n_obs=checkpoint["n_obs"],
            feature_dropout=checkpoint["feature_dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
