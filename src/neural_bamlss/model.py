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
  - model.loss(X_dict, y): scalar ELBO loss (NLL + KL/N).
  - fit(): lightweight training loop built on the same loss; returns a history dict.
  - sample_posterior_predictive(X, T): MixtureSameFamily posterior predictive
    backed by draw_predictive (issue 0007, ADR-0003, GitHub #68).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from neural_bamlss.layers import VariationalDense, collect_kl, set_kl_beta


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
        # Response name captured by from_formula(); None for dict-built models.
        self.response: str | None = None
        # Register all shape functions as sub-modules so collect_kl() reaches them.
        self.nets = nn.ModuleDict(formula)
        self.family = family
        self.n_obs: int | None = int(n_obs) if n_obs is not None else None

        if feature_dropout is None:
            # 0.01 is NAMpy's default; 0.0 when weight posterior supplies stochasticity.
            self.feature_dropout = 0.0 if _has_bayesian_nets(formula) else 0.01
        else:
            self.feature_dropout = float(feature_dropout)

    # ── formula-string construction ───────────────────────────────────────────

    @classmethod
    def from_formula(
        cls,
        formula: str,
        family: Any,
        n_obs: int | None = None,
        feature_dropout: float | None = None,
        data: Any = None,
    ) -> "BayesianNAMLSS":
        """Construct a BayesianNAMLSS from a formula string (issue 0016).

        Parses an additive formula like
        ``"y ~ BayesianMLP(x1, prior_scale=0.5) + NeuralLinearMLP(x2)"``,
        resolves each term via ShapeFunctionRegistry, and builds the model.
        The response name left of ``~`` is stored as ``model.response``.
        When n_obs is given, it is auto-wired as kl_divisor into each
        Bayesian term (unless that term sets kl_divisor itself), so the
        objective is the documented mean-NLL + KL/N.

        Args:
            formula: Additive formula string; each term is a registered
                shape-function name applied to one feature, with optional
                literal keyword arguments forwarded to its constructor.
            family: Family object with .param_count, as in __init__.
            n_obs: As in __init__.
            feature_dropout: As in __init__.
            data: Optional DataModule (issue 0022); supplies n_obs from the
                training data so KL/N needs no explicit n_obs argument.

        Returns:
            BayesianNAMLSS over the parsed terms.

        Raises:
            ValueError: If the formula cannot be parsed or names an
                unregistered shape function.
        """
        from neural_bamlss.formula import build_formula, parse_formula

        if data is not None and n_obs is None:
            n_obs = data.n_obs
        parsed = parse_formula(formula)
        model = cls(
            formula=build_formula(parsed, family=family, n_obs=n_obs),
            family=family,
            n_obs=n_obs,
            feature_dropout=feature_dropout,
        )
        model.response = parsed.response
        return model

    # ── forward ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_input(X: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        # Interaction keys ("x1:x2") are fed by concatenating constituent tensors
        # along the feature dimension; simple keys are looked up directly.
        if ":" in name:
            return torch.cat([X[f] for f in name.split(":")], dim=-1)
        return X[name]

    def predict_params(self, X: dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble the summed predictor from per-feature contributions.

        Single owner of the "inputs → summed predictor" assembly (issue 0060):
        forward() and draw_predictive both delegate here, so interaction-key
        handling and feature dropout cannot drift between the training and
        sampling paths.

        Args:
            X: Dict mapping feature name → tensor of shape (batch, in_features).
                Interaction terms (keyed ``"x1:x2"``) are fed by concatenating
                ``X["x1"]`` and ``X["x2"]`` along the feature dimension.

        Returns:
            Summed predictor tensor of shape (batch, param_count). Stochastic
            when Bayesian nets are present (one reparameterization draw).
        """
        contribs = [
            self.nets[name](self._get_input(X, name)) for name in self.feature_names
        ]
        # Sum contributions (additive model); stack → sum over feature dim.
        stacked = torch.stack(contribs, dim=0)  # (F, batch, param_count)
        if self.training and self.feature_dropout > 0.0:
            # Feature-level dropout: zero entire feature contributions, then
            # rescale by F / #survivors so the additive sum keeps its expected
            # magnitude (inverted-dropout rescale by the *realized* survivor
            # count, exact for the additive sum; clamp guards the all-dropped
            # draw, which yields an all-zero prediction).
            mask = torch.bernoulli(
                torch.full(
                    (len(contribs), 1, 1),
                    1.0 - self.feature_dropout,
                    device=stacked.device,
                    dtype=stacked.dtype,
                )
            )
            stacked = stacked * mask * len(contribs) / mask.sum().clamp(min=1.0)
        return stacked.sum(dim=0)  # (batch, param_count)

    def forward(self, X: dict[str, torch.Tensor]) -> torch.distributions.Distribution:
        """Single stochastic forward pass.

        Args:
            X: Dict mapping feature name → tensor of shape (batch, in_features).
                Interaction terms (keyed ``"x1:x2"``) are fed by concatenating
                ``X["x1"]`` and ``X["x2"]`` along the feature dimension.

        Returns:
            A torch.distributions.Distribution with batch_shape (batch,).
        """
        return self.family(self.predict_params(X))

    # ── loss ──────────────────────────────────────────────────────────────────

    def _loss_components(
        self, X: dict[str, torch.Tensor], y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One stochastic pass → (loss, nll, kl); shared by loss() and fit()."""
        dist = self(X)
        nll = -dist.log_prob(y).mean()
        kl = collect_kl(self)
        return nll + kl, nll, kl

    def loss(self, X: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """ELBO loss: mean-NLL + KL/N as a scalar tensor.

        KL/N is already scaled by kl_divisor inside each VariationalDense;
        collect_kl sums the scaled values.

        Args:
            X: Feature dict mapping name → tensor of shape (batch, in_features).
            y: Target tensor of shape (batch,).

        Returns:
            Scalar loss tensor (one reparameterization draw).
        """
        total, _, _ = self._loss_components(X, y)
        return total

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: dict[str, torch.Tensor] | Any,
        y: torch.Tensor | None = None,
        epochs: int = 100,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        batch_size: int | None = None,
        seed: int | None = None,
        callbacks: list | None = None,
    ) -> dict[str, list[float]]:
        """Train the model on (X, y) using the ELBO loss.

        KL warm-up (ADR-0001, issue 0004) is auto-injected: β ramps from 0→1
        over the first warmup_epochs epochs, guarding against posterior collapse.
        Set warmup_epochs=0 to disable.

        When ``batch_size`` is given and ``X`` is a DataModule, training uses
        minibatches from ``dm.dataloader()``.  The KL divisor stays at
        ``n_obs`` (full training-set size) — never at ``batch_size`` — so the
        ELBO estimate is unbiased (issue 0026).

        Args:
            X: Feature dict; each value is (n_obs, in_features). May instead
                be a DataModule (issue 0022), in which case y is taken from it
                and must not be passed.
            y: Target tensor of shape (n_obs,). None when X is a DataModule.
            epochs: Number of passes over the training data.
            lr: Adam learning rate.
            warmup_epochs: Epochs over which β ramps 0→1. 0 disables warm-up.
            batch_size: Minibatch size. None (default) keeps full-batch training.
                Requires X to be a DataModule.
            seed: Integer seed for the DataLoader shuffle generator. Only used
                when ``batch_size`` is set; controls batch ordering within
                one model object (consistent with the reproducibility rule).
            callbacks: Optional list of callables with signature (epoch: int) → None,
                called at the start of each epoch alongside the warm-up callback.

        Returns:
            History dict with keys 'loss', 'nll', 'kl' — one value per epoch.
        """
        from neural_bamlss.data import DataModule

        data_module: DataModule | None = None
        if isinstance(X, DataModule):
            data_module = X
            X, y = data_module.features, data_module.target
        if y is None:
            raise TypeError("fit() requires y unless X is a DataModule")
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

        # Full-batch training is minibatching with a single batch (issue 0026):
        # one loop, one history-append; the mean over one batch is the value.
        use_minibatch = batch_size is not None and data_module is not None

        self.train()
        for epoch in range(epochs):
            for cb in _callbacks:
                cb(epoch)

            if use_minibatch:
                # Seed a fresh generator each epoch so the shuffle sequence is
                # reproducible within one model object when seed is given.
                gen: torch.Generator | None = None
                if seed is not None:
                    gen = torch.Generator().manual_seed(seed + epoch)
                batches = data_module.dataloader(  # type: ignore[union-attr]
                    batch_size=batch_size, generator=gen
                )
            else:
                batches = [(X, y)]

            epoch_loss = epoch_nll = epoch_kl = 0.0
            n_batches = 0
            for batch_X, batch_y in batches:
                opt.zero_grad()
                loss, nll, kl = self._loss_components(batch_X, batch_y)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach())
                epoch_nll += float(nll.detach())
                epoch_kl += float(kl.detach())
                n_batches += 1
            # Record per-epoch mean across batches.
            history["loss"].append(epoch_loss / n_batches)
            history["nll"].append(epoch_nll / n_batches)
            history["kl"].append(epoch_kl / n_batches)

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
            MixtureSameFamily with batch_shape (n,), backed by draw_predictive.
        """
        from neural_bamlss.sampling.log_lik_sampler import draw_predictive

        # Drawing needs no response (GitHub #68): scoring y is a separate job
        # (pointwise_log_lik) composed by the compare module for WAIC/LOO.
        return draw_predictive(self, X, T=T).predictive

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
