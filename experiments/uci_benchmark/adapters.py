"""Common benchmark adapters (ADR-0008, GitHub #103/#175/#177/#178).

The four primary adapters share family links and scoring; supplemental
mean-only comparators remain labeled separately.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch

from dune_bayes.data import DataModule
from dune_bayes.families import BaseFamily
from dune_bayes.metrics import pit
from dune_bayes.model import BayesianNAMLSS
from dune_bayes.sampling import draw_predictive, pointwise_log_lik
from dune_bayes.utils import EPS


@dataclass(frozen=True)
class PredictiveResult:
    """Per-observation predictive quantities consumed by common scoring.

    Attributes:
        samples: Response draws with shape ``(M, n)``.
        log_density: Predictive log-density at each observed response, shape
            ``(n,)``.
        cdf: Predictive CDF at each observed response, shape ``(n,)``.
        parameter_draws: Optional raw distribution-parameter vectors with shape
            ``(T, n, param_count)``. ``T`` is posterior draws for DUNE, one for
            a deterministic MLP, and ensemble members for the deep ensemble.
    """

    samples: torch.Tensor
    log_density: torch.Tensor
    cdf: torch.Tensor
    parameter_draws: torch.Tensor | None = None


@dataclass(frozen=True)
class ResponseTransform:
    """Train-only response standardization (ADR-0008, GitHub #175).

    Attributes:
        method: Standard affine scaling or identity for native-support families.
        loc: Training-response location used by the affine transform.
        scale: Training-response scale used by the affine transform.
    """

    method: Literal["standard", "identity"]
    loc: float
    scale: float

    @classmethod
    def fit(cls, target: torch.Tensor, *, family: str) -> ResponseTransform:
        """Fit an affine transform from benchmark training responses only.

        Args:
            target: Training responses.
            family: Configured benchmark family name.

        Returns:
            Fitted standard or identity response transform.

        Raises:
            ValueError: If the benchmark family is unsupported.
        """
        if family in ("beta", "negative_binomial"):
            return cls(method="identity", loc=0.0, scale=1.0)
        if family != "normal":
            raise ValueError(f"Unsupported benchmark response family {family!r}.")
        return cls(
            method="standard",
            loc=float(target.mean()),
            # The named floor keeps constant responses finite (numerical rule 3).
            scale=max(float(target.std(unbiased=False)), EPS),
        )

    def to_model_scale(self, target: torch.Tensor) -> torch.Tensor:
        """Apply the fitted transform without refitting on held-out rows.

        Args:
            target: Responses on their original scale.

        Returns:
            Responses on the model's scale.
        """
        return (target - self.loc) / self.scale

    def to_original_prediction(self, prediction: PredictiveResult) -> PredictiveResult:
        """Restore predictions and log-density to original response units.

        Args:
            prediction: Predictive quantities on model scale.

        Returns:
            Predictive samples and density on original scale; CDF unchanged.
        """
        return PredictiveResult(
            samples=prediction.samples * self.scale + self.loc,
            # p_y(y) = p_z((y-loc)/scale) / scale.
            log_density=prediction.log_density - math.log(self.scale),
            cdf=prediction.cdf,
            parameter_draws=prediction.parameter_draws,
        )

    def to_model_prediction(self, prediction: PredictiveResult) -> PredictiveResult:
        """Convert original-scale predictive quantities to model units.

        Args:
            prediction: Predictive quantities on original response scale.

        Returns:
            Predictive samples and density on model scale; CDF unchanged.
        """
        return PredictiveResult(
            samples=(prediction.samples - self.loc) / self.scale,
            log_density=prediction.log_density + math.log(self.scale),
            cdf=prediction.cdf,
            parameter_draws=prediction.parameter_draws,
        )

    def to_original_normal_parameters(
        self, loc: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore Normal location and scale parameters to response units.

        Args:
            loc: Model-scale Normal locations.
            scale: Model-scale Normal standard deviations.

        Returns:
            Original-scale ``(location, standard deviation)`` tensors.
        """
        return loc * self.scale + self.loc, scale * self.scale

    def to_original_variance(self, variance: torch.Tensor) -> torch.Tensor:
        """Restore a model-scale variance to squared response units.

        Args:
            variance: Model-scale variance values.

        Returns:
            Original-scale variance values.
        """
        return variance * self.scale**2


class BenchmarkAdapter(Protocol):
    """Small interface shared by every benchmark model.

    Attributes:
        name: Stable model label written to artifacts.
        comparison_role: ``primary`` or ``supplemental`` panel role.
        uncertainty_scope: Human-readable uncertainty capability label.
    """

    name: str
    comparison_role: str
    uncertainty_scope: str

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit the model on the persisted training partition."""

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Return held-out predictive samples, log-density, and CDF."""


def predict_on_original_scale(
    adapter: BenchmarkAdapter,
    response_transform: ResponseTransform,
    features: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    draws: int,
    predictive_samples: int,
    seed: int,
) -> PredictiveResult:
    """Run one adapter and restore its predictive contract (GitHub #175).

    Args:
        adapter: Benchmark model adapter.
        response_transform: Shared transform fitted on training responses.
        features: Held-out model features.
        target: Held-out responses on original scale.
        draws: Number of distribution-parameter draws.
        predictive_samples: Number of response draws.
        seed: Predictive sampling seed.

    Returns:
        Predictive quantities on original response scale.
    """
    prediction = adapter.predict(
        features,
        response_transform.to_model_scale(target),
        draws=draws,
        predictive_samples=predictive_samples,
        seed=seed,
    )
    return response_transform.to_original_prediction(prediction)


@dataclass
class _ValidationCheckpoint:
    """Track one validation-NLL winner and its patience budget (GitHub #178)."""

    patience_checks: int
    best_epoch: int | None = None
    best_nll: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    checks_without_improvement: int = 0
    failure: str | None = None
    trace: list[dict[str, float | int | None]] = field(default_factory=list)

    def observe(
        self, epoch: int, validation_nll: float, model: torch.nn.Module
    ) -> bool:
        """Record a check and return whether training must stop."""
        if not math.isfinite(validation_nll):
            self.trace.append({"epoch": epoch, "nll": None})
            self.failure = "non_finite_validation_nll"
            return True
        self.trace.append({"epoch": epoch, "nll": validation_nll})
        if self.best_nll is None or validation_nll < self.best_nll:
            self.best_epoch = epoch
            self.best_nll = validation_nll
            self.best_state = copy.deepcopy(model.state_dict())
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1
        return self.checks_without_improvement >= self.patience_checks

    def restore(self, model: torch.nn.Module) -> bool:
        """Restore the best finite checkpoint when one was observed."""
        if self.best_state is None:
            return False
        model.load_state_dict(self.best_state)
        return True

    def metadata(
        self,
        *,
        epochs_ceiling: int,
        epochs_completed: int,
        effective_epochs: int,
        check_every: int,
        monitor_start_epoch: int,
        validation_seed: int | None,
        validation_draws: int,
        history: dict[str, list[float]],
    ) -> dict[str, object]:
        """Return JSON-ready stopping and checkpoint evidence."""
        return {
            "status": "failed" if self.failure else "completed",
            "failure": self.failure,
            "epochs_ceiling": epochs_ceiling,
            "epochs_completed": epochs_completed,
            "stopped_early": epochs_completed < effective_epochs,
            "check_every": check_every,
            "patience_checks": self.patience_checks,
            "monitor_start_epoch": monitor_start_epoch,
            "validation_seed": validation_seed,
            "validation_draws": validation_draws,
            "validation_checks": self.trace,
            "best_epoch": self.best_epoch,
            "best_validation_nll": self.best_nll,
            "restored_best_checkpoint": self.best_state is not None,
            "history": history,
        }


def _check_validation(
    epoch: int, *, epochs: int, check_every: int, monitor_start_epoch: int
) -> bool:
    """Check on the frozen cadence and once at a shorter smoke/final ceiling."""
    return (epoch > monitor_start_epoch and epoch % check_every == 0) or epoch == epochs


def mean_mixture_nll(log_density: torch.Tensor) -> float:
    """Return mean NLL for equally weighted log-density rows (GitHub #178).

    Args:
        log_density: Per-component pointwise log densities, shape ``(T, n)``.

    Returns:
        Mean negative log density of the uniform mixture.
    """
    return float(
        -(
            torch.logsumexp(log_density.to(torch.float64), dim=0)
            - math.log(int(log_density.shape[0]))
        ).mean()
    )


class DuneBayesAdapter:
    """Expose ``BayesianNAMLSS`` through the benchmark prediction contract.

    Args:
        model: Additive Bayesian or deterministic NAMLSS model.
        family: Dataset response family and parameter links.
        epochs: Full-run epoch ceiling.
        learning_rate: Adam learning rate.
        warmup_epochs: KL warm-up length before validation monitoring.
        randomized_pit: Whether to randomize PIT on discrete support.
        validation_data: Persisted validation partition shared by the panel.
        check_every: Epoch cadence for validation NLL.
        patience_checks: Checks without improvement before stopping.
        validation_draws: Fixed posterior draw count for validation NLL.
        validation_seed: Fixed posterior draw seed for validation NLL.
    """

    name = "dune_bayes"
    comparison_role = "primary"
    uncertainty_scope = "distributional_parameter_bands"

    def __init__(
        self,
        model: BayesianNAMLSS,
        family: BaseFamily,
        *,
        epochs: int,
        learning_rate: float,
        warmup_epochs: int,
        randomized_pit: bool,
        validation_data: DataModule,
        check_every: int,
        patience_checks: int,
        validation_draws: int,
        validation_seed: int,
    ) -> None:
        self._model = model
        self._family = family
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._warmup_epochs = warmup_epochs
        self._randomized_pit = randomized_pit
        self._validation_data = validation_data
        self._check_every = check_every
        self._patience_checks = patience_checks
        self._validation_draws = validation_draws
        self._validation_seed = validation_seed
        self.training_metadata: dict[str, object] = {}

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit dune-bayes with the configured full or smoke budget."""
        epochs = 1 if smoke else self._epochs
        tracker = _ValidationCheckpoint(self._patience_checks)

        def validate(epoch: int, logs: Mapping[str, float]) -> bool:
            """Evaluate the persisted validation rows after warm-up."""
            del logs
            epoch_number = epoch + 1
            if not _check_validation(
                epoch_number,
                epochs=epochs,
                check_every=self._check_every,
                monitor_start_epoch=min(self._warmup_epochs, epochs),
            ):
                return False
            with torch.random.fork_rng():
                torch.manual_seed(self._validation_seed)
                posterior = draw_predictive(
                    self._model,
                    self._validation_data.features,
                    T=self._validation_draws,
                )
                log_lik = pointwise_log_lik(
                    self._model,
                    posterior.summed_samples,
                    self._validation_data.target,
                ).to(torch.float64)
                # The posterior mixture must stay in log-space when one draw
                # assigns negligible validation density (numerical rule 2).
                validation_nll = mean_mixture_nll(log_lik)
            return tracker.observe(epoch_number, validation_nll, self._model)

        history = self._model.fit(
            train_data,
            epochs=epochs,
            lr=self._learning_rate,
            warmup_epochs=min(self._warmup_epochs, epochs),
            epoch_end_callbacks=[validate],
        )
        restored = tracker.restore(self._model)
        self.training_metadata = tracker.metadata(
            epochs_ceiling=self._epochs,
            epochs_completed=len(history["loss"]),
            effective_epochs=epochs,
            check_every=self._check_every,
            monitor_start_epoch=self._warmup_epochs,
            validation_seed=self._validation_seed,
            validation_draws=self._validation_draws,
            history=history,
        )
        self.training_metadata["restored_best_checkpoint"] = restored

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Produce one coherent posterior predictive on held-out observations."""
        posterior = draw_predictive(self._model, features, T=draws)
        log_lik = pointwise_log_lik(self._model, posterior.summed_samples, target).to(
            torch.float64
        )
        # The uniform mixture density is accumulated in log-space; forming
        # probabilities first would underflow for poor held-out predictions.
        log_density = torch.logsumexp(log_lik, dim=0) - math.log(draws)
        samples = posterior.predictive.sample((predictive_samples,))
        cdf = pit(
            self._family,
            posterior.summed_samples,
            target,
            randomized=self._randomized_pit,
            seed=seed,
        )
        return PredictiveResult(
            samples=samples,
            log_density=log_density,
            cdf=cdf,
            parameter_draws=posterior.summed_samples,
        )


class DeterministicNamlssAdapter(DuneBayesAdapter):
    """Point-estimated additive NAMLSS behind the shared prediction contract.

    Inherits the DUNE fit/predict path; its model contains deterministic shape
    functions and a point intercept (GitHub #177).
    """

    name = "deterministic_namlss"
    comparison_role = "primary"
    uncertainty_scope = "deterministic_distributional"


class BayesNamStyleAdapter(DuneBayesAdapter):
    """Labeled mean-only variational NAM baseline (ADR-0008, GitHub #106).

    This is deliberately a degenerate dune-bayes configuration rather than a
    separate package implementation: Bayesian location shape functions plus a
    point intercept that learns one homoscedastic Normal scale.
    """

    name = "BayesNAM-style (our implementation)"
    comparison_role = "supplemental"
    uncertainty_scope = "mean_only_variational_location"


class BamlssFixtureAdapter:
    """BAMLSS reference predictions loaded from maintainer-run fixtures (#107).

    BAMLSS stays an external R reference.  This adapter only validates and
    scores the per-observation predictive fixture written by ``bamlss/run.R``.
    """

    name = "bamlss_reference"
    comparison_role = "supplemental"
    uncertainty_scope = "distributional_bamlss_fixture"

    def __init__(
        self,
        *,
        dataset: str,
        fixture_dir: Path,
        response_transform: ResponseTransform,
        n_fit: int,
    ) -> None:
        self._dataset = dataset
        self._fixture_dir = fixture_dir
        self._response_transform = response_transform
        self._n_fit = n_fit

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fixtures were already fit by the seeded R script."""
        del train_data, smoke

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Load and validate held-out BAMLSS predictive quantities."""
        del features, draws, seed
        dataset_dir = self._fixture_dir / self._dataset
        prediction_path = dataset_dir / "predictions.csv"
        provenance_path = dataset_dir / "provenance.json"
        if not prediction_path.is_file():
            raise RuntimeError(f"Missing BAMLSS fixture CSV: {prediction_path}.")
        if not provenance_path.is_file():
            raise RuntimeError(f"Missing BAMLSS fixture provenance: {provenance_path}.")
        with provenance_path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        for key in ("script_version", "seed", "date"):
            if key not in provenance:
                raise RuntimeError(
                    f"BAMLSS fixture provenance is missing required key {key!r}."
                )
        transform = provenance.get("response_transform")
        if not isinstance(transform, dict):
            raise RuntimeError("BAMLSS fixture is missing response-transform metadata.")
        if (
            transform.get("method") != self._response_transform.method
            or transform.get("fit_partition") != "train"
            or transform.get("n_fit") != self._n_fit
            or not math.isclose(
                float(transform.get("loc", math.nan)),
                self._response_transform.loc,
                rel_tol=0.0,
                abs_tol=EPS,
            )
            or not math.isclose(
                float(transform.get("scale", math.nan)),
                self._response_transform.scale,
                rel_tol=0.0,
                abs_tol=EPS,
            )
        ):
            raise RuntimeError(
                "BAMLSS fixture response transform does not match the shared split."
            )

        with prediction_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"BAMLSS fixture CSV is empty: {prediction_path}.")
        rows.sort(key=lambda row: int(row["observation"]))
        n_obs = int(target.shape[0])
        if len(rows) != n_obs:
            raise RuntimeError(
                f"BAMLSS fixture has {len(rows)} rows, expected {n_obs}."
            )
        sample_columns = sorted(
            column for column in rows[0] if column.startswith("sample_")
        )
        if len(sample_columns) < predictive_samples:
            raise RuntimeError(
                "BAMLSS fixture has "
                f"{len(sample_columns)} sample columns, expected at least "
                f"{predictive_samples}."
            )

        log_density = torch.tensor(
            [float(row["log_density"]) for row in rows],
            dtype=torch.float64,
        )
        cdf = torch.tensor([float(row["cdf"]) for row in rows], dtype=torch.float64)
        samples = torch.tensor(
            [
                [float(row[column]) for row in rows]
                for column in sample_columns[:predictive_samples]
            ],
            dtype=torch.float32,
        )
        observations = [int(row["observation"]) for row in rows]
        if observations != list(range(n_obs)):
            raise RuntimeError("BAMLSS fixture observations must be 0-based and dense.")
        if {row["dataset"] for row in rows} != {self._dataset}:
            raise RuntimeError("BAMLSS fixture dataset column does not match config.")
        if not bool(torch.isfinite(log_density).all()):
            raise RuntimeError("BAMLSS fixture contains non-finite log_density.")
        if not bool(((cdf >= 0.0) & (cdf <= 1.0)).all()):
            raise RuntimeError("BAMLSS fixture CDF values must be inside [0, 1].")
        return self._response_transform.to_model_prediction(
            PredictiveResult(samples=samples, log_density=log_density, cdf=cdf)
        )


class _PlainMLP(torch.nn.Module):
    """Minimal point-estimate network used only by the experiment harness."""

    def __init__(
        self, in_features: int, out_features: int, hidden_dims: list[int]
    ) -> None:
        super().__init__()
        widths = [in_features, *hidden_dims, out_features]
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            layers.append(torch.nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(torch.nn.Tanh())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one raw distribution-parameter vector per observation."""
        out: torch.Tensor = self.network(features)
        return out


def _feature_matrix(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate the harness's already-standardized feature tensors."""
    return torch.cat([value.to(torch.float32) for value in features.values()], dim=-1)


class PlainMLPAdapter:
    """Global point-estimate MLP emitting every configured family parameter.

    Args:
        family: Dataset response family and parameter links.
        hidden_dims: Hidden-layer widths.
        epochs: Full-run training epochs.
        learning_rate: Adam learning rate.
        randomized_pit: Whether to randomize PIT on discrete support.
        validation_data: Persisted validation partition shared by the panel.
        check_every: Epoch cadence for validation NLL.
        patience_checks: Checks without improvement before stopping.
    """

    name = "plain_mlp"
    comparison_role = "primary"
    uncertainty_scope = "deterministic_distributional"

    def __init__(
        self,
        family: BaseFamily,
        *,
        hidden_dims: list[int],
        epochs: int,
        learning_rate: float,
        randomized_pit: bool,
        validation_data: DataModule | None = None,
        check_every: int = 10,
        patience_checks: int = 5,
    ) -> None:
        self._family = family
        self._hidden_dims = hidden_dims
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._randomized_pit = randomized_pit
        self._validation_data = validation_data
        self._check_every = check_every
        self._patience_checks = patience_checks
        self._model: _PlainMLP | None = None
        self.training_metadata: dict[str, object] = {}

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit all raw family parameters by maximum likelihood.

        Args:
            train_data: Shared preprocessed training partition.
            smoke: Use one epoch when true.
        """
        features = _feature_matrix(train_data.features)
        target = train_data.target
        validation_data = self._validation_data or train_data
        model = _PlainMLP(
            features.shape[-1], self._family.param_count, self._hidden_dims
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        epochs = 1 if smoke else self._epochs
        tracker = _ValidationCheckpoint(self._patience_checks)
        history: dict[str, list[float]] = {"loss": []}
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = -self._family.log_prob(model(features), target).mean()
            loss.backward()
            optimizer.step()
            history["loss"].append(float(loss.detach()))
            if _check_validation(
                epoch,
                epochs=epochs,
                check_every=self._check_every,
                monitor_start_epoch=0,
            ):
                with torch.no_grad():
                    validation_nll = float(
                        -self._family.log_prob(
                            model(_feature_matrix(validation_data.features)),
                            validation_data.target,
                        ).mean()
                    )
                if tracker.observe(epoch, validation_nll, model):
                    break
        restored = tracker.restore(model)
        self._model = model
        self.training_metadata = tracker.metadata(
            epochs_ceiling=self._epochs,
            epochs_completed=len(history["loss"]),
            effective_epochs=epochs,
            check_every=self._check_every,
            monitor_start_epoch=0,
            validation_seed=None,
            validation_draws=1,
            history=history,
        )
        self.training_metadata["restored_best_checkpoint"] = restored

    def _params(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw family parameters from a fitted network."""
        if self._model is None:
            raise RuntimeError("fit() must be called before predict().")
        params: torch.Tensor = self._model(features)
        return params

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Return the fitted family predictive.

        Args:
            features: Shared held-out feature tensors.
            target: Held-out target on model scale.
            draws: Ignored for a point-estimated model.
            predictive_samples: Number of response samples.
            seed: Predictive sampling and PIT seed.

        Returns:
            Predictive samples, log-density, PIT, and raw parameter vector.
        """
        del draws
        with torch.no_grad():
            params = self._params(_feature_matrix(features))
            distribution = self._family(params)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                samples = distribution.sample((predictive_samples,))
            return PredictiveResult(
                samples=samples,
                log_density=distribution.log_prob(target).to(torch.float64),
                cdf=pit(
                    self._family,
                    params.unsqueeze(0),
                    target,
                    randomized=self._randomized_pit,
                    seed=seed,
                ),
                parameter_draws=params.unsqueeze(0),
            )


class MeanOnlyGaussianAdapter:
    """Supplemental mean-only MLP with a Gaussian residual.

    Args:
        hidden_dims: Hidden-layer widths.
        epochs: Full-run training epochs.
        learning_rate: Adam learning rate.
    """

    name = "mean_only_gaussian"
    comparison_role = "supplemental"
    uncertainty_scope = "predictive_only"

    def __init__(
        self,
        *,
        hidden_dims: list[int],
        epochs: int,
        learning_rate: float,
    ) -> None:
        self._hidden_dims = hidden_dims
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._model: _PlainMLP | None = None
        self._residual_scale = torch.tensor(1.0)

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit the point predictor and its homoscedastic residual scale.

        Args:
            train_data: Shared preprocessed training partition.
            smoke: Use one epoch when true.
        """
        features = _feature_matrix(train_data.features)
        target = train_data.target
        model = _PlainMLP(features.shape[-1], 1, self._hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        for _ in range(1 if smoke else self._epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(features).squeeze(-1), target)
            loss.backward()
            optimizer.step()
        self._model = model
        with torch.no_grad():
            residual = target - self._mean(features)
            self._residual_scale = torch.sqrt(residual.square().mean() + EPS)

    def _mean(self, features: torch.Tensor) -> torch.Tensor:
        """Return model-scale means from a fitted network."""
        if self._model is None:
            raise RuntimeError("fit() must be called before predict().")
        return self._model(features).squeeze(-1)

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Return the fitted homoscedastic Gaussian predictive.

        Args:
            features: Shared held-out feature tensors.
            target: Held-out target on model scale.
            draws: Ignored for a point-estimated model.
            predictive_samples: Number of response samples.
            seed: Predictive sampling seed.

        Returns:
            Predictive samples, log-density, and Gaussian CDF.
        """
        del draws
        with torch.no_grad():
            mean = self._mean(_feature_matrix(features))
            distribution = torch.distributions.Normal(mean, self._residual_scale)
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn((predictive_samples, len(target)), generator=generator)
            samples = mean.unsqueeze(0) + self._residual_scale * noise
            standardized = (target - mean) / self._residual_scale
            cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
            return PredictiveResult(
                samples=samples,
                log_density=distribution.log_prob(target).to(torch.float64),
                cdf=cdf.to(torch.float64),
            )


class NampyNamlssAdapter:
    """External NAMpy/NAMLSS comparator behind the common scoring contract.

    The TensorFlow-era implementation is invoked in a separate Python process
    so neither TensorFlow nor NAMpy can leak into the ``dune_bayes`` package
    namespace (ADR-0006, GitHub #104).
    """

    name = "nampy_namlss"
    comparison_role = "supplemental"
    uncertainty_scope = "deterministic_distributional"

    def __init__(
        self,
        *,
        python: str,
        runner: Path,
        paper_code_dir: Path,
        family: str,
        epochs: int,
        learning_rate: float,
        batch_size: int = 512,
    ) -> None:
        self._python = python
        self._runner = runner
        self._paper_code_dir = paper_code_dir
        self._family = family
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._train_matrix: torch.Tensor | None = None
        self._train_target: torch.Tensor | None = None
        self._smoke = False

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Capture the shared training partition for the external process."""
        self._smoke = smoke
        self._train_matrix = _feature_matrix(train_data.features)
        self._train_target = train_data.target.to(torch.float32)

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Invoke the configured NAMLSS runner and validate its predictions."""
        if self._train_matrix is None or self._train_target is None:
            raise RuntimeError("fit() must be called before predict().")

        test_matrix = _feature_matrix(features)
        with tempfile.TemporaryDirectory(prefix="dune-bayes-nampy-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.npz"
            output_path = tmp_path / "prediction.npz"
            np.savez(
                input_path,
                train_features=self._train_matrix.detach().cpu().numpy(),
                train_target=self._train_target.detach().cpu().numpy(),
                test_features=test_matrix.detach().cpu().numpy(),
                test_target=target.detach().cpu().numpy(),
            )
            completed = subprocess.run(
                [
                    self._python,
                    str(self._runner),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--paper-code-dir",
                    str(self._paper_code_dir),
                    "--family",
                    self._family,
                    "--draws",
                    str(draws),
                    "--predictive-samples",
                    str(predictive_samples),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(1 if self._smoke else self._epochs),
                    "--learning-rate",
                    str(self._learning_rate),
                    "--batch-size",
                    str(self._batch_size),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "NAMpy NAMLSS runner failed with exit code "
                    f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError(
                    f"NAMLSS runner completed but did not write {output_path}."
                )
            with np.load(output_path) as output:
                samples_array = output["samples"]
                log_density_array = output["log_density"]
                cdf_array = output["cdf"]

        samples = torch.tensor(samples_array, dtype=torch.float32)
        log_density = torch.tensor(log_density_array, dtype=torch.float64)
        cdf = torch.tensor(cdf_array, dtype=torch.float64)
        n_obs = int(target.shape[0])
        if samples.shape != (predictive_samples, n_obs):
            raise RuntimeError(
                "NAMLSS runner returned samples with shape "
                f"{tuple(samples.shape)}, expected {(predictive_samples, n_obs)}."
            )
        if log_density.shape != (n_obs,):
            raise RuntimeError(
                "NAMLSS runner returned log_density with shape "
                f"{tuple(log_density.shape)}, expected {(n_obs,)}."
            )
        if cdf.shape != (n_obs,):
            raise RuntimeError(
                "NAMLSS runner returned cdf with shape "
                f"{tuple(cdf.shape)}, expected {(n_obs,)}."
            )
        if not bool(torch.isfinite(log_density).all()):
            raise RuntimeError("NAMLSS runner returned non-finite log_density.")
        if not bool(((cdf >= 0.0) & (cdf <= 1.0)).all()):
            raise RuntimeError("NAMLSS runner returned CDF values outside [0, 1].")
        return PredictiveResult(samples=samples, log_density=log_density, cdf=cdf)


class LANAMAdapter:
    """External LA-NAM comparator behind the common scoring contract.

    LA-NAM is the closest mean-only Bayesian NAM baseline (ADR-0008,
    GitHub #105).  It is invoked in a separate process so the optional
    pinned git dependency remains isolated in the experiments tier.
    """

    name = "lanam"
    comparison_role = "supplemental"
    uncertainty_scope = "mean_only_laplace_location"

    def __init__(
        self,
        *,
        python: str,
        runner: Path,
        family: str,
        epochs: int,
        learning_rate: float,
        batch_size: int = 512,
    ) -> None:
        self._python = python
        self._runner = runner
        self._family = family
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._train_matrix: torch.Tensor | None = None
        self._train_target: torch.Tensor | None = None
        self._smoke = False

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Capture the shared training partition for the external process."""
        self._smoke = smoke
        self._train_matrix = _feature_matrix(train_data.features)
        self._train_target = train_data.target.to(torch.float32)

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Invoke the configured LA-NAM runner and validate its predictions."""
        if self._train_matrix is None or self._train_target is None:
            raise RuntimeError("fit() must be called before predict().")

        test_matrix = _feature_matrix(features)
        with tempfile.TemporaryDirectory(prefix="dune-bayes-lanam-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.npz"
            output_path = tmp_path / "prediction.npz"
            np.savez(
                input_path,
                train_features=self._train_matrix.detach().cpu().numpy(),
                train_target=self._train_target.detach().cpu().numpy(),
                test_features=test_matrix.detach().cpu().numpy(),
                test_target=target.detach().cpu().numpy(),
            )
            completed = subprocess.run(
                [
                    self._python,
                    str(self._runner),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--family",
                    self._family,
                    "--draws",
                    str(draws),
                    "--predictive-samples",
                    str(predictive_samples),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(1 if self._smoke else self._epochs),
                    "--learning-rate",
                    str(self._learning_rate),
                    "--batch-size",
                    str(self._batch_size),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "LA-NAM runner failed with exit code "
                    f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError(
                    f"LA-NAM runner completed but did not write {output_path}."
                )
            with np.load(output_path) as output:
                samples_array = output["samples"]
                log_density_array = output["log_density"]
                cdf_array = output["cdf"]

        samples = torch.tensor(samples_array, dtype=torch.float32)
        log_density = torch.tensor(log_density_array, dtype=torch.float64)
        cdf = torch.tensor(cdf_array, dtype=torch.float64)
        n_obs = int(target.shape[0])
        if samples.shape != (predictive_samples, n_obs):
            raise RuntimeError(
                "LA-NAM runner returned samples with shape "
                f"{tuple(samples.shape)}, expected {(predictive_samples, n_obs)}."
            )
        if log_density.shape != (n_obs,):
            raise RuntimeError(
                "LA-NAM runner returned log_density with shape "
                f"{tuple(log_density.shape)}, expected {(n_obs,)}."
            )
        if cdf.shape != (n_obs,):
            raise RuntimeError(
                "LA-NAM runner returned cdf with shape "
                f"{tuple(cdf.shape)}, expected {(n_obs,)}."
            )
        if not bool(torch.isfinite(log_density).all()):
            raise RuntimeError("LA-NAM runner returned non-finite log_density.")
        if not bool(((cdf >= 0.0) & (cdf <= 1.0)).all()):
            raise RuntimeError("LA-NAM runner returned CDF values outside [0, 1].")
        return PredictiveResult(samples=samples, log_density=log_density, cdf=cdf)


class DeepEnsembleAdapter:
    """Uniform mixture of independently initialized distributional MLPs.

    Args:
        family: Dataset response family and parameter links.
        members: Number of independently initialized global MLPs.
        hidden_dims: Hidden-layer widths per member.
        epochs: Full-run training epochs per member.
        learning_rate: Adam learning rate.
        randomized_pit: Whether to randomize PIT on discrete support.
        validation_data: Persisted validation partition shared by all members.
        check_every: Epoch cadence for validation NLL.
        patience_checks: Checks without improvement before stopping.
    """

    name = "deep_ensemble"
    comparison_role = "primary"
    uncertainty_scope = "ensemble_distributional"

    def __init__(
        self,
        family: BaseFamily,
        *,
        members: int,
        hidden_dims: list[int],
        epochs: int,
        learning_rate: float,
        randomized_pit: bool,
        validation_data: DataModule | None = None,
        check_every: int = 10,
        patience_checks: int = 5,
    ) -> None:
        if members < 2:
            raise ValueError("A deep ensemble requires at least two members.")
        self._family = family
        self._randomized_pit = randomized_pit
        self._epochs = epochs
        self._validation_data = validation_data
        self._check_every = check_every
        self._patience_checks = patience_checks
        self._members = [
            PlainMLPAdapter(
                family,
                hidden_dims=hidden_dims,
                epochs=epochs,
                learning_rate=learning_rate,
                randomized_pit=randomized_pit,
                validation_data=validation_data,
                check_every=check_every,
                patience_checks=patience_checks,
            )
            for _ in range(members)
        ]
        self.training_metadata: dict[str, object] = {}

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit independently initialized members on the same training rows.

        Args:
            train_data: Shared preprocessed training partition.
            smoke: Use one epoch per member when true.
        """
        for member in self._members:
            member.fit(train_data, smoke=smoke)
        validation_data = self._validation_data or train_data
        members = [member.training_metadata for member in self._members]
        failed = any(item["status"] == "failed" for item in members)
        selection_nll: float | None = None
        if not failed:
            predictions = [
                member.predict(
                    validation_data.features,
                    validation_data.target,
                    draws=1,
                    predictive_samples=1,
                    seed=index,
                )
                for index, member in enumerate(self._members)
            ]
            selection_nll = mean_mixture_nll(
                torch.stack([prediction.log_density for prediction in predictions])
            )
            failed = not math.isfinite(selection_nll)
            if failed:
                selection_nll = None
        self.training_metadata = {
            "status": "failed" if failed else "completed",
            "failure": "non_finite_validation_nll" if failed else None,
            "epochs_ceiling": self._epochs,
            "epochs_completed": max(
                cast(int, item["epochs_completed"]) for item in members
            ),
            "stopped_early": any(bool(item["stopped_early"]) for item in members),
            "check_every": self._check_every,
            "patience_checks": self._patience_checks,
            "monitor_start_epoch": 0,
            "validation_seed": None,
            "validation_draws": 1,
            "candidate_selection_validation_nll": selection_nll,
            "restored_best_checkpoint": all(
                bool(item["restored_best_checkpoint"]) for item in members
            ),
            "history": {},
            "members": members,
        }

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Return the equal-weight mixture predictive across ensemble members.

        Args:
            features: Shared held-out feature tensors.
            target: Held-out target on model scale.
            draws: Ignored by point-estimated members.
            predictive_samples: Number of mixture response samples.
            seed: Mixture assignment, member sampling, and PIT seed.

        Returns:
            Equal-weight mixture samples, density, PIT, and member parameters.
        """
        generator = torch.Generator().manual_seed(seed)
        assignments = torch.randint(
            len(self._members), (predictive_samples,), generator=generator
        )
        sample_counts = torch.bincount(
            assignments, minlength=len(self._members)
        ).tolist()
        predictions = [
            member.predict(
                features,
                target,
                draws=draws,
                predictive_samples=max(int(sample_counts[index]), 1),
                seed=seed + index + 1,
            )
            for index, member in enumerate(self._members)
        ]
        member_log_density = torch.stack(
            [prediction.log_density for prediction in predictions]
        )
        # Equal-weight mixture likelihood stays in log-space; this remains
        # finite when one member assigns negligible density to an observation.
        log_density = torch.logsumexp(member_log_density, dim=0) - math.log(
            len(self._members)
        )
        parameter_draws = torch.cat(
            [
                prediction.parameter_draws
                for prediction in predictions
                if prediction.parameter_draws is not None
            ]
        )
        if int(parameter_draws.shape[0]) != len(self._members):
            raise RuntimeError("Ensemble member did not expose family parameters.")
        return PredictiveResult(
            samples=torch.cat(
                [
                    prediction.samples[: int(sample_counts[index])]
                    for index, prediction in enumerate(predictions)
                ],
                dim=0,
            ),
            log_density=log_density,
            # Randomize the predictive mixture's count jump once, after member
            # CDF averaging; per-member randomization is not a valid mixture PIT.
            cdf=pit(
                self._family,
                parameter_draws,
                target,
                randomized=self._randomized_pit,
                seed=seed,
            ),
            parameter_draws=parameter_draws,
        )
