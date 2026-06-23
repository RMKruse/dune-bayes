"""Common predictive adapter for benchmark models (ADR-0008, GitHub #103).

Related approaches such as a plain MLP and deep ensemble are deliberately
evaluated here as sanity floors: unlike dune-bayes, they do not model separate
distributional aleatoric and effect-level epistemic uncertainty.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
        parameter_draws: Optional raw distribution-parameter draws with shape
            ``(T, n, param_count)``. Only Bayesian adapters expose this; the
            common scoring contract consumes the first three fields.
    """

    samples: torch.Tensor
    log_density: torch.Tensor
    cdf: torch.Tensor
    parameter_draws: torch.Tensor | None = None


class BenchmarkAdapter(Protocol):
    """Small interface shared by every benchmark model."""

    name: str
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


class DuneBayesAdapter:
    """Expose ``BayesianNAMLSS`` through the benchmark prediction contract."""

    name = "dune_bayes"
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
    ) -> None:
        self._model = model
        self._family = family
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._warmup_epochs = warmup_epochs
        self._randomized_pit = randomized_pit

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit dune-bayes with the configured full or smoke budget."""
        epochs = 1 if smoke else self._epochs
        self._model.fit(
            train_data,
            epochs=epochs,
            lr=self._learning_rate,
            warmup_epochs=min(self._warmup_epochs, epochs),
        )

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


class BayesNamStyleAdapter(DuneBayesAdapter):
    """Labeled mean-only variational NAM baseline (ADR-0008, GitHub #106).

    This is deliberately a degenerate dune-bayes configuration rather than a
    separate package implementation: Bayesian location shape functions plus a
    point intercept that learns one homoscedastic Normal scale.
    """

    name = "BayesNAM-style (our implementation)"
    uncertainty_scope = "mean_only_variational_location"


class BamlssFixtureAdapter:
    """BAMLSS reference predictions loaded from maintainer-run fixtures (#107).

    BAMLSS stays an external R reference.  This adapter only validates and
    scores the per-observation predictive fixture written by ``bamlss/run.R``.
    """

    name = "bamlss_reference"
    uncertainty_scope = "distributional_bamlss_fixture"

    def __init__(self, *, dataset: str, fixture_dir: Path) -> None:
        self._dataset = dataset
        self._fixture_dir = fixture_dir

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
        return PredictiveResult(samples=samples, log_density=log_density, cdf=cdf)


class _PlainMLP(torch.nn.Module):
    """Minimal point-estimate network used only by the experiment harness."""

    def __init__(self, in_features: int, hidden_dims: list[int]) -> None:
        super().__init__()
        widths = [in_features, *hidden_dims, 1]
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            layers.append(torch.nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(torch.nn.Tanh())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one point prediction per observation."""
        out: torch.Tensor = self.network(features)
        return out.squeeze(-1)


def _feature_matrix(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate the harness's already-standardized feature tensors."""
    return torch.cat([value.to(torch.float32) for value in features.values()], dim=-1)


class PlainMLPAdapter:
    """Point-estimate MLP with a fitted homoscedastic Gaussian residual."""

    name = "plain_mlp"
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
        self._target_loc = torch.tensor(0.0)
        self._target_scale = torch.tensor(1.0)
        self._residual_scale = torch.tensor(1.0)

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit the point predictor, then estimate its Gaussian residual scale."""
        features = _feature_matrix(train_data.features)
        target = train_data.target
        self._target_loc = target.mean()
        # Adding the named floor under the square root keeps constant-response
        # fixtures finite without a magic clamp on a learned quantity.
        self._target_scale = torch.sqrt(target.var(unbiased=False) + EPS)
        standardized_target = (target - self._target_loc) / self._target_scale
        model = _PlainMLP(features.shape[-1], self._hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        epochs = 1 if smoke else self._epochs
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(features), standardized_target)
            loss.backward()
            optimizer.step()
        self._model = model
        with torch.no_grad():
            residual = target - self._mean(features)
            self._residual_scale = torch.sqrt(residual.square().mean() + EPS)

    def _mean(self, features: torch.Tensor) -> torch.Tensor:
        """Return response-scale means from a fitted network."""
        if self._model is None:
            raise RuntimeError("fit() must be called before predict().")
        mean: torch.Tensor = self._model(features)
        return mean * self._target_scale + self._target_loc

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Return the fitted homoscedastic Gaussian predictive."""
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
    """Uniform mixture of independently initialized Gaussian-residual MLPs."""

    name = "deep_ensemble"
    uncertainty_scope = "predictive_only"

    def __init__(
        self,
        *,
        members: int,
        hidden_dims: list[int],
        epochs: int,
        learning_rate: float,
    ) -> None:
        if members < 2:
            raise ValueError("A deep ensemble requires at least two members.")
        self._members = [
            PlainMLPAdapter(
                hidden_dims=hidden_dims,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            for _ in range(members)
        ]

    def fit(self, train_data: DataModule, *, smoke: bool) -> None:
        """Fit independently initialized members on the same training rows."""
        for member in self._members:
            member.fit(train_data, smoke=smoke)

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        draws: int,
        predictive_samples: int,
        seed: int,
    ) -> PredictiveResult:
        """Return the equal-weight mixture predictive across ensemble members."""
        samples_per_member = math.ceil(predictive_samples / len(self._members))
        predictions = [
            member.predict(
                features,
                target,
                draws=draws,
                predictive_samples=samples_per_member,
                seed=seed + index,
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
        return PredictiveResult(
            samples=torch.cat(
                [prediction.samples for prediction in predictions], dim=0
            )[:predictive_samples],
            log_density=log_density,
            cdf=torch.stack([prediction.cdf for prediction in predictions]).mean(dim=0),
        )
