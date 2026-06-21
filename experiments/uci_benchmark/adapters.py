"""Common predictive adapter for benchmark models (ADR-0008, GitHub #103).

Related approaches such as a plain MLP and deep ensemble are deliberately
evaluated here as sanity floors: unlike dune-bayes, they do not model separate
distributional aleatoric and effect-level epistemic uncertainty.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

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
    """

    samples: torch.Tensor
    log_density: torch.Tensor
    cdf: torch.Tensor


class BenchmarkAdapter(Protocol):
    """Small interface shared by every benchmark model."""

    name: str

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
        return self.network(features).squeeze(-1)


def _feature_matrix(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate the harness's already-standardized feature tensors."""
    return torch.cat([value.to(torch.float32) for value in features.values()], dim=-1)


class PlainMLPAdapter:
    """Point-estimate MLP with a fitted homoscedastic Gaussian residual."""

    name = "plain_mlp"

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
        return self._model(features) * self._target_scale + self._target_loc

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


class DeepEnsembleAdapter:
    """Uniform mixture of independently initialized Gaussian-residual MLPs."""

    name = "deep_ensemble"

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
