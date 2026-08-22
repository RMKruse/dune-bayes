"""UCI benchmark runner (ADR-0008, GitHub #102–#103, #176–#178)."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import shutil
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_harness = importlib.import_module("experiments._harness")
ArtifactPaths = _harness.ArtifactPaths
run_experiment = _harness.run_experiment

from dune_bayes.data import DataModule  # noqa: E402
from dune_bayes.families import (  # noqa: E402
    BaseFamily,
    BetaFamily,
    NegativeBinomialFamily,
    NormalFamily,
)
from dune_bayes.metrics import (  # noqa: E402
    crps,
    variance_decomposition,
)
from dune_bayes.model import BayesianNAMLSS  # noqa: E402
from dune_bayes.shapes import BayesianMLP, DeterministicMLP  # noqa: E402
from dune_bayes.utils import EPS  # noqa: E402
from experiments.uci_benchmark.adapters import (  # noqa: E402
    BamlssFixtureAdapter,
    BayesNamStyleAdapter,
    BenchmarkAdapter,
    DeepEnsembleAdapter,
    DeterministicNamlssAdapter,
    DuneBayesAdapter,
    LANAMAdapter,
    MeanOnlyGaussianAdapter,
    NampyNamlssAdapter,
    PlainMLPAdapter,
    PredictiveResult,
    ResponseTransform,
    predict_on_original_scale,
)

_EXPERIMENT_DIR = Path(__file__).resolve().parent


class _LocationOnlyShape(nn.Module):
    """Lift a one-parameter Bayesian feature effect into a Normal parameter vector."""

    def __init__(self, loc_net: nn.Module, param_count: int) -> None:
        super().__init__()
        self.loc_net = loc_net
        self.param_count = int(param_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return stochastic location contributions and zero scale contributions."""
        loc = self.loc_net(x)
        zeros = torch.zeros(
            (*loc.shape[:-1], self.param_count - 1),
            dtype=loc.dtype,
            device=loc.device,
        )
        return torch.cat([loc, zeros], dim=-1)


def _data_path(config: Mapping[str, Any], key: str) -> Path:
    """Resolve one configured data directory relative to this experiment."""
    path = Path(str(config["data"][key]))
    if not path.is_absolute():
        path = _EXPERIMENT_DIR / path
    return path.resolve()


def _experiment_path(value: str) -> Path:
    """Resolve a configured experiment path without touching package state."""
    path = Path(value)
    if not path.is_absolute():
        path = _EXPERIMENT_DIR / path
    return path.resolve()


def _command_path(value: str) -> str:
    """Resolve a configured command path only when it is path-like."""
    if "/" not in value and "\\" not in value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = _EXPERIMENT_DIR / path
    # Do not call Path.resolve(): virtualenv Python executables are often
    # symlinks to a base interpreter, and following them drops pyvenv.cfg.
    return str(path.absolute())


def _cache_key(dataset: Mapping[str, Any], *, smoke: bool) -> str:
    """Keep CI fixtures from masquerading as full benchmark downloads."""
    name = str(dataset["name"])
    return f"{name}-smoke" if smoke else name


def _catalog_frame(
    config: Mapping[str, Any], dataset: Mapping[str, Any]
) -> pd.DataFrame:
    """Fetch one pinned catalog dataset into the common numeric table contract."""
    source = dataset["source"]
    if source["kind"] == "uci":
        from ucimlrepo import fetch_ucirepo

        fetched = fetch_ucirepo(id=int(source["id"]))
        features = fetched.data.features.copy()
        targets = fetched.data.targets
    elif source["kind"] == "openml":
        from sklearn.datasets import fetch_openml

        features, targets = fetch_openml(
            data_id=int(source["id"]),
            return_X_y=True,
            as_frame=True,
            data_home=_data_path(config, "cache_dir") / "_openml",
        )
    else:
        raise ValueError(f"Unknown dataset source kind {source['kind']!r}.")

    if isinstance(targets, pd.DataFrame):
        target = targets.iloc[:, int(dataset.get("target_index", 0))]
    else:
        target = targets
    numeric_features = pd.get_dummies(features, dtype=float)
    frame = numeric_features.copy()
    response = str(dataset["response"])
    frame[response] = pd.to_numeric(target, errors="coerce").to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if dataset.get("response_transform") == "open_unit_interval":
        values = frame[response]
        if bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(f"{dataset['name']} response is not bounded in [0, 1].")
        # Beta has open support; this minimal affine contraction moves exact
        # boundary observations inward by the package-wide numerical floor.
        frame[response] = values * (1.0 - 2.0 * EPS) + EPS
    return frame


def _prepare_dataset(
    config: Mapping[str, Any], dataset: Mapping[str, Any], *, smoke: bool
) -> None:
    """Cache one source and create its split exactly once."""
    cache_dir = _data_path(config, "cache_dir")
    split_dir = _data_path(config, "split_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    name = str(dataset["name"])
    cache_key = _cache_key(dataset, smoke=smoke)
    cached = cache_dir / f"{cache_key}.csv"
    if not cached.exists():
        if smoke:
            shutil.copyfile(_EXPERIMENT_DIR / "fixtures" / f"{name}_smoke.csv", cached)
        elif "url" in dataset:
            temporary = cached.with_suffix(".csv.partial")
            try:
                with (
                    urllib.request.urlopen(str(dataset["url"])) as response,  # noqa: S310
                    temporary.open("wb") as handle,
                ):
                    shutil.copyfileobj(response, handle)
                temporary.replace(cached)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            temporary = cached.with_suffix(".csv.partial")
            try:
                _catalog_frame(config, dataset).to_csv(temporary, index=False)
                temporary.replace(cached)
            finally:
                temporary.unlink(missing_ok=True)

    split_path = split_dir / f"{cache_key}.npz"
    n_rows = len(pd.read_csv(cached))
    if split_path.exists():
        with np.load(split_path) as persisted:
            if "validation_indices" in persisted:
                return
            train_indices = persisted["train_indices"].copy()
            test_indices = persisted["test_indices"].copy()
    else:
        rng = np.random.default_rng(int(dataset["split_seed"]))
        indices = rng.permutation(n_rows)
        n_test = max(1, round(n_rows * float(config["data"]["test_fraction"])))
        train_indices = np.sort(indices[n_test:])
        test_indices = np.sort(indices[:n_test])
    validation_seed = int(dataset["split_seed"]) + 1_000_000
    validation_order = np.random.default_rng(validation_seed).permutation(train_indices)
    validation_fraction = float(config["data"]["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be inside (0, 1).")
    if len(train_indices) < 2:
        raise ValueError("An outer training partition needs at least two rows.")
    n_validation = max(
        1,
        round(len(train_indices) * validation_fraction),
    )
    n_validation = min(n_validation, len(train_indices) - 1)
    np.savez(
        split_path,
        train_indices=train_indices,
        fit_indices=np.sort(validation_order[n_validation:]),
        validation_indices=np.sort(validation_order[:n_validation]),
        validation_seed=np.asarray(validation_seed),
        test_indices=test_indices,
        n_rows=np.asarray(n_rows),
    )


def _data_subset(
    preprocessing: DataModule, frame: pd.DataFrame, *, response: str
) -> DataModule:
    """Reuse one fitted preprocessing state for a persisted row subset."""
    subset = copy.copy(preprocessing)
    subset.features = preprocessing.transform(frame)
    subset.target = torch.tensor(frame[response].to_numpy(), dtype=torch.float32)
    subset.n_obs = len(frame)
    return subset


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty list of records with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_dune_bayes_model(
    train_data: DataModule,
    family: BaseFamily,
    config: Mapping[str, Any],
    *,
    location_only: bool = False,
    deterministic: bool = False,
) -> BayesianNAMLSS:
    """Build the configured Bayesian or point-estimated additive model."""
    hidden_dims = [int(width) for width in config["architecture"]["hidden_dims"]]
    prior_scale = float(config["architecture"]["prior_scale"])
    activation = str(config["architecture"]["activation"])
    if deterministic:
        formula = {
            feature: DeterministicMLP(
                1,
                family.param_count,
                hidden_dims=hidden_dims,
                activation=activation,
            )
            for feature in train_data.features
        }
        model = BayesianNAMLSS(
            formula=formula,
            family=family,
            n_obs=train_data.n_obs,
            intercept_mode="point",
        )
    elif location_only:
        formula = {
            feature: _LocationOnlyShape(
                BayesianMLP(
                    1,
                    1,
                    hidden_dims=hidden_dims,
                    prior_scale=prior_scale,
                    kl_divisor=train_data.n_obs,
                    activation=activation,
                ),
                family.param_count,
            )
            for feature in train_data.features
        }
        model = BayesianNAMLSS(
            formula=formula,
            family=family,
            n_obs=train_data.n_obs,
            intercept_mode="point",
        )
    else:
        formula = {
            feature: BayesianMLP(
                1,
                family.param_count,
                hidden_dims=hidden_dims,
                prior_scale=prior_scale,
                kl_divisor=train_data.n_obs,
                activation=activation,
            )
            for feature in train_data.features
        }
        model = BayesianNAMLSS(
            formula=formula,
            family=family,
            n_obs=train_data.n_obs,
        )
    if isinstance(family, NegativeBinomialFamily):
        target = train_data.target
        mean = target.mean()
        variance = target.var(correction=0)
        dispersion = (
            (variance - mean) / mean.square() if mean > EPS else mean.new_tensor(EPS)
        )
        # The family adds EPS after softplus. Subtract it before the stable
        # inverse link; an EPS softplus floor gives the finite 2*EPS Poisson
        # limit when either empirical moment is non-positive (GitHub #176).
        linked = torch.stack((mean, dispersion))
        positive = (linked - EPS).clamp_min(EPS)
        raw = positive + torch.log(-torch.expm1(-positive))
        with torch.no_grad():
            model.intercept.loc.copy_(raw.to(model.intercept.loc))
    return model


def _score_dataset(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    paths: ArtifactPaths,
    *,
    smoke: bool,
) -> list[dict[str, object]]:
    """Fit the family-matched panel and write held-out headline metrics."""
    name = str(dataset["name"])
    cache_key = _cache_key(dataset, smoke=smoke)
    frame = pd.read_csv(_data_path(config, "cache_dir") / f"{cache_key}.csv")
    with np.load(_data_path(config, "split_dir") / f"{cache_key}.npz") as split:
        outer_train = frame.iloc[split["train_indices"]].reset_index(drop=True)
        fit = frame.iloc[split["fit_indices"]].reset_index(drop=True)
        validation = frame.iloc[split["validation_indices"]].reset_index(drop=True)
        test = frame.iloc[split["test_indices"]].reset_index(drop=True)

    response = str(dataset["response"])
    preprocessing = DataModule(outer_train, response=response, numeric_scaling={})
    train_data = _data_subset(preprocessing, fit, response=response)
    validation_data = _data_subset(preprocessing, validation, response=response)
    test_features = preprocessing.transform(test)
    test_target = torch.tensor(test[response].to_numpy(), dtype=torch.float32)
    family_name = str(dataset["family"])
    response_transform = ResponseTransform.fit(preprocessing.target, family=family_name)
    train_data.target = response_transform.to_model_scale(train_data.target)
    validation_data.target = response_transform.to_model_scale(validation_data.target)
    transform_path = paths.metrics / name / "response_transform.json"
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.write_text(
        json.dumps(
            {
                "fit_partition": "train",
                "loc": response_transform.loc,
                "method": response_transform.method,
                "n_fit": preprocessing.n_obs,
                "scale": response_transform.scale,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if family_name == "negative_binomial":
        family = NegativeBinomialFamily()
    elif family_name == "beta":
        family = BetaFamily()
    else:
        family = NormalFamily()
    hidden_dims = [int(width) for width in config["architecture"]["hidden_dims"]]
    model = _build_dune_bayes_model(train_data, family, config)
    draws_count = min(int(config["draws"]), 16) if smoke else int(config["draws"])
    predictive_samples = (
        min(int(config["predictive_samples"]), 32)
        if smoke
        else int(config["predictive_samples"])
    )
    comparison: list[dict[str, object]] = []
    training = config["training"]
    check_every = int(training["validation_check_every"])
    patience_checks = int(training["validation_patience_checks"])
    validation_draws = int(training["validation_draws"])

    def score(
        adapter: BenchmarkAdapter,
        scored_family: str,
        metric_dir: Path | None = None,
    ) -> PredictiveResult | None:
        """Fit and score one adapter through the shared contract."""
        adapter.fit(train_data, smoke=smoke)
        training_metadata = getattr(adapter, "training_metadata", None)
        if training_metadata:
            destination = metric_dir or paths.metrics / name / adapter.name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "training.json").write_text(
                json.dumps(
                    training_metadata,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if training_metadata["status"] == "failed":
                return None
        result = predict_on_original_scale(
            adapter,
            response_transform,
            test_features,
            test_target,
            draws=draws_count,
            predictive_samples=predictive_samples,
            seed=int(dataset["split_seed"]),
        )
        comparison.extend(
            _write_scores(
                result,
                adapter=adapter,
                target=test_target,
                dataset=name,
                family=scored_family,
                bins=int(config["calibration_bins"]),
                metric_dir=metric_dir or paths.metrics / name / adapter.name,
            )
        )
        return result

    adapter: BenchmarkAdapter = DuneBayesAdapter(
        model,
        family,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        randomized_pit=family_name == "negative_binomial",
        validation_data=validation_data,
        check_every=check_every,
        patience_checks=patience_checks,
        validation_draws=min(validation_draws, 16) if smoke else validation_draws,
        validation_seed=int(config["seed"]),
    )
    prediction = score(adapter, family_name, paths.metrics / name)
    if prediction is not None:
        _write_dune_bayes_uncertainty(
            prediction,
            model=model,
            dataset=name,
            family=family_name,
            metric_dir=paths.metrics / name,
            response_transform=response_transform,
        )

    deterministic_model = _build_dune_bayes_model(
        train_data,
        family,
        config,
        deterministic=True,
    )
    deterministic: BenchmarkAdapter = DeterministicNamlssAdapter(
        deterministic_model,
        family,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_epochs=0,
        randomized_pit=family_name == "negative_binomial",
        validation_data=validation_data,
        check_every=check_every,
        patience_checks=patience_checks,
        validation_draws=1,
        validation_seed=int(config["seed"]),
    )
    score(deterministic, family_name)
    bayesnam_config = config.get("baselines", {}).get("bayesnam_style", {})
    if bool(bayesnam_config.get("enabled", False)):
        if str(bayesnam_config["label"]) != BayesNamStyleAdapter.name:
            raise ValueError("BayesNAM-style baseline label must match issue #106.")
        if str(bayesnam_config["family"]) != "normal_homoscedastic":
            raise ValueError("BayesNAM-style baseline must use a Normal scale.")
        if str(bayesnam_config["effect"]) != "location_only":
            raise ValueError("BayesNAM-style baseline must be location-only.")
        bayesnam_family = NormalFamily()
        bayesnam_model = _build_dune_bayes_model(
            train_data,
            bayesnam_family,
            config,
            location_only=True,
        )
        bayesnam: BenchmarkAdapter = BayesNamStyleAdapter(
            bayesnam_model,
            bayesnam_family,
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            warmup_epochs=int(config["training"]["warmup_epochs"]),
            randomized_pit=False,
            validation_data=validation_data,
            check_every=check_every,
            patience_checks=patience_checks,
            validation_draws=min(validation_draws, 16) if smoke else validation_draws,
            validation_seed=int(config["seed"]),
        )
        bayesnam_prediction = score(bayesnam, "normal_homoscedastic")
        if prediction is not None and bayesnam_prediction is not None:
            _write_bayesnam_band_contrast(
                prediction,
                bayesnam_prediction,
                dataset=name,
                family=family_name,
                figure_dir=paths.figures / name,
                response_transform=response_transform,
            )
    namlss_config = config.get("baselines", {}).get("namlss", {})
    if bool(namlss_config.get("enabled", False)):
        namlss: BenchmarkAdapter = NampyNamlssAdapter(
            python=_command_path(str(namlss_config["python"])),
            runner=_experiment_path(str(namlss_config["runner"])),
            paper_code_dir=_experiment_path(str(namlss_config["paper_code_dir"])),
            family=family_name,
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            batch_size=int(namlss_config.get("batch_size", 512)),
        )
        score(namlss, family_name)
    lanam_config = config.get("baselines", {}).get("lanam", {})
    if bool(lanam_config.get("enabled", False)):
        lanam: BenchmarkAdapter = LANAMAdapter(
            python=_command_path(str(lanam_config["python"])),
            runner=_experiment_path(str(lanam_config["runner"])),
            family=family_name,
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            batch_size=int(lanam_config.get("batch_size", 512)),
        )
        score(lanam, "mean_only_laplace_gaussian")
    bamlss_config = config.get("baselines", {}).get("bamlss_reference", {})
    if bool(bamlss_config.get("enabled", False)):
        bamlss: BenchmarkAdapter = BamlssFixtureAdapter(
            dataset=name,
            fixture_dir=_experiment_path(str(bamlss_config["fixture_dir"])),
            response_transform=response_transform,
            n_fit=preprocessing.n_obs,
        )
        score(bamlss, family_name)
    plain: BenchmarkAdapter = PlainMLPAdapter(
        family,
        hidden_dims=hidden_dims,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        randomized_pit=family_name == "negative_binomial",
        validation_data=validation_data,
        check_every=check_every,
        patience_checks=patience_checks,
    )
    score(plain, family_name)
    ensemble: BenchmarkAdapter = DeepEnsembleAdapter(
        family,
        members=int(config["baselines"]["deep_ensemble_members"]),
        hidden_dims=hidden_dims,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        randomized_pit=family_name == "negative_binomial",
        validation_data=validation_data,
        check_every=check_every,
        patience_checks=patience_checks,
    )
    score(ensemble, family_name)
    mean_only_config = config["baselines"].get("mean_only_gaussian", {})
    if family_name == "normal" and bool(mean_only_config.get("enabled", False)):
        mean_only: BenchmarkAdapter = MeanOnlyGaussianAdapter(
            hidden_dims=hidden_dims,
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
        )
        score(mean_only, "gaussian_residual")
    return comparison


def _linked_parameter_draws(
    raw_draws: torch.Tensor,
    family: str,
    response_transform: ResponseTransform,
) -> dict[str, torch.Tensor]:
    """Convert raw network outputs to family-scale parameter draws."""
    if family == "negative_binomial":
        return {
            "mean": F.softplus(raw_draws[..., 0]) + EPS,
            "dispersion": F.softplus(raw_draws[..., 1]) + EPS,
        }
    if family == "beta":
        return {
            "mean": EPS + (1.0 - 2.0 * EPS) * torch.sigmoid(raw_draws[..., 0]),
            "precision": F.softplus(raw_draws[..., 1]) + EPS,
        }
    if family == "normal":
        loc, scale = response_transform.to_original_normal_parameters(
            raw_draws[..., 0], F.softplus(raw_draws[..., 1]) + EPS
        )
        return {
            "loc": loc,
            "scale": scale,
        }
    return {
        f"parameter_{index}": raw_draws[..., index]
        for index in range(int(raw_draws.shape[-1]))
    }


def _write_dune_bayes_uncertainty(
    prediction: PredictiveResult,
    *,
    model: BayesianNAMLSS,
    dataset: str,
    family: str,
    metric_dir: Path,
    response_transform: ResponseTransform,
) -> None:
    """Write the Bayesian-only parameter bands and variance decomposition."""
    raw_draws = prediction.parameter_draws
    if raw_draws is None:
        raise RuntimeError("dune-bayes prediction did not expose parameter draws.")

    linked_draws = _linked_parameter_draws(raw_draws, family, response_transform)

    band_rows: list[dict[str, object]] = []
    for parameter, draws in linked_draws.items():
        quantiles = torch.quantile(
            draws.to(torch.float64),
            torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64),
            dim=0,
        )
        for observation in range(int(draws.shape[1])):
            band_rows.append(
                {
                    "dataset": dataset,
                    "model": "dune_bayes",
                    "observation": observation,
                    "parameter": parameter,
                    "q05": float(quantiles[0, observation]),
                    "q50": float(quantiles[1, observation]),
                    "q95": float(quantiles[2, observation]),
                }
            )
    _write_csv(metric_dir / "parameter_bands.csv", band_rows)

    split = variance_decomposition(model, raw_draws)
    _write_csv(
        metric_dir / "variance_split.csv",
        [
            {
                "dataset": dataset,
                "model": "dune_bayes",
                "observation": observation,
                "aleatoric": float(
                    response_transform.to_original_variance(
                        split.aleatoric[observation]
                    )
                ),
                "epistemic": float(
                    response_transform.to_original_variance(
                        split.epistemic[observation]
                    )
                ),
                "total": float(
                    response_transform.to_original_variance(split.total[observation])
                ),
            }
            for observation in range(int(split.total.shape[0]))
        ],
    )


def _write_bayesnam_band_contrast(
    dune_prediction: PredictiveResult,
    bayesnam_prediction: PredictiveResult,
    *,
    dataset: str,
    family: str,
    figure_dir: Path,
    response_transform: ResponseTransform,
) -> None:
    """Draw the mean-only BayesNAM contrast against distributional dune-bayes."""
    if dune_prediction.parameter_draws is None:
        raise RuntimeError("dune-bayes prediction did not expose parameter draws.")
    if bayesnam_prediction.parameter_draws is None:
        raise RuntimeError("BayesNAM-style prediction did not expose parameter draws.")

    import matplotlib.pyplot as plt

    def interval(draws: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quantiles = torch.quantile(
            draws.to(torch.float64),
            torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64),
            dim=0,
        )
        arrays = [item.detach().cpu().numpy() for item in quantiles]
        return arrays[0], arrays[1], arrays[2]

    dune_params = _linked_parameter_draws(
        dune_prediction.parameter_draws, family, response_transform
    )
    bayesnam_params = _linked_parameter_draws(
        bayesnam_prediction.parameter_draws, "normal", response_transform
    )
    dune_items = list(dune_params.items())
    x = np.arange(int(dune_prediction.parameter_draws.shape[1]))
    bayesnam_label = BayesNamStyleAdapter.name

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    loc_lo, loc_mid, loc_hi = interval(bayesnam_params["loc"])
    dune_lo, dune_mid, dune_hi = interval(dune_items[0][1])
    axes[0].fill_between(x, dune_lo, dune_hi, alpha=0.22, label="dune_bayes")
    axes[0].plot(x, dune_mid, linewidth=1.2)
    axes[0].fill_between(
        x,
        loc_lo,
        loc_hi,
        alpha=0.22,
        label=bayesnam_label,
    )
    axes[0].plot(x, loc_mid, linewidth=1.2)
    axes[0].set(ylabel=dune_items[0][0], title=f"{dataset}: location effect bands")
    axes[0].legend(loc="best", fontsize=8)

    second_name, second_draws = dune_items[1]
    scale_lo, scale_mid, scale_hi = interval(second_draws)
    bayes_scale_lo, bayes_scale_mid, bayes_scale_hi = interval(bayesnam_params["scale"])
    axes[1].fill_between(x, scale_lo, scale_hi, alpha=0.22, label="dune_bayes")
    axes[1].plot(x, scale_mid, linewidth=1.2)
    axes[1].fill_between(
        x,
        bayes_scale_lo,
        bayes_scale_hi,
        alpha=0.16,
        label=bayesnam_label,
    )
    axes[1].plot(x, bayes_scale_mid, linewidth=1.2)
    axes[1].set(
        xlabel="held-out observation",
        ylabel=second_name,
        title="distributional parameter band vs learned homoscedastic scale",
    )
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "bayesnam_style_band_contrast.pdf")
    plt.close(fig)


def _write_scores(
    prediction: PredictiveResult,
    *,
    adapter: BenchmarkAdapter,
    target: torch.Tensor,
    dataset: str,
    family: str,
    bins: int,
    metric_dir: Path,
) -> list[dict[str, object]]:
    """Score one adapter prediction and write its public metric artifacts."""
    nll = -prediction.log_density
    crps_values = crps(prediction.samples, target)
    pit_values = prediction.cdf.to(torch.float64)

    summary = {
        "comparison_role": adapter.comparison_role,
        "dataset": dataset,
        "family": family,
        "model": adapter.name,
        "n_test": len(target),
    }
    _write_csv(
        metric_dir / "nll.csv",
        [{**summary, "mean_nll": float(nll.mean())}],
    )
    _write_csv(
        metric_dir / "crps.csv",
        [{**summary, "mean_crps": float(crps_values.mean())}],
    )
    counts, edges = np.histogram(pit_values.numpy(), bins=bins, range=(0.0, 1.0))
    _write_csv(
        metric_dir / "calibration.csv",
        [
            {
                "dataset": dataset,
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": int(count),
                "fraction": float(count / len(target)),
                "expected_fraction": float(1.0 / bins),
            }
            for index, count in enumerate(counts)
        ],
    )
    fractions = counts / len(target)
    return [
        {
            "comparison_role": adapter.comparison_role,
            "dataset": dataset,
            "family": family,
            "model": adapter.name,
            "uncertainty_scope": getattr(adapter, "uncertainty_scope", "predictive"),
            "n_test": len(target),
            "mean_nll": float(nll.mean()),
            "mean_crps": float(crps_values.mean()),
            "calibration_error": float(np.abs(fractions - 1.0 / bins).mean()),
        }
    ]


def _run(
    config: Mapping[str, Any],
    paths: ArtifactPaths,
    smoke: bool,
    *,
    dataset_name: str | None = None,
) -> None:
    """Prepare and score the configured benchmark datasets."""
    datasets = config["datasets"]
    if dataset_name is not None:
        datasets = [item for item in datasets if item["name"] == dataset_name]
        if not datasets:
            raise ValueError(f"Unknown dataset {dataset_name!r}.")
    elif smoke:
        datasets = datasets[:1]
    comparison: list[dict[str, object]] = []
    for dataset in datasets:
        _prepare_dataset(config, dataset, smoke=smoke)
        comparison.extend(_score_dataset(config, dataset, paths, smoke=smoke))
    _write_csv(paths.metrics / "comparison.csv", comparison)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the UCI benchmark panel from one complete config.

    Args:
        argv: Optional CLI arguments; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset")
    args = parser.parse_args(argv)

    def selected_run(
        config: Mapping[str, Any], paths: ArtifactPaths, smoke: bool
    ) -> None:
        """Bind the optional public dataset selector to the harness callback."""
        _run(config, paths, smoke, dataset_name=args.dataset)

    run_experiment(args.config, smoke=args.smoke, experiment=selected_run)


if __name__ == "__main__":
    main()
