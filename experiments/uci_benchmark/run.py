"""UCI benchmark panel runner and baseline scoring (ADR-0008, GitHub #102–#103)."""

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_harness = importlib.import_module("experiments._harness")
ArtifactPaths = _harness.ArtifactPaths
run_experiment = _harness.run_experiment

from dune_bayes.data import DataModule  # noqa: E402
from dune_bayes.families import (  # noqa: E402
    BetaFamily,
    NegativeBinomialFamily,
    NormalFamily,
)
from dune_bayes.metrics import (  # noqa: E402
    crps,
    variance_decomposition,
)
from dune_bayes.model import BayesianNAMLSS  # noqa: E402
from dune_bayes.shapes import BayesianMLP  # noqa: E402
from dune_bayes.utils import EPS  # noqa: E402
from experiments.uci_benchmark.adapters import (  # noqa: E402
    BenchmarkAdapter,
    DeepEnsembleAdapter,
    DuneBayesAdapter,
    LANAMAdapter,
    NampyNamlssAdapter,
    PlainMLPAdapter,
    PredictiveResult,
)

_EXPERIMENT_DIR = Path(__file__).resolve().parent


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
    if split_path.exists():
        return
    n_rows = len(pd.read_csv(cached))
    rng = np.random.default_rng(int(dataset["split_seed"]))
    indices = rng.permutation(n_rows)
    n_test = max(1, round(n_rows * float(config["data"]["test_fraction"])))
    np.savez(
        split_path,
        train_indices=np.sort(indices[n_test:]),
        test_indices=np.sort(indices[:n_test]),
        n_rows=np.asarray(n_rows),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty list of records with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _score_dataset(
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    paths: ArtifactPaths,
    *,
    smoke: bool,
) -> list[dict[str, object]]:
    """Fit dune-bayes and write held-out headline metrics."""
    name = str(dataset["name"])
    cache_key = _cache_key(dataset, smoke=smoke)
    frame = pd.read_csv(_data_path(config, "cache_dir") / f"{cache_key}.csv")
    with np.load(_data_path(config, "split_dir") / f"{cache_key}.npz") as split:
        train = frame.iloc[split["train_indices"]].reset_index(drop=True)
        test = frame.iloc[split["test_indices"]].reset_index(drop=True)

    response = str(dataset["response"])
    train_data = DataModule(train, response=response, numeric_scaling={})
    test_features = train_data.transform(test)
    test_target = torch.tensor(test[response].to_numpy(), dtype=torch.float32)
    family_name = str(dataset["family"])
    if family_name == "negative_binomial":
        family = NegativeBinomialFamily()
    elif family_name == "beta":
        family = BetaFamily()
    else:
        family = NormalFamily()
    hidden_dims = [int(width) for width in config["architecture"]["hidden_dims"]]
    formula = {
        feature: BayesianMLP(
            1,
            family.param_count,
            hidden_dims=hidden_dims,
            prior_scale=float(config["architecture"]["prior_scale"]),
            kl_divisor=train_data.n_obs,
            activation=str(config["architecture"]["activation"]),
        )
        for feature in train_data.features
    }
    model = BayesianNAMLSS(
        formula=formula,
        family=family,
        n_obs=train_data.n_obs,
    )
    draws_count = min(int(config["draws"]), 16) if smoke else int(config["draws"])
    predictive_samples = (
        min(int(config["predictive_samples"]), 32)
        if smoke
        else int(config["predictive_samples"])
    )
    adapter: BenchmarkAdapter = DuneBayesAdapter(
        model,
        family,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        randomized_pit=family_name == "negative_binomial",
    )
    adapter.fit(train_data, smoke=smoke)
    prediction = adapter.predict(
        test_features,
        test_target,
        draws=draws_count,
        predictive_samples=predictive_samples,
        seed=int(dataset["split_seed"]),
    )
    comparison = _write_scores(
        prediction,
        adapter=adapter,
        target=test_target,
        dataset=name,
        family=family_name,
        bins=int(config["calibration_bins"]),
        metric_dir=paths.metrics / name,
    )
    _write_dune_bayes_uncertainty(
        prediction,
        model=model,
        dataset=name,
        family=family_name,
        metric_dir=paths.metrics / name,
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
        namlss.fit(train_data, smoke=smoke)
        namlss_prediction = namlss.predict(
            test_features,
            test_target,
            draws=draws_count,
            predictive_samples=predictive_samples,
            seed=int(dataset["split_seed"]),
        )
        comparison.extend(
            _write_scores(
                namlss_prediction,
                adapter=namlss,
                target=test_target,
                dataset=name,
                family=family_name,
                bins=int(config["calibration_bins"]),
                metric_dir=paths.metrics / name / namlss.name,
            )
        )
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
        lanam.fit(train_data, smoke=smoke)
        lanam_prediction = lanam.predict(
            test_features,
            test_target,
            draws=draws_count,
            predictive_samples=predictive_samples,
            seed=int(dataset["split_seed"]),
        )
        comparison.extend(
            _write_scores(
                lanam_prediction,
                adapter=lanam,
                target=test_target,
                dataset=name,
                family="mean_only_laplace_gaussian",
                bins=int(config["calibration_bins"]),
                metric_dir=paths.metrics / name / lanam.name,
            )
        )
    plain: BenchmarkAdapter = PlainMLPAdapter(
        hidden_dims=hidden_dims,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
    )
    plain.fit(train_data, smoke=smoke)
    plain_prediction = plain.predict(
        test_features,
        test_target,
        draws=draws_count,
        predictive_samples=predictive_samples,
        seed=int(dataset["split_seed"]),
    )
    comparison.extend(
        _write_scores(
            plain_prediction,
            adapter=plain,
            target=test_target,
            dataset=name,
            family="gaussian_residual",
            bins=int(config["calibration_bins"]),
            metric_dir=paths.metrics / name / plain.name,
        )
    )
    ensemble: BenchmarkAdapter = DeepEnsembleAdapter(
        members=int(config["baselines"]["deep_ensemble_members"]),
        hidden_dims=hidden_dims,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
    )
    ensemble.fit(train_data, smoke=smoke)
    ensemble_prediction = ensemble.predict(
        test_features,
        test_target,
        draws=draws_count,
        predictive_samples=predictive_samples,
        seed=int(dataset["split_seed"]),
    )
    comparison.extend(
        _write_scores(
            ensemble_prediction,
            adapter=ensemble,
            target=test_target,
            dataset=name,
            family="gaussian_residual_mixture",
            bins=int(config["calibration_bins"]),
            metric_dir=paths.metrics / name / ensemble.name,
        )
    )
    return comparison


def _linked_parameter_draws(
    raw_draws: torch.Tensor, family: str
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
        return {
            "loc": raw_draws[..., 0],
            "scale": F.softplus(raw_draws[..., 1]) + EPS,
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
) -> None:
    """Write the Bayesian-only parameter bands and variance decomposition."""
    raw_draws = prediction.parameter_draws
    if raw_draws is None:
        raise RuntimeError("dune-bayes prediction did not expose parameter draws.")

    band_rows: list[dict[str, object]] = []
    for parameter, draws in _linked_parameter_draws(raw_draws, family).items():
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
                "aleatoric": float(split.aleatoric[observation]),
                "epistemic": float(split.epistemic[observation]),
                "total": float(split.total[observation]),
            }
            for observation in range(int(split.total.shape[0]))
        ],
    )


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

    summary = {"dataset": dataset, "family": family, "n_test": len(target)}
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
