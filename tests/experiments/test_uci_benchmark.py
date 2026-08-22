"""UCI benchmark panel boundaries (ADR-0008, GitHub #102–#103/#175–#177)."""

from __future__ import annotations

import csv
import functools
import http.server
import json
import math
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml

from dune_bayes.data import DataModule
from dune_bayes.families import (
    BaseFamily,
    BetaFamily,
    NegativeBinomialFamily,
    NormalFamily,
)
from dune_bayes.layers import collect_kl
from dune_bayes.metrics import crps, pit
from dune_bayes.utils import EPS
from experiments.uci_benchmark.adapters import (
    DeepEnsembleAdapter,
    DuneBayesAdapter,
    PlainMLPAdapter,
    PredictiveResult,
    ResponseTransform,
    mean_mixture_nll,
    predict_on_original_scale,
)
from experiments.uci_benchmark.run import _build_dune_bayes_model


def _smoke_config(tmp_path: Path, *, seed: int = 102) -> Path:
    """Copy the public panel config with all mutable state in the tempdir."""
    source = Path("experiments/uci_benchmark/config.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["seed"] = seed
    config["data"]["cache_dir"] = str(tmp_path / "cache")
    config["data"]["split_dir"] = str(tmp_path / "splits")
    config["artifacts"] = {"root": str(tmp_path / "runs"), "run_name": f"seed-{seed}"}
    target = tmp_path / f"config-{seed}.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _run_smoke(
    config_path: Path, *, dataset: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the public smoke command."""
    command = [
        sys.executable,
        "experiments/uci_benchmark/run.py",
        str(config_path),
        "--smoke",
    ]
    if dataset is not None:
        command.extend(("--dataset", dataset))
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def test_smoke_persists_one_split_and_reuses_it_across_runs(tmp_path: Path) -> None:
    """Later runs consume the original row partition instead of resplitting."""
    first = _run_smoke(_smoke_config(tmp_path, seed=102))
    assert first.returncode == 0, first.stderr

    split_path = tmp_path / "splits" / "autompg-smoke.npz"
    first_bytes = split_path.read_bytes()
    first_mtime = split_path.stat().st_mtime_ns
    with np.load(split_path) as split:
        train = split["train_indices"]
        fit = split["fit_indices"]
        validation = split["validation_indices"]
        test = split["test_indices"]
        assert set(train).isdisjoint(test)
        assert set(fit).isdisjoint(validation)
        assert sorted(np.concatenate((fit, validation)).tolist()) == train.tolist()
        assert int(split["validation_seed"]) == 1_010_201
        assert sorted(np.concatenate((train, test)).tolist()) == list(
            range(int(split["n_rows"]))
        )

    second = _run_smoke(_smoke_config(tmp_path, seed=999))
    assert second.returncode == 0, second.stderr
    assert split_path.read_bytes() == first_bytes
    assert split_path.stat().st_mtime_ns == first_mtime


def test_plain_mlp_restores_best_validation_checkpoint() -> None:
    """The public adapter fit contract stops on checks and restores its winner."""
    torch.manual_seed(178)
    train = DataModule(
        pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [-1.0, 0.0, 1.0]}),
        response="y",
        numeric_scaling={},
    )
    validation = DataModule(
        pd.DataFrame({"x": [-0.5, 0.5], "y": [-0.5, 0.5]}),
        response="y",
        numeric_scaling={},
    )
    adapter = PlainMLPAdapter(
        NormalFamily(),
        hidden_dims=[2],
        epochs=5,
        learning_rate=0.0,
        randomized_pit=False,
        validation_data=validation,
        check_every=1,
        patience_checks=1,
    )

    adapter.fit(train, smoke=False)

    assert adapter.training_metadata["status"] == "completed"
    assert adapter.training_metadata["epochs_completed"] == 2
    assert adapter.training_metadata["best_epoch"] == 1
    assert adapter.training_metadata["restored_best_checkpoint"] is True
    prediction = adapter.predict(
        validation.features,
        validation.target,
        draws=1,
        predictive_samples=2,
        seed=178,
    )
    assert prediction.parameter_draws is not None
    raw = prediction.parameter_draws.squeeze(0)
    scale = F.softplus(raw[:, 1]) + EPS
    reference_nll = (
        torch.log(scale)
        + 0.5 * math.log(2.0 * math.pi)
        + 0.5 * ((validation.target - raw[:, 0]) / scale).square()
    ).mean()
    assert float(reference_nll) == pytest.approx(
        adapter.training_metadata["best_validation_nll"],
        rel=0.0,
        # One float32 family evaluation separates the persisted Python float.
        abs=1e-6,
    )


def test_plain_mlp_records_non_finite_validation_as_failure() -> None:
    """A non-finite validation score is an explicit failed fit."""
    train = DataModule(
        pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}),
        response="y",
        numeric_scaling={},
    )
    validation = DataModule(
        pd.DataFrame({"x": [0.5], "y": [float("nan")]}),
        response="y",
        numeric_scaling={},
    )
    adapter = PlainMLPAdapter(
        NormalFamily(),
        hidden_dims=[2],
        epochs=5,
        learning_rate=0.0,
        randomized_pit=False,
        validation_data=validation,
        check_every=1,
        patience_checks=1,
    )

    adapter.fit(train, smoke=False)

    assert adapter.training_metadata["status"] == "failed"
    assert adapter.training_metadata["failure"] == "non_finite_validation_nll"
    assert adapter.training_metadata["validation_checks"] == [{"epoch": 1, "nll": None}]


def test_uniform_mixture_validation_nll_matches_hand_computed_reference() -> None:
    """Validation mixtures use the analytic equal-weight density in log-space."""
    component_density = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float64)

    actual = mean_mixture_nll(torch.log(component_density))

    # Both equal-weight mixture densities are exactly 0.5.
    assert actual == pytest.approx(math.log(2.0), rel=0.0, abs=1e-12)


def test_dune_validation_starts_after_warmup_and_reuses_draw_seed() -> None:
    """DUNE checks only post-warm-up and repeats its seeded score trajectory."""
    data = DataModule(
        pd.DataFrame({"x": [-1.0, -0.5, 0.5, 1.0], "y": [-1.0, -0.5, 0.5, 1.0]}),
        response="y",
        numeric_scaling={},
    )
    config = {
        "architecture": {
            "hidden_dims": [2],
            "prior_scale": 1.0,
            "activation": "tanh",
        }
    }

    def fit_once() -> dict[str, object]:
        """Build and fit one identically seeded DUNE validation trajectory."""
        torch.manual_seed(178)
        model = _build_dune_bayes_model(data, NormalFamily(), config)
        adapter = DuneBayesAdapter(
            model,
            NormalFamily(),
            epochs=3,
            learning_rate=0.01,
            warmup_epochs=1,
            randomized_pit=False,
            validation_data=data,
            check_every=1,
            patience_checks=5,
            validation_draws=4,
            validation_seed=901,
        )
        adapter.fit(data, smoke=False)
        return adapter.training_metadata

    first = fit_once()
    second = fit_once()

    assert [item["epoch"] for item in first["validation_checks"]] == [2, 3]
    assert first["validation_checks"] == second["validation_checks"]


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one public metric table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


_PRIMARY_MODELS = (
    "dune_bayes",
    "deterministic_namlss",
    "plain_mlp",
    "deep_ensemble",
)


def _assert_primary_panel(
    comparison: list[dict[str, str]], family: str
) -> list[dict[str, str]]:
    """Assert the shared finite four-model contract and return its rows."""
    primary = [row for row in comparison if row["comparison_role"] == "primary"]
    assert {row["model"] for row in primary} == set(_PRIMARY_MODELS)
    assert {row["family"] for row in primary} == {family}
    assert all(
        np.isfinite(float(row[metric]))
        for row in primary
        for metric in ("mean_nll", "mean_crps", "calibration_error")
    )
    return primary


def test_response_transform_restores_continuous_predictions_and_scores() -> None:
    """Affine predictions and proper scores return to the response's units."""
    transform = ResponseTransform.fit(torch.tensor([2.0, 6.0]), family="normal")

    # Training mean=4 and population std=2; the held-out value never enters fit.
    assert transform.to_model_scale(torch.tensor([10.0])).item() == 3.0
    standard_log_density = -0.5 * 0.5**2 - 0.5 * math.log(2.0 * math.pi)
    reference_pit = 0.5 * (1.0 + math.erf(0.5 / math.sqrt(2.0)))
    standardized = PredictiveResult(
        samples=torch.tensor([[-1.0], [0.0], [1.0]]),
        log_density=torch.tensor([standard_log_density], dtype=torch.float64),
        cdf=torch.tensor([reference_pit], dtype=torch.float64),
    )
    original = transform.to_original_prediction(standardized)

    # All values are exactly representable binary floats under this affine map.
    torch.testing.assert_close(
        original.samples,
        torch.tensor([[2.0], [4.0], [6.0]]),
        rtol=0.0,
        atol=0.0,
    )
    # 1e-12 covers float64 evaluation order against the analytic Normal reference.
    assert float(-original.log_density[0]) == pytest.approx(
        -standard_log_density + math.log(2.0), rel=0.0, abs=1e-12
    )
    # Fair empirical CRPS: 5/3 - (8 / (3*2)) = 1/3.
    assert float(crps(original.samples, torch.tensor([5.0]))[0]) == pytest.approx(
        1.0 / 3.0, rel=0.0, abs=1e-12
    )
    # PIT is invariant under a monotone affine transform; float64 error is <1e-12.
    assert float(original.cdf[0]) == pytest.approx(reference_pit, rel=0.0, abs=1e-12)

    original_reference = PredictiveResult(
        samples=torch.tensor([[2.0], [4.0], [6.0]]),
        log_density=torch.tensor(
            [standard_log_density - math.log(2.0)], dtype=torch.float64
        ),
        cdf=torch.tensor([reference_pit], dtype=torch.float64),
    )
    model_scale = transform.to_model_prediction(original_reference)
    # Binary-exact literals independently check the inverse sample affine map.
    torch.testing.assert_close(
        model_scale.samples,
        torch.tensor([[-1.0], [0.0], [1.0]]),
        rtol=0.0,
        atol=0.0,
    )
    # 1e-12 covers float64 cancellation in the analytic +log(scale) Jacobian.
    assert float(model_scale.log_density[0]) == pytest.approx(
        standard_log_density, rel=0.0, abs=1e-12
    )

    loc, scale = transform.to_original_normal_parameters(
        torch.tensor([-1.0, 1.0]), torch.tensor([0.5, 1.5])
    )
    # Binary-exact values isolate the location/stddev affine rule from MC noise.
    torch.testing.assert_close(loc, torch.tensor([2.0, 6.0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(scale, torch.tensor([1.0, 3.0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        transform.to_original_variance(torch.tensor([1.0, 4.0])),
        torch.tensor([4.0, 16.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_common_adapter_boundary_reuses_transform_for_held_out_rows() -> None:
    """Every routed adapter receives model units and returns original units."""

    class RecordingAdapter:
        name = "recording"
        uncertainty_scope = "test"

        def __init__(self) -> None:
            self.target: torch.Tensor | None = None

        def fit(self, train_data: object, *, smoke: bool) -> None:
            del train_data, smoke

        def predict(
            self,
            features: object,
            target: torch.Tensor,
            *,
            draws: int,
            predictive_samples: int,
            seed: int,
        ) -> PredictiveResult:
            del features, draws, predictive_samples, seed
            self.target = target
            return PredictiveResult(
                samples=target.unsqueeze(0),
                log_density=torch.zeros_like(target, dtype=torch.float64),
                cdf=torch.full_like(target, 0.5, dtype=torch.float64),
            )

    adapter = RecordingAdapter()
    transform = ResponseTransform.fit(torch.tensor([2.0, 6.0]), family="normal")
    prediction = predict_on_original_scale(
        adapter,
        transform,
        {},
        torch.tensor([10.0]),
        draws=3,
        predictive_samples=4,
        seed=5,
    )

    assert adapter.target is not None
    # Mean=4/std=2 gives held-out z=3 and maps the returned sample back to 10.
    torch.testing.assert_close(adapter.target, torch.tensor([3.0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        prediction.samples, torch.tensor([[10.0]]), rtol=0.0, atol=0.0
    )
    # The model-scale zero log-density acquires exactly the -log(2) Jacobian.
    assert float(prediction.log_density[0]) == pytest.approx(
        -math.log(2.0), rel=0.0, abs=1e-12
    )


def test_smoke_persists_train_only_response_standardization(tmp_path: Path) -> None:
    """The public smoke path records the transform fitted on training rows."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    cached = np.genfromtxt(
        tmp_path / "cache" / "autompg-smoke.csv",
        delimiter=",",
        names=True,
    )
    with np.load(tmp_path / "splits" / "autompg-smoke.npz") as split:
        train_response = cached["mpg"][split["train_indices"]].astype(np.float32)
    metadata = json.loads(
        (
            tmp_path
            / "runs"
            / "uci_benchmark"
            / "seed-102"
            / "metrics"
            / "autompg"
            / "response_transform.json"
        ).read_text(encoding="utf-8")
    )

    assert metadata["method"] == "standard"
    assert metadata["fit_partition"] == "train"
    assert metadata["n_fit"] == len(train_response)
    # EPS is wider than one float32 reduction ULP for this smoke fixture.
    assert metadata["loc"] == pytest.approx(
        float(train_response.mean()), rel=0.0, abs=EPS
    )
    assert metadata["scale"] == pytest.approx(
        float(train_response.std()), rel=0.0, abs=EPS
    )


@pytest.mark.parametrize("family", ["beta", "negative_binomial"])
def test_native_support_responses_remain_untransformed(family: str) -> None:
    """Bounded and count likelihoods keep their native response support."""
    target = torch.tensor([0.25, 0.75])
    transform = ResponseTransform.fit(target, family=family)

    assert transform.method == "identity"
    # Identity has no arithmetic, so bit equality is the contract.
    torch.testing.assert_close(
        transform.to_model_scale(target), target, rtol=0.0, atol=0.0
    )


def test_constant_response_transform_is_finite_and_round_trips() -> None:
    """The EPS scale floor keeps a degenerate training response usable."""
    transform = ResponseTransform.fit(torch.full((4,), 7.0), family="normal")
    original = torch.tensor([7.0, 7.000001])
    model_scale = transform.to_model_scale(original)
    restored = transform.to_original_prediction(
        PredictiveResult(
            samples=model_scale.unsqueeze(0),
            log_density=torch.zeros(2, dtype=torch.float64),
            cdf=torch.full((2,), 0.5, dtype=torch.float64),
        )
    )

    assert transform.scale == EPS
    assert bool(torch.isfinite(model_scale).all())
    assert bool(torch.isfinite(restored.log_density).all())
    # EPS covers one float32 subtract/divide/multiply/add round-trip near 7.
    torch.testing.assert_close(restored.samples[0], original, rtol=0.0, atol=EPS)


def test_smoke_writes_nll_crps_and_pit_calibration_tables(tmp_path: Path) -> None:
    """One command scores held-out data through every headline metric."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    metrics = tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics"
    nll = _read_rows(metrics / "autompg" / "nll.csv")
    crps = _read_rows(metrics / "autompg" / "crps.csv")
    calibration = _read_rows(metrics / "autompg" / "calibration.csv")

    assert len(nll) == len(crps) == 1
    assert nll[0]["dataset"] == crps[0]["dataset"] == "autompg"
    assert np.isfinite(float(nll[0]["mean_nll"]))
    assert np.isfinite(float(crps[0]["mean_crps"]))
    assert float(crps[0]["mean_crps"]) >= 0.0
    assert len(calibration) == 10
    assert sum(int(row["count"]) for row in calibration) == int(nll[0]["n_test"])
    assert np.isclose(sum(float(row["fraction"]) for row in calibration), 1.0)


def test_dune_bayes_is_scored_through_the_common_comparison_table(
    tmp_path: Path,
) -> None:
    """The package model obeys the same held-out prediction contract as baselines."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    comparison = _read_rows(
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "comparison.csv"
    )
    dune = next(row for row in comparison if row["model"] == "dune_bayes")
    assert dune["dataset"] == "autompg"
    assert np.isfinite(float(dune["mean_nll"]))
    assert np.isfinite(float(dune["mean_crps"]))
    assert np.isfinite(float(dune["calibration_error"]))


def test_smoke_writes_dune_bayes_parameter_bands_and_variance_split(
    tmp_path: Path,
) -> None:
    """The Bayesian model emits uncertainty artifacts deterministic baselines lack."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    metric_dir = tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics"
    comparison = _read_rows(metric_dir / "comparison.csv")
    n_test = int(
        next(row for row in comparison if row["model"] == "dune_bayes")["n_test"]
    )
    bands = _read_rows(metric_dir / "autompg" / "parameter_bands.csv")
    variance = _read_rows(metric_dir / "autompg" / "variance_split.csv")

    assert len(bands) == 2 * n_test
    assert {row["model"] for row in bands} == {"dune_bayes"}
    assert {row["parameter"] for row in bands} == {"loc", "scale"}
    assert all(
        float(row["q05"]) <= float(row["q50"]) <= float(row["q95"]) for row in bands
    )
    assert len(variance) == n_test
    assert {row["model"] for row in variance} == {"dune_bayes"}
    for row in variance:
        aleatoric = float(row["aleatoric"])
        epistemic = float(row["epistemic"])
        total = float(row["total"])
        assert aleatoric >= 0.0
        assert epistemic >= 0.0
        assert total == pytest.approx(aleatoric + epistemic)


def test_plain_mlp_is_scored_on_the_same_held_out_observations(
    tmp_path: Path,
) -> None:
    """The global distributional MLP and supplemental Gaussian share the split."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    comparison = _read_rows(
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "comparison.csv"
    )
    rows = {row["model"]: row for row in comparison}
    plain = rows["plain_mlp"]
    assert plain["dataset"] == rows["dune_bayes"]["dataset"] == "autompg"
    assert plain["n_test"] == rows["dune_bayes"]["n_test"]
    assert np.isfinite(float(plain["mean_nll"]))
    assert np.isfinite(float(plain["mean_crps"]))
    assert np.isfinite(float(plain["calibration_error"]))
    supplemental = rows["mean_only_gaussian"]
    assert supplemental["comparison_role"] == "supplemental"
    assert supplemental["family"] == "gaussian_residual"


def test_deep_ensemble_completes_the_built_in_sanity_panel(
    tmp_path: Path,
) -> None:
    """Independent MLP members contribute beside the built-in baselines."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    comparison = _read_rows(
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "comparison.csv"
    )
    primary = _assert_primary_panel(comparison, "normal")
    assert {row["dataset"] for row in primary} == {"autompg"}
    assert len({row["n_test"] for row in primary}) == 1
    ensemble = next(row for row in primary if row["model"] == "deep_ensemble")
    assert np.isfinite(float(ensemble["mean_nll"]))
    assert np.isfinite(float(ensemble["mean_crps"]))
    assert np.isfinite(float(ensemble["calibration_error"]))


def test_bayesnam_style_baseline_is_labeled_and_scored_on_shared_split(
    tmp_path: Path,
) -> None:
    """The degenerate first-party baseline is explicit in the shared table."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    metrics = tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics"
    comparison = _read_rows(metrics / "comparison.csv")
    rows = {row["model"]: row for row in comparison}
    label = "BayesNAM-style (our implementation)"
    baseline = rows[label]
    dune = rows["dune_bayes"]

    assert baseline["dataset"] == dune["dataset"] == "autompg"
    assert baseline["n_test"] == dune["n_test"]
    assert baseline["uncertainty_scope"] == "mean_only_variational_location"
    assert np.isfinite(float(baseline["mean_nll"]))
    assert np.isfinite(float(baseline["mean_crps"]))
    assert np.isfinite(float(baseline["calibration_error"]))
    nll = _read_rows(metrics / "autompg" / label / "nll.csv")
    assert nll[0]["model"] == label


def test_bayesnam_style_outputs_band_contrast_figure(tmp_path: Path) -> None:
    """The promoted comparison includes the mean-only vs per-parameter contrast."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    figure = (
        tmp_path
        / "runs"
        / "uci_benchmark"
        / "seed-102"
        / "figures"
        / "autompg"
        / "bayesnam_style_band_contrast.pdf"
    )
    assert figure.is_file()
    assert figure.stat().st_size > 1_000


def test_configured_nampy_namlss_runner_is_scored_beside_dune_bayes(
    tmp_path: Path,
) -> None:
    """The live NAMLSS comparator enters through the same public metric seam."""
    fake_runner = tmp_path / "fake_nampy_runner.py"
    fake_runner.write_text(
        """
from __future__ import annotations

import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--paper-code-dir", required=True)
parser.add_argument("--family", required=True)
parser.add_argument("--draws", type=int, required=True)
parser.add_argument("--predictive-samples", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--batch-size", type=int, required=True)
args = parser.parse_args()

payload = np.load(args.input)
y = payload["test_target"].astype(float)
rng = np.random.default_rng(args.seed)
samples = np.tile(y, (args.predictive_samples, 1)) + rng.normal(
    scale=1e-6, size=(args.predictive_samples, y.shape[0])
)
np.savez(
    args.output,
    samples=samples,
    log_density=np.zeros(y.shape[0], dtype=float),
    cdf=np.linspace(0.05, 0.95, y.shape[0], dtype=float),
)
""",
        encoding="utf-8",
    )
    python_shim = tmp_path / "python-with-uv"
    python_shim.write_text('#!/bin/sh\nexec uv run python "$@"\n', encoding="utf-8")
    python_shim.chmod(0o755)
    config_path = _smoke_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["baselines"]["namlss"] = {
        "enabled": True,
        "python": str(python_shim),
        "runner": str(fake_runner),
        "paper_code_dir": "namlss-paper-code",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = _run_smoke(config_path)
    assert completed.returncode == 0, completed.stderr

    comparison = _read_rows(
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "comparison.csv"
    )
    rows = {row["model"]: row for row in comparison}
    namlss = rows["nampy_namlss"]
    assert namlss["dataset"] == rows["dune_bayes"]["dataset"] == "autompg"
    assert namlss["n_test"] == rows["dune_bayes"]["n_test"]
    assert np.isfinite(float(namlss["mean_nll"]))
    assert np.isfinite(float(namlss["mean_crps"]))
    assert np.isfinite(float(namlss["calibration_error"]))


def test_configured_lanam_runner_is_scored_as_mean_only_baseline(
    tmp_path: Path,
) -> None:
    """LA-NAM enters through the shared seam and labels its mean-only scope."""
    fake_runner = tmp_path / "fake_lanam_runner.py"
    fake_runner.write_text(
        """
from __future__ import annotations

import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--family", required=True)
parser.add_argument("--draws", type=int, required=True)
parser.add_argument("--predictive-samples", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--batch-size", type=int, required=True)
args = parser.parse_args()

payload = np.load(args.input)
y = payload["test_target"].astype(float)
rng = np.random.default_rng(args.seed)
samples = np.tile(y, (args.predictive_samples, 1)) + rng.normal(
    scale=1e-6, size=(args.predictive_samples, y.shape[0])
)
np.savez(
    args.output,
    samples=samples,
    log_density=np.zeros(y.shape[0], dtype=float),
    cdf=np.linspace(0.05, 0.95, y.shape[0], dtype=float),
)
""",
        encoding="utf-8",
    )
    python_shim = tmp_path / "python-with-uv"
    python_shim.write_text('#!/bin/sh\nexec uv run python "$@"\n', encoding="utf-8")
    python_shim.chmod(0o755)
    config_path = _smoke_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["baselines"]["lanam"] = {
        "enabled": True,
        "python": str(python_shim),
        "runner": str(fake_runner),
        "batch_size": 512,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = _run_smoke(config_path)
    assert completed.returncode == 0, completed.stderr

    comparison = _read_rows(
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "comparison.csv"
    )
    rows = {row["model"]: row for row in comparison}
    lanam = rows["lanam"]
    assert lanam["dataset"] == rows["dune_bayes"]["dataset"] == "autompg"
    assert lanam["n_test"] == rows["dune_bayes"]["n_test"]
    assert lanam["uncertainty_scope"] == "mean_only_laplace_location"
    assert np.isfinite(float(lanam["mean_nll"]))
    assert np.isfinite(float(lanam["mean_crps"]))
    assert np.isfinite(float(lanam["calibration_error"]))


def test_configured_bamlss_fixture_is_scored_on_shared_split(
    tmp_path: Path,
) -> None:
    """The committed BAMLSS fixture enters through the shared scorer."""
    fixture_dir = Path("experiments/uci_benchmark/fixtures/bamlss").resolve()
    provenance = json.loads(
        (fixture_dir / "autompg" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["script_version"] == "issue-0175-response-standardization-v1"
    assert provenance["response_transform"]["method"] == "standard"
    assert provenance["response_transform"]["fit_partition"] == "train"

    config_path = _smoke_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["baselines"]["bamlss_reference"] = {
        "enabled": True,
        "fixture_dir": str(fixture_dir),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = _run_smoke(config_path)
    assert completed.returncode == 0, completed.stderr

    metrics = tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics"
    rows_by_model = {
        row["model"]: row for row in _read_rows(metrics / "comparison.csv")
    }
    bamlss = rows_by_model["bamlss_reference"]
    assert bamlss["dataset"] == rows_by_model["dune_bayes"]["dataset"] == "autompg"
    assert (
        bamlss["n_test"]
        == rows_by_model["dune_bayes"]["n_test"]
        == str(provenance["n_test"])
    )
    assert bamlss["uncertainty_scope"] == "distributional_bamlss_fixture"
    assert np.isfinite(float(bamlss["mean_nll"]))
    assert np.isfinite(float(bamlss["mean_crps"]))
    assert np.isfinite(float(bamlss["calibration_error"]))
    nll = _read_rows(metrics / "autompg" / "bamlss_reference" / "nll.csv")
    assert nll[0]["model"] == "bamlss_reference"


def test_bamlss_reference_route_is_documented_with_seeded_r_script() -> None:
    """The HITL BAMLSS fixture producer is committed but disabled by default."""
    script = Path("experiments/uci_benchmark/bamlss/run.R").read_text(encoding="utf-8")
    readme = Path("experiments/uci_benchmark/README.md").read_text(encoding="utf-8")
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )

    assert "Script version: issue-0175-response-standardization-v1" in script
    assert "R version pinned for fixture generation:" in script
    assert "bamlss package version pinned for fixture generation:" in script
    assert "set.seed" in script
    assert "provenance.json" in script
    assert "predictions.csv" in script
    assert "sample_0001" in script
    assert "response_transform" in script
    assert 'reticulate::import("numpy"' in script
    assert config["baselines"]["bamlss_reference"]["enabled"] is False
    assert config["baselines"]["bamlss_reference"]["fixture_dir"] == "fixtures/bamlss"
    assert "Rscript experiments/uci_benchmark/bamlss/run.R" in readme
    assert "bamlss_reference" in readme
    assert "provenance.json" in readme
    assert "HITL" in readme


def test_readme_documents_the_common_predictive_adapter() -> None:
    """Future baselines can implement the scoring seam without reading its code."""
    readme = Path("experiments/uci_benchmark/README.md").read_text(encoding="utf-8")

    assert "fit(train_data, *, smoke)" in readme
    assert "predict(features, target" in readme
    assert "samples" in readme
    assert "log_density" in readme
    assert "cdf" in readme
    assert "plain_mlp" in readme
    assert "deep_ensemble" in readme
    assert "BayesNAM-style (our implementation)" in readme
    assert "location parameter" in readme
    assert "bayesnam_style_band_contrast.pdf" in readme
    assert "nampy_namlss" in readme
    assert "separate TensorFlow" in readme
    assert "namlss-paper-code" in readme


def test_lanam_route_is_documented_and_pinned_to_mit_git_dependency() -> None:
    """The HITL license decision selected the pinned dependency route."""
    readme = Path("experiments/uci_benchmark/README.md").read_text(encoding="utf-8")
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    lanam_requirements = Path(
        "experiments/uci_benchmark/requirements-lanam.txt"
    ).read_text(encoding="utf-8")

    assert "MIT" in readme
    assert "fortuinlab/LA-NAM" in readme
    assert "d6748ebcb1dd5b5c15ca3120c4dcc19667ead111" in readme
    assert config["baselines"]["lanam"]["enabled"] is False
    assert config["baselines"]["lanam"]["runner"] == "lanam_runner.py"
    assert "lanam" not in project["optional-dependencies"]
    assert "requirements-lanam.txt" in readme
    assert "laplace-skorch @ git+https://github.com/fortuinlab/LA-NAM.git" in (
        lanam_requirements
    )
    assert "@d6748ebcb1dd5b5c15ca3120c4dcc19667ead111" in lanam_requirements
    assert "curvlinops-for-pytorch>=2.0,<3" in lanam_requirements
    assert "scikit-learn>=1.5,<1.6" in lanam_requirements


def test_nampy_runner_help_is_available_without_tensorflow_import() -> None:
    """The external runner keeps TensorFlow imports behind execution, not import."""
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/uci_benchmark/nampy_namlss_runner.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--paper-code-dir" in completed.stdout
    assert "--family" in completed.stdout


def test_lanam_runner_help_is_available_without_optional_dependency() -> None:
    """The LA-NAM runner keeps laplace-skorch imports behind execution."""
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/uci_benchmark/lanam_runner.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--family" in completed.stdout
    assert "--predictive-samples" in completed.stdout


def test_results_include_a_promoted_bayesnam_style_smoke_comparison() -> None:
    """The degenerate baseline result is reviewable without rerunning."""
    result_dir = Path("experiments/uci_benchmark/results/smoke")
    rows = _read_rows(result_dir / "metrics" / "comparison.csv")

    assert {row["model"] for row in rows} == {
        "BayesNAM-style (our implementation)",
        "dune_bayes",
        "plain_mlp",
        "deep_ensemble",
    }
    assert all(row["dataset"] == "autompg" for row in rows)
    baseline = next(
        row for row in rows if row["model"] == "BayesNAM-style (our implementation)"
    )
    assert baseline["uncertainty_scope"] == "mean_only_variational_location"
    assert (
        result_dir / "figures" / "autompg" / "bayesnam_style_band_contrast.pdf"
    ).is_file()


def test_results_include_a_promoted_live_nampy_smoke_comparison() -> None:
    """The live comparator table is reviewable without rerunning TensorFlow."""
    result_dir = Path("experiments/uci_benchmark/results/namlss-smoke")
    rows = _read_rows(result_dir / "metrics" / "comparison.csv")
    readme = (result_dir / "README.md").read_text(encoding="utf-8")

    assert {row["model"] for row in rows} == {
        "dune_bayes",
        "nampy_namlss",
        "plain_mlp",
        "deep_ensemble",
    }
    assert all(row["dataset"] == "autompg" for row in rows)
    assert "not a published-number comparison" in readme
    assert "original configs" in readme


def test_panel_config_declares_standard_datasets_and_response_families() -> None:
    """The benchmark population and likelihood choices are reviewable data."""
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )
    families = {item["name"]: item["family"] for item in config["datasets"]}

    assert set(families) == {
        "autompg",
        "bike",
        "concrete",
        "energy",
        "kin8nm",
        "naval",
        "power",
        "protein",
        "wine",
        "yacht",
    }
    assert families["bike"] == "negative_binomial"
    assert families["naval"] == "beta"
    assert set(families.values()) == {"normal", "negative_binomial", "beta"}


def test_panel_config_declares_exactly_four_family_matched_models() -> None:
    """The primary comparison classes and ensemble size are reviewable data."""
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )

    assert config["primary_models"] == list(_PRIMARY_MODELS)
    assert config["baselines"]["deep_ensemble_members"] == 5
    assert config["data"]["validation_fraction"] == 0.2
    assert config["training"]["epochs"] == 500
    assert config["training"]["validation_check_every"] == 10
    assert config["training"]["validation_patience_checks"] == 5


def test_smoke_persists_primary_training_and_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    """Primary models expose the frozen stopping contract in run artifacts."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    metric_root = (
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "autompg"
    )
    paths = {
        "dune_bayes": metric_root / "training.json",
        "deterministic_namlss": metric_root / "deterministic_namlss" / "training.json",
        "plain_mlp": metric_root / "plain_mlp" / "training.json",
        "deep_ensemble": metric_root / "deep_ensemble" / "training.json",
    }
    metadata = {
        model: json.loads(path.read_text(encoding="utf-8"))
        for model, path in paths.items()
    }

    assert all(item["status"] == "completed" for item in metadata.values())
    assert all(item["epochs_ceiling"] == 500 for item in metadata.values())
    assert all(item["check_every"] == 10 for item in metadata.values())
    assert all(item["patience_checks"] == 5 for item in metadata.values())
    assert all(item["restored_best_checkpoint"] is True for item in metadata.values())
    assert metadata["dune_bayes"]["monitor_start_epoch"] == 20
    assert metadata["dune_bayes"]["validation_seed"] == 102
    assert set(metadata["dune_bayes"]["history"]) == {"loss", "nll", "kl"}
    members = metadata["deep_ensemble"]["members"]
    assert len(members) == 5
    assert all(member["restored_best_checkpoint"] is True for member in members)
    assert np.isfinite(metadata["deep_ensemble"]["candidate_selection_validation_nll"])


def test_deterministic_namlss_has_zero_kl_through_the_additive_model_boundary() -> None:
    """Point-estimated shape functions keep the shared additive contract."""
    train_data = DataModule(
        pd.DataFrame({"x1": [-1.0, 0.0, 1.0], "x2": [1.0, 0.0, -1.0], "y": [0, 1, 2]}),
        response="y",
        numeric_scaling={},
    )
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )
    model = _build_dune_bayes_model(
        train_data,
        NormalFamily(validate_args=True),
        config,
        deterministic=True,
    )
    model.eval()

    first = model.predict_params(train_data.features)
    second = model.predict_params(train_data.features)
    # Point-estimated weights and intercept contain no sampling path, so repeat
    # evaluation must be bit-identical rather than merely numerically close.
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert float(collect_kl(model)) == 0.0


@pytest.mark.parametrize(
    ("family_name", "family", "target"),
    [
        ("normal", NormalFamily(validate_args=True), [-1.0, 0.0, 1.0, 2.0]),
        ("beta", BetaFamily(validate_args=True), [0.2, 0.4, 0.6, 0.8]),
        (
            "negative_binomial",
            NegativeBinomialFamily(validate_args=True),
            [0.0, 1.0, 3.0, 6.0],
        ),
    ],
)
def test_global_distributional_mlp_obeys_each_family_contract(
    family_name: str,
    family: BaseFamily,
    target: list[float],
) -> None:
    """Every global MLP output reaches family-linked density, samples, and PIT."""
    train_data = DataModule(
        pd.DataFrame({"x": [-1.0, -0.25, 0.25, 1.0], "y": target}),
        response="y",
        numeric_scaling={},
    )
    transform = ResponseTransform.fit(train_data.target, family=family_name)
    original_target = train_data.target.clone()
    train_data.target = transform.to_model_scale(train_data.target)
    adapter = PlainMLPAdapter(
        family,
        hidden_dims=[4],
        epochs=1,
        learning_rate=0.01,
        randomized_pit=family_name == "negative_binomial",
    )
    adapter.fit(train_data, smoke=True)
    prediction = predict_on_original_scale(
        adapter,
        transform,
        train_data.features,
        original_target,
        draws=2,
        predictive_samples=4,
        seed=12,
    )

    assert prediction.samples.shape == (4, 4)
    assert prediction.parameter_draws is not None
    assert prediction.parameter_draws.shape == (1, 4, family.param_count)
    assert bool(torch.isfinite(prediction.samples).all())
    assert bool(torch.isfinite(prediction.log_density).all())
    assert bool(torch.isfinite(prediction.cdf).all())


def test_count_ensemble_randomizes_pit_after_member_mixture() -> None:
    """The count ensemble randomizes one predictive jump, not member jumps."""
    family = NegativeBinomialFamily(validate_args=True)
    train_data = DataModule(
        pd.DataFrame({"x": [-1.0, -0.25, 0.25, 1.0], "y": [0, 1, 3, 6]}),
        response="y",
        numeric_scaling={},
    )
    adapter = DeepEnsembleAdapter(
        family,
        members=5,
        hidden_dims=[4],
        epochs=1,
        learning_rate=0.01,
        randomized_pit=True,
    )
    adapter.fit(train_data, smoke=True)
    prediction = adapter.predict(
        train_data.features,
        train_data.target,
        draws=2,
        predictive_samples=7,
        seed=12,
    )

    assert prediction.parameter_draws is not None
    reference = pit(
        family,
        prediction.parameter_draws,
        train_data.target,
        randomized=True,
        seed=12,
    )
    # Exact equality is expected because the adapter delegates to this shared
    # PIT contract after assembling the same five parameter vectors.
    torch.testing.assert_close(prediction.cdf, reference, rtol=0.0, atol=0.0)
    assert prediction.samples.shape == (7, 4)


def test_panel_config_declares_bayesnam_style_baseline() -> None:
    """The degenerate first-party baseline is enabled by plain config."""
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )

    baseline = config["baselines"]["bayesnam_style"]
    assert baseline["enabled"] is True
    assert baseline["label"] == "BayesNAM-style (our implementation)"
    assert baseline["family"] == "normal_homoscedastic"
    assert baseline["effect"] == "location_only"


def test_panel_config_pins_every_remote_dataset_source() -> None:
    """Full-panel downloads resolve immutable catalog IDs, never fuzzy names."""
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )

    assert all(
        item["source"]["kind"] in {"uci", "openml"}
        and isinstance(item["source"]["id"], int)
        for item in config["datasets"]
    )
    naval = next(item for item in config["datasets"] if item["name"] == "naval")
    assert naval["response_transform"] == "open_unit_interval"


def test_count_dataset_runs_with_negative_binomial(tmp_path: Path) -> None:
    """Bike counts are fitted and scored on discrete response support."""
    completed = _run_smoke(_smoke_config(tmp_path), dataset="bike")
    assert completed.returncode == 0, completed.stderr

    metric_path = (
        tmp_path
        / "runs"
        / "uci_benchmark"
        / "seed-102"
        / "metrics"
        / "bike"
        / "nll.csv"
    )
    row = _read_rows(metric_path)[0]
    assert row["family"] == "negative_binomial"
    assert np.isfinite(float(row["mean_nll"]))
    metadata = json.loads(
        (metric_path.parent / "response_transform.json").read_text(encoding="utf-8")
    )
    assert metadata["method"] == "identity"
    comparison = _read_rows(metric_path.parents[1] / "comparison.csv")
    _assert_primary_panel(comparison, "negative_binomial")
    assert "mean_only_gaussian" not in {row["model"] for row in comparison}


def test_negative_binomial_candidate_intercepts_match_training_moments() -> None:
    """The selected count candidate starts at the NBI method-of-moments values."""
    train_data = DataModule(
        pd.DataFrame({"x": range(6), "target": [0, 0, 0, 6, 6, 6]}),
        response="target",
        numeric_scaling={},
    )
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )

    model = _build_dune_bayes_model(
        train_data, NegativeBinomialFamily(validate_args=True), config
    )
    linked = F.softplus(model.intercept.loc.detach()) + EPS

    # Population moments are independently hand-computable here: mean=3,
    # variance=9, so NBI dispersion=(9-3)/3**2=2/3. The tolerance covers only
    # the float32 inverse-softplus/link round trip; there is no MC noise.
    torch.testing.assert_close(
        linked,
        torch.tensor([3.0, 2.0 / 3.0]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ([1, 1, 1, 1], [1.0, 2.0 * EPS]),
        ([0, 0, 0, 0], [2.0 * EPS, 2.0 * EPS]),
    ],
)
def test_negative_binomial_candidate_uses_finite_poisson_limit_fallback(
    target: list[int], expected: list[float]
) -> None:
    """Non-positive moment estimates use the smallest finite linked values."""
    train_data = DataModule(
        pd.DataFrame({"x": range(4), "target": target}),
        response="target",
        numeric_scaling={},
    )
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )
    family = NegativeBinomialFamily(validate_args=True)

    model = _build_dune_bayes_model(train_data, family, config)
    linked = F.softplus(model.intercept.loc.detach()) + EPS
    log_prob = family(model.intercept.loc.detach().expand(4, 2)).log_prob(
        train_data.target
    )

    # A finite raw value can only approach the family floor from above. Using
    # EPS as the pre-floor softplus value therefore gives the 2*EPS fallback.
    # rtol covers float32 inverse-link rounding at that scale; atol=1e-12 is
    # small enough that losing the EPS-scale value cannot be masked.
    torch.testing.assert_close(linked, torch.tensor(expected), rtol=1e-6, atol=1e-12)
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("family", [NormalFamily(), BetaFamily()])
def test_moment_initialization_is_not_applied_to_other_families(
    family: BaseFamily,
) -> None:
    """The benchmark keeps the package's zero intercept default outside NBI."""
    train_data = DataModule(
        pd.DataFrame({"x": range(4), "target": [0.2, 0.4, 0.6, 0.8]}),
        response="target",
        numeric_scaling={},
    )
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )

    model = _build_dune_bayes_model(train_data, family, config)

    assert torch.equal(model.intercept.loc, torch.zeros(2))


def test_bounded_dataset_runs_with_beta(tmp_path: Path) -> None:
    """Naval decay coefficients are fitted and scored inside (0, 1)."""
    completed = _run_smoke(_smoke_config(tmp_path), dataset="naval")
    assert completed.returncode == 0, completed.stderr

    metric_path = (
        tmp_path
        / "runs"
        / "uci_benchmark"
        / "seed-102"
        / "metrics"
        / "naval"
        / "nll.csv"
    )
    row = _read_rows(metric_path)[0]
    assert row["family"] == "beta"
    assert np.isfinite(float(row["mean_nll"]))
    metadata = json.loads(
        (metric_path.parent / "response_transform.json").read_text(encoding="utf-8")
    )
    assert metadata["method"] == "identity"
    comparison = _read_rows(metric_path.parents[1] / "comparison.csv")
    _assert_primary_panel(comparison, "beta")
    assert "mean_only_gaussian" not in {row["model"] for row in comparison}


def test_full_run_downloads_once_then_uses_the_cache_offline(tmp_path: Path) -> None:
    """A fetched dataset becomes a durable input rather than a live dependency."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "autompg.csv"
    fixture = Path("experiments/uci_benchmark/fixtures/autompg_smoke.csv").read_text(
        encoding="utf-8"
    )
    # Distinct content proves the non-smoke path fetched this source rather
    # than silently substituting the bundled CI fixture.
    source.write_text(fixture.replace("18.0,8,307", "18.1,8,307"), encoding="utf-8")
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=source_dir
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    config_path = _smoke_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["datasets"][0]["url"] = f"http://127.0.0.1:{server.server_port}/autompg.csv"
    config["training"]["epochs"] = 1
    config["training"]["warmup_epochs"] = 1
    config["draws"] = 8
    config["predictive_samples"] = 16
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        "experiments/uci_benchmark/run.py",
        str(config_path),
        "--dataset",
        "autompg",
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    server.shutdown()
    thread.join()
    assert first.returncode == 0, first.stderr
    cached = tmp_path / "cache" / "autompg.csv"
    first_bytes = cached.read_bytes()
    first_mtime = cached.stat().st_mtime_ns

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert cached.read_bytes() == first_bytes == source.read_bytes()
    assert cached.stat().st_mtime_ns == first_mtime


def test_full_panel_is_opt_in_while_ci_discovers_the_smoke_config() -> None:
    """The expensive panel stays manual; its tiny tracer remains CI-gated."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pytest_config = project["tool"]["pytest"]["ini_options"]
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert pytest_config["addopts"] == "-m 'not hmc and not experiment'"
    assert any(marker.startswith("experiment:") for marker in pytest_config["markers"])
    assert "for config in experiments/*/config*.yaml" in workflow
    assert '"$config" --smoke' in workflow


def test_bounded_experiment_selector_excludes_full_panel() -> None:
    """The audit selector collects smoke checks without full canonical reruns."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "experiment and not full_experiment",
            "tests/experiments/test_uci_benchmark.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "test_full_panel_writes_every_dataset_table" not in completed.stdout


@pytest.mark.experiment
@pytest.mark.full_experiment
def test_full_panel_writes_every_dataset_table(tmp_path: Path) -> None:
    """The opt-in run downloads, fits, and scores the complete declared panel."""
    config_path = _smoke_config(tmp_path)
    completed = subprocess.run(
        [sys.executable, "experiments/uci_benchmark/run.py", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics"
    for dataset in config["datasets"]:
        assert (metrics / dataset["name"] / "nll.csv").is_file()
        assert (metrics / dataset["name"] / "crps.csv").is_file()
        assert (metrics / dataset["name"] / "calibration.csv").is_file()
