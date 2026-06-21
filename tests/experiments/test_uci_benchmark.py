"""UCI benchmark panel boundaries (ADR-0008, GitHub #102–#103)."""

from __future__ import annotations

import csv
import functools
import http.server
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml


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
        test = split["test_indices"]
        assert set(train).isdisjoint(test)
        assert sorted(np.concatenate((train, test)).tolist()) == list(
            range(int(split["n_rows"]))
        )

    second = _run_smoke(_smoke_config(tmp_path, seed=999))
    assert second.returncode == 0, second.stderr
    assert split_path.read_bytes() == first_bytes
    assert split_path.stat().st_mtime_ns == first_mtime


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one public metric table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def test_plain_mlp_is_scored_on_the_same_held_out_observations(
    tmp_path: Path,
) -> None:
    """The conventional point predictor has a Gaussian predictive sanity floor."""
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


def test_deep_ensemble_completes_the_three_model_sanity_panel(
    tmp_path: Path,
) -> None:
    """Independent MLP members contribute one mixture-predictive comparison row."""
    completed = _run_smoke(_smoke_config(tmp_path))
    assert completed.returncode == 0, completed.stderr

    comparison = _read_rows(
        tmp_path / "runs" / "uci_benchmark" / "seed-102" / "metrics" / "comparison.csv"
    )
    assert {row["model"] for row in comparison} == {
        "dune_bayes",
        "plain_mlp",
        "deep_ensemble",
    }
    assert {row["dataset"] for row in comparison} == {"autompg"}
    assert len({row["n_test"] for row in comparison}) == 1
    ensemble = next(row for row in comparison if row["model"] == "deep_ensemble")
    assert np.isfinite(float(ensemble["mean_nll"]))
    assert np.isfinite(float(ensemble["mean_crps"]))
    assert np.isfinite(float(ensemble["calibration_error"]))


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


def test_results_include_a_promoted_three_model_smoke_comparison() -> None:
    """The sanity-floor result is reviewable without rerunning the experiment."""
    rows = _read_rows(
        Path("experiments/uci_benchmark/results/smoke/metrics/comparison.csv")
    )

    assert {row["model"] for row in rows} == {
        "dune_bayes",
        "plain_mlp",
        "deep_ensemble",
    }
    assert all(row["dataset"] == "autompg" for row in rows)


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

    assert pytest_config["addopts"] == "-m 'not experiment'"
    assert any(marker.startswith("experiment:") for marker in pytest_config["markers"])
    assert "for config in experiments/*/config*.yaml" in workflow
    assert '"$config" --smoke' in workflow


@pytest.mark.experiment
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
