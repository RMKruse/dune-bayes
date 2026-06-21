"""Johnson's SU showcase experiment boundaries (ADR-0008, GitHub #100)."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _smoke_config(tmp_path: Path) -> Path:
    """Copy the public config while keeping scratch artifacts temporary."""
    source = Path("experiments/jsu_showcase/config.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["artifacts"] = {"root": str(tmp_path), "run_name": "smoke"}
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def test_smoke_writes_effect_ribbon_and_coverage_evidence(tmp_path: Path) -> None:
    """The public CLI reaches every paper-facing artifact class."""
    config_path = _smoke_config(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/jsu_showcase/run.py",
            str(config_path),
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run = tmp_path / "jsu_showcase" / "smoke"
    assert (run / "arrays" / "recovery.npz").is_file()
    assert (run / "figures" / "effect_ribbons.pdf").is_file()
    assert (run / "metrics" / "coverage.csv").is_file()


def test_all_jsu_parameters_report_shape_and_intercept_coverage(
    tmp_path: Path,
) -> None:
    """Coverage doctrine applies to all four effects and absorbed levels."""
    config_path = _smoke_config(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "experiments/jsu_showcase/run.py",
            str(config_path),
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run = tmp_path / "jsu_showcase" / "smoke"

    for filename in ("coverage.csv", "intercept_coverage.csv"):
        with (run / "metrics" / filename).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["parameter"] for row in rows} == {
            "skew",
            "tail",
            "loc",
            "scale",
        }
        assert {float(row["nominal"]) for row in rows} == {0.5, 0.8, 0.9, 0.95}
        assert len(rows) == 16


def test_recovery_centers_truth_and_every_posterior_shape_draw(
    tmp_path: Path,
) -> None:
    """Shape recovery removes additive level non-identifiability."""
    config_path = _smoke_config(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "experiments/jsu_showcase/run.py",
            str(config_path),
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    arrays = np.load(tmp_path / "jsu_showcase" / "smoke" / "arrays" / "recovery.npz")

    # Centering sums float32 grid values; 1e-6 bounds the reduction rounding.
    np.testing.assert_allclose(arrays["truth"].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(arrays["centered_draws"].mean(axis=1), 0.0, atol=1e-6)
    assert arrays["truth"].shape == (48, 4)
    assert arrays["centered_draws"].shape == (32, 48, 4)


def test_truth_independently_varies_jsu_skew_and_linked_scale(
    tmp_path: Path,
) -> None:
    """Known construction is both covariate-skewed and heteroscedastic."""
    config_path = _smoke_config(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "experiments/jsu_showcase/run.py",
            str(config_path),
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    arrays = np.load(tmp_path / "jsu_showcase" / "smoke" / "arrays" / "recovery.npz")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grid = arrays["grid"]

    raw_columns = []
    for specification in config["truth"]["effects"]:
        if specification["kind"] == "linear":
            basis = grid
        else:
            phase = float(specification["frequency"]) * np.pi * grid
            basis = np.sin(phase) if specification["kind"] == "sin" else np.cos(phase)
        raw_columns.append(float(specification["amplitude"]) * basis)
    raw_truth = np.stack(raw_columns, axis=-1)
    raw_truth -= raw_truth.mean(axis=0, keepdims=True)
    raw_parameters = raw_truth + np.asarray(config["truth"]["intercept"])
    linked_scale = np.logaddexp(0.0, raw_parameters[:, 3]) + 1e-6

    # 2e-7 covers NumPy-versus-torch float32 elementary-operation ordering.
    np.testing.assert_allclose(arrays["truth"], raw_truth, atol=2e-7)
    np.testing.assert_allclose(
        arrays["linked_truth"][:, 0], raw_parameters[:, 0], atol=2e-7
    )
    np.testing.assert_allclose(arrays["linked_truth"][:, 3], linked_scale, atol=2e-7)
    assert np.ptp(arrays["linked_truth"][:, 0]) > 1.5
    assert np.ptp(arrays["linked_truth"][:, 3]) > 0.5


def test_canonical_evidence_is_promoted() -> None:
    """The exact paper evidence is reviewable and regenerable from the repo."""
    result = Path("experiments/jsu_showcase/results/canonical")

    assert (result / "config.yaml").is_file()
    assert (result / "run.json").is_file()
    assert (result / "arrays" / "recovery.npz").is_file()
    assert (result / "figures" / "effect_ribbons.pdf").is_file()
    assert (result / "metrics" / "coverage.csv").is_file()
    assert (result / "metrics" / "intercept_coverage.csv").is_file()


def test_config_and_seed_regenerate_identical_evidence(tmp_path: Path) -> None:
    """The sole input reproduces every scientific artifact byte for byte."""
    config_path = _smoke_config(tmp_path)
    command = [
        sys.executable,
        "experiments/jsu_showcase/run.py",
        str(config_path),
        "--smoke",
    ]
    run = tmp_path / "jsu_showcase" / "smoke"
    artifacts = (
        run / "arrays" / "recovery.npz",
        run / "metrics" / "coverage.csv",
        run / "metrics" / "intercept_coverage.csv",
        run / "figures" / "effect_ribbons.pdf",
    )

    subprocess.run(command, check=True, capture_output=True, text=True)
    first = tuple(path.read_bytes() for path in artifacts)
    # Cross a wall-clock boundary to expose PDF/ZIP timestamp metadata.
    time.sleep(2.1)
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert tuple(path.read_bytes() for path in artifacts) == first
