"""Parameter-recovery experiment boundaries (ADR-0008, GitHub #98)."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _smoke_config(tmp_path: Path, source_name: str) -> Path:
    """Copy one public config while redirecting artifacts into a tempdir."""
    source = Path("experiments/parameter_recovery") / source_name
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["artifacts"] = {"root": str(tmp_path), "run_name": "smoke"}
    target = tmp_path / source_name
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def test_normal_smoke_writes_recovery_and_calibration_artifacts(
    tmp_path: Path,
) -> None:
    """The Normal tracer bullet reaches both paper-facing artifact classes."""
    config_path = _smoke_config(tmp_path, "config-normal.yaml")
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/parameter_recovery/run.py",
            str(config_path),
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run = tmp_path / "parameter_recovery_normal" / "smoke"
    assert (run / "figures" / "recovery.pdf").is_file()
    assert (run / "metrics" / "calibration.csv").is_file()


def test_family_panel_reports_every_parameter_and_intercept_separately(
    tmp_path: Path,
) -> None:
    """All core continuous families report the complete calibration grid."""
    expected = {
        "config-normal.yaml": ("parameter_recovery_normal", {"location", "scale"}),
        "config-student-t.yaml": (
            "parameter_recovery_student_t",
            {"location", "scale", "df"},
        ),
        "config-gamma.yaml": (
            "parameter_recovery_gamma",
            {"concentration", "rate"},
        ),
    }
    for source_name, (experiment, parameters) in expected.items():
        config_path = _smoke_config(tmp_path, source_name)
        completed = subprocess.run(
            [
                sys.executable,
                "experiments/parameter_recovery/run.py",
                str(config_path),
                "--smoke",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

        run = tmp_path / experiment / "smoke"
        for filename in ("calibration.csv", "intercept_coverage.csv"):
            with (run / "metrics" / filename).open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert {row["parameter"] for row in rows} == parameters
            assert {float(row["nominal"]) for row in rows} == {0.5, 0.8, 0.9, 0.95}
            assert len(rows) == len(parameters) * 4
        assert (run / "figures" / "calibration.pdf").is_file()


def test_config_and_seed_regenerate_identical_recovery_artifacts(
    tmp_path: Path,
) -> None:
    """The complete scientific output is byte-reproducible from config + seed."""
    config_path = _smoke_config(tmp_path, "config-normal.yaml")
    command = [
        sys.executable,
        "experiments/parameter_recovery/run.py",
        str(config_path),
        "--smoke",
    ]
    run = tmp_path / "parameter_recovery_normal" / "smoke"
    artifacts = (
        run / "metrics" / "calibration.csv",
        run / "metrics" / "intercept_coverage.csv",
        run / "figures" / "recovery.pdf",
        run / "figures" / "calibration.pdf",
        run / "arrays" / "recovery.npz",
    )

    subprocess.run(command, check=True, capture_output=True, text=True)
    first = tuple(path.read_bytes() for path in artifacts)
    # Exercise wall-clock metadata boundaries in PDF and ZIP containers.
    time.sleep(2.1)
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert tuple(path.read_bytes() for path in artifacts) == first


def test_recovery_arrays_center_truth_and_each_posterior_draw(
    tmp_path: Path,
) -> None:
    """Recovery removes level non-identifiability before comparing shapes."""
    config_path = _smoke_config(tmp_path, "config-student-t.yaml")
    subprocess.run(
        [
            sys.executable,
            "experiments/parameter_recovery/run.py",
            str(config_path),
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    arrays = np.load(
        tmp_path / "parameter_recovery_student_t" / "smoke" / "arrays" / "recovery.npz"
    )

    # Centering sums float32 grid values; 1e-6 is the rounding bound for n=32.
    np.testing.assert_allclose(arrays["truth"].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(arrays["centered_draws"].mean(axis=1), 0.0, atol=1e-6)
    assert arrays["intercept_truth"].shape == (3,)
    assert arrays["intercept_draws"].shape == (32, 3)
