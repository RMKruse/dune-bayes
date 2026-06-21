"""Walking-skeleton experiment boundary tests (ADR-0008, GitHub #97)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def _write_config(tmp_path: Path, run_name: str) -> Path:
    """Write a complete config whose artifacts stay inside the test tempdir."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "experiment: walking_skeleton",
                "seed: 97",
                "family: normal",
                "architecture:",
                "  hidden_dims: [8]",
                "draws: 32",
                "data:",
                "  n: 40",
                "training:",
                "  epochs: 3",
                "artifacts:",
                f"  root: {tmp_path / 'runs'}",
                f"  run_name: {run_name}",
            )
        ),
        encoding="utf-8",
    )
    return config_path


def _smoke_command(config_path: Path) -> list[str]:
    """Build the public smoke CLI command."""
    return [
        sys.executable,
        "experiments/walking_skeleton/run.py",
        str(config_path),
        "--smoke",
    ]


def test_smoke_run_writes_conventional_artifacts(tmp_path: Path) -> None:
    """The public experiment CLI writes every required artifact class."""
    run_root = tmp_path / "runs"
    config_path = _write_config(tmp_path, "test-run")
    completed = subprocess.run(
        _smoke_command(config_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run_dir = run_root / "walking_skeleton" / "test-run"
    assert (run_dir / "metrics" / "metrics.json").is_file()
    assert (run_dir / "figures" / "sample_mean.pdf").is_file()
    assert (run_dir / "arrays" / "samples.npz").is_file()
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "config.yaml").is_file()
    metrics = json.loads((run_dir / "metrics" / "metrics.json").read_text())
    assert {key: metrics[key] for key in ("draws", "n", "epochs")} == {
        "draws": 8,
        "n": 16,
        "epochs": 1,
    }


def test_config_and_seed_regenerate_identical_artifacts(tmp_path: Path) -> None:
    """A rerun from the same sole input reproduces every scientific artifact."""
    config_path = _write_config(tmp_path, "reproducible")
    command = _smoke_command(config_path)
    run_dir = tmp_path / "runs" / "walking_skeleton" / "reproducible"
    artifacts = (
        run_dir / "metrics" / "metrics.json",
        run_dir / "arrays" / "samples.npz",
        run_dir / "figures" / "sample_mean.pdf",
    )

    subprocess.run(command, check=True, capture_output=True, text=True)
    first = tuple(path.read_bytes() for path in artifacts)
    # PDF metadata records wall-clock time unless the experiment removes it.
    time.sleep(1.1)
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert tuple(path.read_bytes() for path in artifacts) == first
