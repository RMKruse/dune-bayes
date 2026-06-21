"""Repository-level experiment conventions (ADR-0008, GitHub #97)."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import dune_bayes as db


def test_runs_are_ignored_but_results_are_trackable() -> None:
    """Scratch runs cannot be committed accidentally; promoted results can."""
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "experiments/walking_skeleton/runs/example/metrics.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    promoted = subprocess.run(
        [
            "git",
            "check-ignore",
            "experiments/walking_skeleton/results/metrics.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert ignored.returncode == 0
    assert promoted.returncode == 1


def test_experiment_dependencies_are_isolated_from_runtime() -> None:
    """Experiment backends install by opt-in and never become runtime deps."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    runtime = " ".join(project["dependencies"]).lower()
    experiment = " ".join(project["optional-dependencies"]["experiments"]).lower()

    assert "jax" not in runtime
    assert "numpyro" not in runtime
    assert all(name in experiment for name in ("pyyaml", "jax", "numpyro"))


def test_package_surface_does_not_expose_experiment_backends() -> None:
    """The PyTorch package surface does not expose JAX or NumPyro."""
    assert not hasattr(db, "jax")
    assert not hasattr(db, "numpyro")
    assert set(db.__all__).isdisjoint({"jax", "numpyro"})
