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
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/dune_bayes").rglob("*.py")
    )

    assert not hasattr(db, "jax")
    assert not hasattr(db, "numpyro")
    assert set(db.__all__).isdisjoint({"jax", "numpyro"})
    assert "import jax" not in package_source
    assert "import numpyro" not in package_source


def test_slow_suites_are_opt_in_and_covered_by_the_experiment_job() -> None:
    """Core tests avoid slow experiment tiers while CI runs their contracts."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pytest_config = project["tool"]["pytest"]["ini_options"]
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert pytest_config["addopts"] == "-m 'not hmc and not experiment'"
    assert any(marker.startswith("hmc:") for marker in pytest_config["markers"])
    assert any(marker.startswith("experiment:") for marker in pytest_config["markers"])
    assert "pytest -q -m experiment" in workflow
    assert "pytest -q -m hmc" in workflow
