"""Config, seeding, and artifact conventions (ADR-0008, GitHub #97)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dune_bayes import seed_everything


@dataclass(frozen=True)
class ArtifactPaths:
    """Conventional output directories for one experiment run."""

    root: Path
    metrics: Path
    figures: Path
    arrays: Path


Experiment = Callable[[Mapping[str, Any], ArtifactPaths, bool], None]


def run_experiment(
    config_path: str | Path,
    *,
    smoke: bool,
    experiment: Experiment,
) -> ArtifactPaths:
    """Load, deterministically seed, run, and record one experiment.

    Args:
        config_path: YAML file containing the complete experiment definition.
        smoke: Whether the experiment should use its tiny CI workload.
        experiment: Scientific experiment callback invoked after setup.

    Returns:
        The conventional artifact paths populated by the run.
    """
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]), deterministic=True)

    artifact_config = config["artifacts"]
    artifact_root = Path(artifact_config["root"])
    if not artifact_root.is_absolute():
        artifact_root = source.parent / artifact_root
    root = artifact_root / str(config["experiment"]) / str(artifact_config["run_name"])
    paths = ArtifactPaths(
        root=root,
        metrics=root / "metrics",
        figures=root / "figures",
        arrays=root / "arrays",
    )
    for directory in (paths.metrics, paths.figures, paths.arrays):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(source, root / "config.yaml")
    experiment(config, paths, smoke)
    (root / "run.json").write_text(
        json.dumps(
            {
                "experiment": config["experiment"],
                "seed": config["seed"],
                "smoke": smoke,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
