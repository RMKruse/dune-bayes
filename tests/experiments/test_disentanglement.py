"""Disentanglement experiment boundaries (ADR-0008, GitHub #99)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _smoke_config(tmp_path: Path) -> Path:
    """Copy the public config while keeping scratch artifacts temporary."""
    source = Path("experiments/disentanglement/config.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["artifacts"] = {"root": str(tmp_path), "run_name": "smoke"}
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def test_smoke_writes_disentanglement_evidence(tmp_path: Path) -> None:
    """The public CLI reaches every paper-facing artifact class."""
    config_path = _smoke_config(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/disentanglement/run.py",
            str(config_path),
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run = tmp_path / "disentanglement" / "smoke"
    assert (run / "arrays" / "decomposition.npz").is_file()
    assert (run / "metrics" / "regional_components.json").is_file()
    assert (run / "figures" / "disentanglement.pdf").is_file()


def test_fixed_seed_separates_sparse_epistemic_from_noisy_aleatoric(
    tmp_path: Path,
) -> None:
    """Regional components follow their distinct known-by-construction drivers."""
    config_path = _smoke_config(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "experiments/disentanglement/run.py",
            str(config_path),
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run = tmp_path / "disentanglement" / "smoke"
    metrics = json.loads(
        (run / "metrics" / "regional_components.json").read_text(encoding="utf-8")
    )

    assert metrics["epistemic_sparse"] > metrics["epistemic_dense"]
    assert metrics["aleatoric_dense"] > metrics["aleatoric_sparse"]


def test_run_log_reports_pre_registered_regional_ratios(tmp_path: Path) -> None:
    """The paper claim is directly auditable without reopening raw arrays."""
    config_path = _smoke_config(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "experiments/disentanglement/run.py",
            str(config_path),
            "--smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run = tmp_path / "disentanglement" / "smoke"
    metrics = json.loads(
        (run / "metrics" / "regional_components.json").read_text(encoding="utf-8")
    )

    assert metrics["epistemic_sparse_to_dense_ratio"] > 1.0
    assert metrics["aleatoric_dense_to_sparse_ratio"] > 1.0


def test_canonical_evidence_is_promoted() -> None:
    """The exact paper evidence is reviewable and regenerable from the repo."""
    result = Path("experiments/disentanglement/results/canonical")

    assert (result / "config.yaml").is_file()
    assert (result / "run.json").is_file()
    assert (result / "arrays" / "decomposition.npz").is_file()
    assert (result / "metrics" / "regional_components.json").is_file()
    assert (result / "figures" / "disentanglement.pdf").is_file()


def test_config_and_seed_regenerate_identical_evidence(tmp_path: Path) -> None:
    """The sole input reproduces every scientific artifact byte for byte."""
    config_path = _smoke_config(tmp_path)
    command = [
        sys.executable,
        "experiments/disentanglement/run.py",
        str(config_path),
        "--smoke",
    ]
    run = tmp_path / "disentanglement" / "smoke"
    artifacts = (
        run / "arrays" / "decomposition.npz",
        run / "metrics" / "regional_components.json",
        run / "figures" / "disentanglement.pdf",
    )

    subprocess.run(command, check=True, capture_output=True, text=True)
    first = tuple(path.read_bytes() for path in artifacts)
    # Cross a wall-clock boundary to expose PDF/ZIP timestamp metadata.
    time.sleep(2.1)
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert tuple(path.read_bytes() for path in artifacts) == first


def test_promoted_truth_matches_independent_known_construction() -> None:
    """Raw truth uses the pre-registered noise driver without model leakage."""
    config = yaml.safe_load(
        Path("experiments/disentanglement/config.yaml").read_text(encoding="utf-8")
    )
    arrays = np.load(
        "experiments/disentanglement/results/canonical/arrays/decomposition.npz"
    )
    truth = config["truth"]
    dense_weight = 1.0 / (
        1.0 + np.exp(arrays["grid"] / float(truth["transition_width"]))
    )
    scale = (
        float(truth["quiet_scale"])
        + (float(truth["noisy_scale"]) - float(truth["quiet_scale"])) * dense_weight
    )

    # The artifact is float32; 1e-7 covers two rounded elementary operations.
    np.testing.assert_allclose(arrays["truth_aleatoric"], scale**2, atol=1e-7)
