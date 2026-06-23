"""HMC agreement experiment boundary tests (ADR-0008, GitHub #101)."""

from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.hmc


def test_torch_and_jax_log_joints_agree_at_fixed_parameters() -> None:
    """The independently evaluated model definitions have the same log joint."""
    model = importlib.import_module("experiments.hmc_agreement.model")
    data = {
        "x1": np.array([-1.0, -0.2, 0.4, 1.1], dtype=np.float64),
        "x2": np.array([0.7, -0.5, 0.3, -1.2], dtype=np.float64),
        "y": np.array([-0.8, 0.1, 0.9, 1.4], dtype=np.float64),
    }
    parameters = {
        "x1_weight": np.array([0.75, -0.15], dtype=np.float64),
        "x2_weight": np.array([-0.35, 0.2], dtype=np.float64),
        "intercept": np.array([0.1, -0.4], dtype=np.float64),
    }

    torch_value = model.torch_log_joint(data, parameters)
    jax_value = model.jax_log_joint(data, parameters)

    # Both routes use float64; 1e-10 isolates formula disagreement from the
    # last-bit differences between torch and JAX's Normal implementations.
    assert torch_value == pytest.approx(jax_value, abs=1e-10)
    assert torch.isfinite(torch.tensor(torch_value))


def test_smoke_cli_writes_agreement_artifacts(tmp_path: Path) -> None:
    """The reduced public run records every VI-versus-NUTS evidence class."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "experiment: hmc_agreement",
                "seed: 101",
                "family: normal",
                "architecture:",
                "  hidden_dims: []",
                "  prior_scale: 1.0",
                "data:",
                "  n: 32",
                "  noise_seed: 1101",
                "  feature_correlation: 0.85",
                "  truth:",
                "    x1_weight: [1.0, -0.25]",
                "    x2_weight: [-0.6, 0.2]",
                "    intercept: [0.3, -0.5]",
                "training:",
                "  epochs: 40",
                "  lr: 0.02",
                "  warmup_epochs: 10",
                "nuts:",
                "  warmup: 60",
                "  samples: 60",
                "  chains: 2",
                "  target_accept_prob: 0.8",
                "comparison:",
                "  credible_mass: 0.9",
                "  grid_points: 21",
                "  vi_draws: 100",
                "artifacts:",
                f"  root: {tmp_path / 'runs'}",
                "  run_name: smoke-test",
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "experiments/hmc_agreement/run.py",
            str(config_path),
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    root = tmp_path / "runs" / "hmc_agreement" / "smoke-test"
    required = (
        root / "metrics" / "diagnostics.json",
        root / "metrics" / "parameter_intervals.csv",
        root / "metrics" / "band_width_ratios.csv",
        root / "arrays" / "posterior_bands.npz",
        root / "figures" / "vi_vs_nuts.pdf",
        root / "run.json",
        root / "config.yaml",
    )
    assert all(path.is_file() for path in required)

    diagnostics = json.loads(required[0].read_text(encoding="utf-8"))
    assert diagnostics["chains"] == 2
    assert diagnostics["r_hat_max"] > 0.0
    assert diagnostics["ess_bulk_min"] > 0.0
    run_log = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert run_log["nuts"] == diagnostics

    with (root / "metrics" / "parameter_intervals.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        intervals = list(csv.DictReader(handle))
    assert {row["parameter"] for row in intervals} == {
        "x1_weight[loc]",
        "x1_weight[raw_scale]",
        "x2_weight[loc]",
        "x2_weight[raw_scale]",
        "intercept[loc]",
        "intercept[raw_scale]",
    }

    with (root / "metrics" / "band_width_ratios.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ratios = list(csv.DictReader(handle))
    assert {(row["feature"], row["distribution_parameter"]) for row in ratios} == {
        ("x1", "loc"),
        ("x1", "raw_scale"),
        ("x2", "loc"),
        ("x2", "raw_scale"),
    }


def test_canonical_run_meets_preregistered_agreement_contract() -> None:
    """Canonical NUTS converges and brackets VI bands with matching centers."""
    root = Path("experiments/hmc_agreement/results/canonical")
    diagnostics = json.loads(
        (root / "metrics" / "diagnostics.json").read_text(encoding="utf-8")
    )
    with (root / "metrics" / "band_width_ratios.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ratios = list(csv.DictReader(handle))

    assert diagnostics["chains"] == 4
    assert diagnostics["r_hat_max"] <= 1.01
    assert diagnostics["divergences"] == 0
    assert diagnostics["ess_bulk_min"] >= 400
    for row in ratios:
        assert float(row["median_vi_to_nuts_width_ratio"]) < 1.0
        assert float(row["vi_inside_nuts_fraction"]) >= 0.9
        assert float(row["median_normalized_center_difference"]) <= 0.25
