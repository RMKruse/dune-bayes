"""Prior/smoothness sweep review notebook helpers (PRD-0004, GitHub #160)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.publication.prior_sweep_review import (
    materialize_confirmatory_configs,
    prior_sweep_commands,
    summarize_prior_sweep_outputs,
)


def test_prior_sweep_review_lists_registered_smoke_and_full_commands() -> None:
    """The notebook review path exposes every pre-registered Normal candidate."""
    commands = prior_sweep_commands(Path("."))

    assert [row["candidate"] for row in commands] == [
        "normal-empirical-bayes",
        "normal-fixed-0p3",
        "normal-fixed-1p0",
        "normal-fixed-3p0",
        "normal-hierarchical-ig",
    ]
    assert all(row["smoke_command"].endswith(" --smoke") for row in commands)
    assert all(not row["full_command"].endswith(" --smoke") for row in commands)
    assert {row["config"] for row in commands} == {
        "experiments/parameter_recovery/prior_sweep/normal-empirical-bayes.yaml",
        "experiments/parameter_recovery/prior_sweep/normal-fixed-0p3.yaml",
        "experiments/parameter_recovery/prior_sweep/normal-fixed-1p0.yaml",
        "experiments/parameter_recovery/prior_sweep/normal-fixed-3p0.yaml",
        "experiments/parameter_recovery/prior_sweep/normal-hierarchical-ig.yaml",
    }


def test_prior_sweep_review_summarizes_available_scratch_outputs(
    tmp_path: Path,
) -> None:
    """Scratch sweep summaries include coverage errors and smoothness diagnostics."""
    assert summarize_prior_sweep_outputs(tmp_path) == ()

    run_dir = (
        tmp_path
        / "experiments"
        / "parameter_recovery"
        / "runs"
        / "parameter_recovery_normal"
        / "prior-sweep-normal-fixed-0p3"
    )
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "architecture": {"prior_scale": 0.3},
                "artifacts": {"run_name": "prior-sweep-normal-fixed-0p3"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics" / "calibration.csv").write_text(
        "\n".join(
            [
                "parameter,nominal,empirical_coverage",
                "location,0.5,0.4",
                "location,0.8,0.7",
                "scale,0.5,0.7",
                "scale,0.8,0.5",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics" / "prior_scale.json").write_text(
        '{"configured_prior_scale": 0.3, "mode": "fixed", "scale": 0.3}',
        encoding="utf-8",
    )

    summary = summarize_prior_sweep_outputs(tmp_path)

    assert len(summary) == 1
    row = summary[0]
    assert row["candidate"] == "prior-sweep-normal-fixed-0p3"
    assert row["configured_prior_scale"] == 0.3
    assert row["prior"] == "fixed"
    # These values are exact in decimal but represented in binary float.
    assert row["mean_abs_coverage_error"] == pytest.approx(0.175)
    assert row["max_abs_coverage_error"] == pytest.approx(0.3)
    assert row["min_empirical_coverage"] == 0.4
    assert row["prior_mode"] == "fixed"
    assert row["scale"] == 0.3
    assert row["scale_median"] is None
    assert row["scale_mean"] is None
    assert row["run_path"] == (
        "experiments/parameter_recovery/runs/"
        "parameter_recovery_normal/prior-sweep-normal-fixed-0p3"
    )


def test_prior_sweep_review_materializes_confirmatory_configs_in_scratch(
    tmp_path: Path,
) -> None:
    """Confirmatory configs compare the baseline with one selected candidate."""
    config_dir = tmp_path / "experiments" / "parameter_recovery" / "prior_sweep"
    config_dir.mkdir(parents=True)
    baseline = {
        "experiment": "parameter_recovery_normal",
        "seed": 9801,
        "family": "normal",
        "architecture": {"prior_scale": 1.0},
        "artifacts": {"root": "../runs", "run_name": "prior-sweep-normal-fixed-1p0"},
    }
    candidate = {
        "experiment": "parameter_recovery_normal",
        "seed": 9801,
        "family": "normal",
        "architecture": {"prior_scale": 3.0},
        "artifacts": {"root": "../runs", "run_name": "prior-sweep-normal-fixed-3p0"},
    }
    (config_dir / "normal-fixed-1p0.yaml").write_text(
        yaml.safe_dump(baseline, sort_keys=False),
        encoding="utf-8",
    )
    (config_dir / "normal-fixed-3p0.yaml").write_text(
        yaml.safe_dump(candidate, sort_keys=False),
        encoding="utf-8",
    )

    assert materialize_confirmatory_configs(tmp_path, None) == ()
    assert not (tmp_path / "experiments" / "parameter_recovery" / "results").exists()

    rows = materialize_confirmatory_configs(
        tmp_path, config_dir / "normal-fixed-3p0.yaml"
    )

    assert [(row["candidate"], row["seed"]) for row in rows] == [
        ("normal-fixed-1p0", 9801),
        ("normal-fixed-1p0", 9811),
        ("normal-fixed-1p0", 9821),
        ("normal-fixed-3p0", 9801),
        ("normal-fixed-3p0", 9811),
        ("normal-fixed-3p0", 9821),
    ]
    for row in rows:
        target = tmp_path / row["config"]
        assert target.is_file()
        config = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert config["seed"] == row["seed"]
        assert config["artifacts"]["root"] == "../runs"
        assert config["artifacts"]["run_name"] == (
            f"confirmatory-{row['candidate']}-seed-{row['seed']}"
        )
        assert " --smoke" in row["smoke_command"]
        assert not row["full_command"].endswith(" --smoke")
        assert Path(row["config"]).parts[:3] == (
            "experiments",
            "parameter_recovery",
            "runs",
        )
    assert not (tmp_path / "experiments" / "parameter_recovery" / "results").exists()


def test_paper_results_explorer_executes_top_to_bottom_without_running_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The notebook review surface runs without opting into experiment launches."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook_path = Path("experiments/publication/paper_results_explorer.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__paper_results_explorer_test__"}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
            category=UserWarning,
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)

    assert namespace["RUN_LOCAL_SWEEP"] is False
    assert len(namespace["prior_sweep_commands"]) == 5
    assert namespace["confirmatory_commands"].empty


def test_paper_results_explorer_keeps_simulation_and_benchmark_settings_distinct() -> (
    None
):
    """Notebook guidance keeps calibration tuning separate from defaults."""
    notebook = json.loads(
        Path("experiments/publication/paper_results_explorer.ipynb").read_text(
            encoding="utf-8"
        )
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "Calibration-improving simulation settings" in markdown
    assert "package default or benchmark configuration" in markdown
