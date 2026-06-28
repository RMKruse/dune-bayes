"""Prior/smoothness screening review helpers (PRD-0004, GitHub #160/#161)."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.publication.prior_sweep_review import (
    main,
    materialize_confirmatory_configs,
    prior_sweep_commands,
    summarize_prior_sweep_outputs,
)

DECISION = Path("experiments/publication/prior-smoothness-calibration-decision.yaml")
EVIDENCE_MANIFEST = Path("experiments/publication/evidence-manifest.yaml")
NORMAL_CANONICAL_CONFIG = Path(
    "experiments/parameter_recovery/results/canonical-normal/config.yaml"
)


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write one long-form metrics table for a fake sweep run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_fake_run(
    root: Path,
    *,
    run_name: str,
    mean_offset: float,
    prior_scale: dict[str, object],
) -> None:
    """Create the public artifacts consumed by the review command."""
    run = root / "parameter_recovery_normal" / run_name
    rows = [
        {
            "parameter": parameter,
            "nominal": nominal,
            "empirical_coverage": nominal - mean_offset,
        }
        for parameter in ("location", "scale")
        for nominal in (0.5, 0.8, 0.9, 0.95)
    ]
    _write_csv(run / "metrics" / "calibration.csv", rows)
    _write_csv(run / "metrics" / "intercept_coverage.csv", rows)
    (run / "metrics" / "prior_scale.json").write_text(
        json.dumps(prior_scale), encoding="utf-8"
    )
    (run / "metrics" / "training.json").write_text(
        json.dumps(
            {"epochs": 800, "final_loss": 1.25, "final_nll": 1.0, "final_kl": 0.25}
        ),
        encoding="utf-8",
    )
    (run / "figures").mkdir(parents=True, exist_ok=True)
    (run / "figures" / "recovery.pdf").write_bytes(b"%PDF-1.4\n")
    (run / "figures" / "calibration.pdf").write_bytes(b"%PDF-1.4\n")
    (run / "run.json").write_text(
        json.dumps({"seed": 9801, "smoke": False}), encoding="utf-8"
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


def test_prior_sweep_review_records_coverage_ranking_and_decision(
    tmp_path: Path,
) -> None:
    """The screening review ranks all candidates without promoting artifacts."""
    config_dir = Path("experiments/parameter_recovery/prior_sweep")
    runs_root = tmp_path / "runs"
    offsets = {
        "prior-sweep-normal-empirical-bayes": 0.20,
        "prior-sweep-normal-fixed-0p3": 0.15,
        "prior-sweep-normal-fixed-1p0": 0.10,
        "prior-sweep-normal-fixed-3p0": 0.05,
        "prior-sweep-normal-hierarchical-ig": 0.30,
    }
    for config_path in sorted(config_dir.glob("normal-*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_name = str(config["artifacts"]["run_name"])
        _write_fake_run(
            runs_root,
            run_name=run_name,
            mean_offset=offsets[run_name],
            prior_scale={"mode": "fixed", "scale": 1.0},
        )

    output_dir = tmp_path / "review"
    main(
        [
            "--config-dir",
            str(config_dir),
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(output_dir),
            "--decision",
            "nominate:prior-sweep-normal-fixed-3p0",
            "--decision-notes",
            "Synthetic review nominates the lowest coverage-error candidate.",
        ]
    )

    summary = json.loads((output_dir / "screening_summary.json").read_text())
    assert summary["baseline_run_name"] == "prior-sweep-normal-fixed-1p0"
    assert summary["decision"]["status"] == "nominate"
    assert summary["decision"]["candidate"] == "prior-sweep-normal-fixed-3p0"
    assert [row["run_name"] for row in summary["ranking"]] == [
        "prior-sweep-normal-fixed-3p0",
        "prior-sweep-normal-fixed-1p0",
        "prior-sweep-normal-fixed-0p3",
        "prior-sweep-normal-empirical-bayes",
        "prior-sweep-normal-hierarchical-ig",
    ]
    assert summary["ranking"][0]["training"]["final_loss"] == 1.25
    assert (output_dir / "screening_summary.csv").is_file()


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


def test_no_promotion_decision_preserves_canonical_publication_evidence() -> None:
    """The reviewed prior/smoothness branch records no promotion for #163."""
    decision = yaml.safe_load(DECISION.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    normal_config = yaml.safe_load(
        NORMAL_CANONICAL_CONFIG.read_text(encoding="utf-8")
    )

    assert decision["status"] == "no_promotion"
    assert decision["reviewed_against"] == (
        "docs/issues/0028-prior-smoothness-calibration-sweep.md"
    )
    assert decision["screening_summary"] == (
        "experiments/parameter_recovery/runs/prior-sweep-review-9801/"
        "screening_summary.json"
    )
    assert decision["follow_up"] == "last-layer richer posterior / low-rank covariance"
    assert decision["unchanged_defaults"]["package_defaults"] is True
    assert decision["unchanged_defaults"]["benchmark_configs"] is True

    normal_evidence = manifest["claims"][1]["evidence"][0]
    assert normal_evidence["path"] == (
        "experiments/parameter_recovery/results/canonical-normal"
    )
    assert normal_config["architecture"]["prior_scale"] == 1.0
    assert decision["canonical_artifact"] == normal_evidence["path"]
