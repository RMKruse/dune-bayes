"""Benchmark publication gate boundaries (ADR-0008, GitHub #130)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small result table fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_full_result(tmp_path: Path) -> Path:
    """Create one promoted benchmark result with complete canonical evidence."""
    result = tmp_path / "canonical-full"
    result.mkdir()
    (result / "run.json").write_text(
        json.dumps({"experiment": "uci_benchmark", "seed": 130, "smoke": False}),
        encoding="utf-8",
    )
    (result / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": "uci_benchmark",
                "datasets": [
                    {"name": "autompg", "family": "normal"},
                    {"name": "bike", "family": "negative_binomial"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_csv(
        result / "metrics" / "comparison.csv",
        [
            {
                "dataset": dataset,
                "family": family,
                "model": model,
                "uncertainty_scope": "predictive",
                "n_test": 20,
                "mean_nll": 1.0,
                "mean_crps": 0.5,
                "calibration_error": 0.02,
            }
            for dataset, family in (
                ("autompg", "normal"),
                ("bike", "negative_binomial"),
            )
            for model in ("dune_bayes", "plain_mlp")
        ],
    )
    for dataset in ("autompg", "bike"):
        for model in ("dune_bayes", "plain_mlp"):
            metric_dir = result / "metrics" / dataset
            if model != "dune_bayes":
                metric_dir = metric_dir / model
            _write_csv(
                metric_dir / "nll.csv",
                [{"dataset": dataset, "model": model, "mean_nll": 1.0}],
            )
            _write_csv(
                metric_dir / "crps.csv",
                [{"dataset": dataset, "model": model, "mean_crps": 0.5}],
            )
            _write_csv(
                metric_dir / "calibration.csv",
                [{"dataset": dataset, "model": model, "calibration_error": 0.02}],
            )
    return result


def _write_manifest(
    tmp_path: Path,
    result: Path,
    *,
    exclusions: list[dict[str, str]] | None = None,
) -> Path:
    """Declare the paper-facing benchmark claim against one result tree."""
    manifest = tmp_path / "benchmark-claims.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "claims": [
                    {
                        "id": "uci-benchmark-panel",
                        "result": str(result),
                        "evidence": "full",
                        "datasets": [
                            {"name": "autompg", "family": "normal"},
                            {"name": "bike", "family": "negative_binomial"},
                        ],
                        "baselines": ["dune_bayes", "plain_mlp"],
                        "metrics": ["nll", "crps", "calibration"],
                        "exclusions": exclusions or [],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def _run_gate(manifest: Path) -> subprocess.CompletedProcess[str]:
    """Run the public benchmark publication gate CLI."""
    return subprocess.run(
        [
            sys.executable,
            "experiments/uci_benchmark/publication_gate.py",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a small result table fixture."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_complete_benchmark_evidence_passes_publication_gate(tmp_path: Path) -> None:
    """Claimed datasets, families, baselines, and metrics match full evidence."""
    result = _write_full_result(tmp_path)
    manifest = _write_manifest(tmp_path, result)

    completed = _run_gate(manifest)

    assert completed.returncode == 0, completed.stderr
    assert "uci-benchmark-panel: ready" in completed.stdout


def test_missing_baseline_row_fails_with_dataset_and_baseline(
    tmp_path: Path,
) -> None:
    """A claimed comparator must appear in the promoted comparison table."""
    result = _write_full_result(tmp_path)
    comparison = result / "metrics" / "comparison.csv"
    rows = [
        row
        for row in _read_csv(comparison)
        if not (row["dataset"] == "bike" and row["model"] == "plain_mlp")
    ]
    _write_csv(comparison, rows)
    manifest = _write_manifest(tmp_path, result)

    completed = _run_gate(manifest)

    assert completed.returncode == 1
    assert "dataset=bike" in completed.stderr
    assert "baseline=plain_mlp" in completed.stderr


def test_missing_metric_file_fails_with_dataset_baseline_and_metric(
    tmp_path: Path,
) -> None:
    """A claimed metric must have its own promoted result table."""
    result = _write_full_result(tmp_path)
    (result / "metrics" / "bike" / "plain_mlp" / "crps.csv").unlink()
    manifest = _write_manifest(tmp_path, result)

    completed = _run_gate(manifest)

    assert completed.returncode == 1
    assert "dataset=bike" in completed.stderr
    assert "baseline=plain_mlp" in completed.stderr
    assert "metric=crps" in completed.stderr


def test_smoke_only_evidence_fails_full_benchmark_claim(tmp_path: Path) -> None:
    """CI tracer output cannot stand in for paper benchmark evidence."""
    result = _write_full_result(tmp_path)
    (result / "run.json").write_text(
        json.dumps({"experiment": "uci_benchmark", "seed": 130, "smoke": True}),
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, result)

    completed = _run_gate(manifest)

    assert completed.returncode == 1
    assert "smoke evidence" in completed.stderr
    assert "full benchmark evidence is required" in completed.stderr


def test_documented_manifest_exclusion_allows_missing_metric(
    tmp_path: Path,
) -> None:
    """An explicit exclusion can account for one unavailable comparator metric."""
    result = _write_full_result(tmp_path)
    (result / "metrics" / "bike" / "plain_mlp" / "crps.csv").unlink()
    manifest = _write_manifest(
        tmp_path,
        result,
        exclusions=[
            {
                "dataset": "bike",
                "baseline": "plain_mlp",
                "metric": "crps",
                "reason": "Fixture intentionally omits CRPS for this comparator.",
            }
        ],
    )

    completed = _run_gate(manifest)

    assert completed.returncode == 0, completed.stderr
    assert "uci-benchmark-panel: ready" in completed.stdout


def test_documented_manifest_exclusion_allows_unavailable_comparator(
    tmp_path: Path,
) -> None:
    """A baseline with every metric excluded does not need a comparison row."""
    result = _write_full_result(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        result,
        exclusions=[
            {
                "dataset": dataset,
                "baseline": "lanam",
                "metric": "*",
                "reason": "LA-NAM optional environment was unavailable.",
            }
            for dataset in ("autompg", "bike")
        ],
    )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["claims"][0]["baselines"].append("lanam")
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = _run_gate(manifest)

    assert completed.returncode == 0, completed.stderr
    assert "uci-benchmark-panel: ready" in completed.stdout


def test_readme_distinguishes_publication_gate_from_smoke_tests() -> None:
    """The benchmark docs explain when to use the publication gate."""
    readme = Path("experiments/uci_benchmark/README.md").read_text(encoding="utf-8")

    assert "publication_gate.py" in readme
    assert "claim/evidence manifest" in readme
    assert "smoke tests" in readme
    assert "not acceptable paper evidence" in readme
