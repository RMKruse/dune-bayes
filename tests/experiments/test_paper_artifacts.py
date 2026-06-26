"""Paper artifact builder boundary tests (PRD-0003, GitHub #132)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.publication.artifacts import build_paper_artifacts

README = Path("experiments/README.md")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_promoted_result(
    root: Path,
    *,
    result_path: str = "experiments/disentanglement/results/canonical",
    smoke: bool = False,
) -> None:
    result = root / result_path
    _write_json(
        result / "run.json",
        {
            "experiment": "disentanglement",
            "seed": 9901,
            "smoke": smoke,
            "git_sha": "abc123",
        },
    )
    (result / "config.yaml").write_text("seed: 9901\n", encoding="utf-8")
    _write_json(result / "metrics" / "regional_components.json", {"draws": 500})
    (result / "figures").mkdir()
    (result / "figures" / "disentanglement.pdf").write_bytes(b"%PDF-1.4\n")


def _write_manifest(
    root: Path,
    *,
    evidence_path: str = "experiments/disentanglement/results/canonical",
) -> Path:
    manifest = {
        "version": 1,
        "claims": [
            {
                "id": "central-disentanglement",
                "family": "disentanglement",
                "statement": "Variance decomposition separates uncertainty.",
                "requires": "full",
                "evidence": {
                    "path": evidence_path,
                    "artifact_class": "simulation",
                    "expected_files": [
                        "config.yaml",
                        "run.json",
                        "metrics/regional_components.json",
                        "figures/disentanglement.pdf",
                    ],
                    "provenance": {"experiment": "disentanglement", "seed": 9901},
                },
            }
        ],
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_builder_writes_stable_artifact_paths_and_provenance(
    tmp_path: Path,
) -> None:
    """Promoted evidence becomes stable manuscript-facing files."""
    _write_promoted_result(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    assert report.ready is True
    assert report.failures == ()
    assert report.outputs == (
        output_dir / "figures" / "central-disentanglement__disentanglement.pdf",
        output_dir / "tables" / "evidence-summary.csv",
        output_dir / "provenance.json",
    )
    assert (
        output_dir / "figures" / "central-disentanglement__disentanglement.pdf"
    ).read_bytes() == b"%PDF-1.4\n"
    provenance = json.loads((output_dir / "provenance.json").read_text("utf-8"))
    assert provenance["manifest"] == str(manifest_path)
    assert provenance["inputs"] == [
        {
            "claim_id": "central-disentanglement",
            "artifact_class": "simulation",
            "path": "experiments/disentanglement/results/canonical",
            "run_metadata": {
                "experiment": "disentanglement",
                "seed": 9901,
                "smoke": False,
                "git_sha": "abc123",
            },
        }
    ]


def test_builder_reports_missing_canonical_input(tmp_path: Path) -> None:
    """Missing promoted files fail before paper artifacts are written."""
    _write_promoted_result(tmp_path)
    (
        tmp_path
        / "experiments"
        / "disentanglement"
        / "results"
        / "canonical"
        / "figures"
        / "disentanglement.pdf"
    ).unlink()
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    assert report.ready is False
    assert report.outputs == ()
    assert report.failures == (
        "central-disentanglement: missing artifact file figures/disentanglement.pdf",
    )
    assert not output_dir.exists()


def test_builder_rejects_scratch_artifacts_for_full_outputs(
    tmp_path: Path,
) -> None:
    """Scratch runs cannot be promoted implicitly by the paper builder."""
    scratch_path = "experiments/disentanglement/runs/manual/candidate"
    _write_promoted_result(tmp_path, result_path=scratch_path)
    manifest_path = _write_manifest(tmp_path, evidence_path=scratch_path)
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    assert report.ready is False
    assert report.outputs == ()
    assert report.failures == (
        "central-disentanglement: scratch artifact paths under runs/ cannot be used",
    )
    assert not output_dir.exists()


def test_builder_rejects_smoke_artifacts_for_full_outputs(
    tmp_path: Path,
) -> None:
    """CI-scale smoke evidence cannot back full manuscript artifacts."""
    _write_promoted_result(tmp_path, smoke=True)
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    assert report.ready is False
    assert report.outputs == ()
    assert report.failures == (
        "central-disentanglement: requires full paper evidence but "
        "run.json marks the artifact as smoke",
    )
    assert not output_dir.exists()


def test_builder_writes_declared_metric_tables_to_stable_paths(
    tmp_path: Path,
) -> None:
    """Declared promoted metric tables become stable manuscript table files."""
    _write_promoted_result(tmp_path)
    result = tmp_path / "experiments" / "disentanglement" / "results" / "canonical"
    (result / "metrics" / "comparison.csv").write_text(
        "dataset,model,nll\nfixture,dune_bayes,1.25\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["claims"][0]["evidence"]["expected_files"].append("metrics/comparison.csv")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), "utf-8")
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    stable_table = output_dir / "tables" / "central-disentanglement__comparison.csv"
    assert report.ready is True
    assert stable_table in report.outputs
    assert stable_table.read_text("utf-8") == (
        "dataset,model,nll\nfixture,dune_bayes,1.25\n"
    )


def test_builder_disambiguates_repeated_declared_table_basenames(
    tmp_path: Path,
) -> None:
    """Multiple canonical inputs with the same table name get unique paths."""
    path_a = "experiments/benchmark/results/canonical-a"
    path_b = "experiments/benchmark/results/canonical-b"
    _write_promoted_result(tmp_path, result_path=path_a)
    _write_promoted_result(tmp_path, result_path=path_b)
    (tmp_path / path_a / "metrics" / "comparison.csv").write_text(
        "dataset,model,nll\na,dune_bayes,1.25\n",
        encoding="utf-8",
    )
    (tmp_path / path_b / "metrics" / "comparison.csv").write_text(
        "dataset,model,nll\nb,dune_bayes,1.50\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "claims": [
            {
                "id": "benchmark-comparator-panel",
                "family": "benchmark_comparator",
                "statement": "Fixture benchmark comparison.",
                "requires": "full",
                "evidence": [
                    {
                        "path": path_a,
                        "artifact_class": "real_data_benchmark",
                        "expected_files": [
                            "config.yaml",
                            "run.json",
                            "metrics/comparison.csv",
                        ],
                        "provenance": {
                            "experiment": "disentanglement",
                            "seed": 9901,
                        },
                    },
                    {
                        "path": path_b,
                        "artifact_class": "real_data_benchmark",
                        "expected_files": [
                            "config.yaml",
                            "run.json",
                            "metrics/comparison.csv",
                        ],
                        "provenance": {
                            "experiment": "disentanglement",
                            "seed": 9901,
                        },
                    },
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), "utf-8")
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    table_a = (
        output_dir
        / "tables"
        / "benchmark-comparator-panel__canonical-a__comparison.csv"
    )
    table_b = (
        output_dir
        / "tables"
        / "benchmark-comparator-panel__canonical-b__comparison.csv"
    )
    assert report.ready is True
    assert table_a in report.outputs
    assert table_b in report.outputs
    assert table_a.read_text("utf-8") == "dataset,model,nll\na,dune_bayes,1.25\n"
    assert table_b.read_text("utf-8") == "dataset,model,nll\nb,dune_bayes,1.50\n"


def test_cli_builds_paper_artifacts_from_manifest(tmp_path: Path) -> None:
    """The documented builder command regenerates paper-facing artifacts."""
    _write_promoted_result(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "paper"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.publication.artifacts",
            str(manifest_path),
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Paper artifacts: READY" in completed.stdout
    assert (output_dir / "provenance.json").is_file()


def test_readme_documents_paper_artifact_builder_command() -> None:
    """The experiment docs explain how to regenerate paper artifacts."""
    readme = README.read_text(encoding="utf-8")

    assert "experiments.publication.artifacts" in readme
    assert "--output-dir" in readme
    assert "promoted" in readme
