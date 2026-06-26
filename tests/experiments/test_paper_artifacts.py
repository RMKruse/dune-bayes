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
    experiment: str = "disentanglement",
    seed: int = 9901,
    smoke: bool = False,
) -> None:
    result = root / result_path
    _write_json(
        result / "run.json",
        {
            "experiment": experiment,
            "seed": seed,
            "smoke": smoke,
            "git_sha": "abc123",
        },
    )
    (result / "config.yaml").write_text(f"seed: {seed}\n", encoding="utf-8")
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
        output_dir / "reviewer-evidence-appendix.md",
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


def test_builder_writes_reviewer_appendix_from_promoted_evidence(
    tmp_path: Path,
) -> None:
    """The paper build emits a reviewer-facing claim-to-evidence appendix."""
    _write_promoted_result(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    appendix = output_dir / "reviewer-evidence-appendix.md"
    assert report.ready is True
    assert appendix in report.outputs
    text = appendix.read_text(encoding="utf-8")
    assert "# Reviewer Evidence Appendix" in text
    assert "central-disentanglement" in text
    assert "Variance decomposition separates uncertainty." in text
    assert "simulation" in text
    assert "experiments/disentanglement/results/canonical" in text
    assert "figures/central-disentanglement__disentanglement.pdf" in text


def test_reviewer_appendix_separates_simulation_and_real_data_evidence(
    tmp_path: Path,
) -> None:
    """The appendix distinguishes simulation claims from benchmark claims."""
    simulation_path = "experiments/disentanglement/results/canonical"
    benchmark_path = "experiments/uci_benchmark/results/canonical"
    _write_promoted_result(tmp_path, result_path=simulation_path)
    _write_promoted_result(
        tmp_path,
        result_path=benchmark_path,
        experiment="uci_benchmark",
        seed=102,
    )
    manifest = {
        "version": 1,
        "claims": [
            {
                "id": "central-disentanglement",
                "family": "disentanglement",
                "statement": "Simulation uncertainty decomposition.",
                "requires": "full",
                "evidence": {
                    "path": simulation_path,
                    "artifact_class": "simulation",
                    "expected_files": [
                        "config.yaml",
                        "run.json",
                        "metrics/regional_components.json",
                        "figures/disentanglement.pdf",
                    ],
                    "provenance": {"experiment": "disentanglement", "seed": 9901},
                },
            },
            {
                "id": "benchmark-comparator-panel",
                "family": "benchmark_comparator",
                "statement": "Real-data benchmark comparison.",
                "requires": "full",
                "evidence": {
                    "path": benchmark_path,
                    "artifact_class": "real_data_benchmark",
                    "expected_files": [
                        "config.yaml",
                        "run.json",
                        "metrics/regional_components.json",
                        "figures/disentanglement.pdf",
                    ],
                    "provenance": {"experiment": "uci_benchmark", "seed": 102},
                },
            },
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), "utf-8")
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    assert report.ready is True
    text = (output_dir / "reviewer-evidence-appendix.md").read_text("utf-8")
    assert "## Simulation Evidence" in text
    assert "central-disentanglement" in text
    assert "## Real-Data Benchmark Evidence" in text
    assert "benchmark-comparator-panel" in text


def test_reviewer_appendix_documents_uncertainty_and_methods_conventions(
    tmp_path: Path,
) -> None:
    """The appendix carries ADR-backed reviewer notes, not ad hoc method text."""
    _write_promoted_result(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "paper"

    report = build_paper_artifacts(manifest_path, output_dir=output_dir, root=tmp_path)

    assert report.ready is True
    text = (output_dir / "reviewer-evidence-appendix.md").read_text("utf-8")
    assert "epistemic effect ribbons" in text
    assert "response-level predictive bands" in text
    assert "epistemic-only" in text
    assert "epistemic + aleatoric" in text
    assert "centered effect recovery" in text
    assert "intercept coverage" in text
    assert "coverage is measured" in text
    assert "mean-field VI narrowness" in text
    assert "validation-only NUTS" in text
    assert "does not ship an MCMC backend" in text
    assert "ADR-0001" in text
    assert "ADR-0006" in text
    assert "Johnson's SU" in text
    assert "softplus(x) + EPS" in text


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
    appendix = (output_dir / "reviewer-evidence-appendix.md").read_text("utf-8")
    path_a_block = appendix.split(f"- Promoted evidence: {path_a}", maxsplit=1)[
        1
    ].split(f"- Promoted evidence: {path_b}", maxsplit=1)[0]
    assert "benchmark-comparator-panel__canonical-a__comparison.csv" in path_a_block
    assert "benchmark-comparator-panel__canonical-b__comparison.csv" not in path_a_block


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
    assert "reviewer-evidence-appendix.md" in readme
