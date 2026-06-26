"""Bounded reproducibility audit tests (PRD-0003, GitHub #133)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from experiments.publication.audit import (
    AuditCommand,
    audit_publication_reproducibility,
)

README = Path("experiments/README.md")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_canonical_evidence(root: Path) -> Path:
    result = root / "experiments" / "disentanglement" / "results" / "canonical"
    _write_json(
        result / "run.json",
        {"experiment": "disentanglement", "seed": 9901, "smoke": False},
    )
    (result / "config.yaml").write_text("seed: 9901\n", encoding="utf-8")
    _write_json(result / "metrics" / "regional_components.json", {"draws": 500})
    (result / "figures").mkdir(parents=True)
    (result / "figures" / "disentanglement.pdf").write_bytes(b"%PDF-1.4\n")

    manifest = root / "experiments" / "publication" / "evidence-manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "claims": [
                    {
                        "id": "central-disentanglement",
                        "family": "disentanglement",
                        "statement": "Variance decomposition separates uncertainty.",
                        "requires": "full",
                        "evidence": {
                            "path": "experiments/disentanglement/results/canonical",
                            "artifact_class": "simulation",
                            "expected_files": [
                                "config.yaml",
                                "run.json",
                                "metrics/regional_components.json",
                                "figures/disentanglement.pdf",
                            ],
                            "provenance": {
                                "experiment": "disentanglement",
                                "seed": 9901,
                            },
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def _write_benchmark_gate_fixture(root: Path) -> Path:
    result = root / "experiments" / "uci_benchmark" / "results" / "canonical"
    _write_json(result / "run.json", {"experiment": "uci_benchmark", "smoke": False})
    (result / "config.yaml").write_text(
        yaml.safe_dump({"datasets": [{"name": "autompg", "family": "normal"}]}),
        encoding="utf-8",
    )
    (result / "metrics").mkdir(parents=True)
    (result / "metrics" / "comparison.csv").write_text(
        "dataset,model,nll\nautompg,dune_bayes,1.2\n",
        encoding="utf-8",
    )
    (result / "metrics" / "autompg").mkdir()
    (result / "metrics" / "autompg" / "nll.csv").write_text(
        "dataset,model,nll\nautompg,dune_bayes,1.2\n",
        encoding="utf-8",
    )
    manifest = result / "benchmark-claims.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "claims": [
                    {
                        "id": "fixture-benchmark-panel",
                        "result": ".",
                        "evidence": "full",
                        "datasets": [{"name": "autompg", "family": "normal"}],
                        "baselines": ["dune_bayes"],
                        "metrics": ["nll"],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def _write_reproducibility_files(root: Path) -> None:
    (root / "uv.lock").write_text("# fixture lock\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """
[project]
dependencies = ["torch>=2.12"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "pyyaml>=6.0"]
experiments = ["jax>=0.7", "numpyro>=0.19", "pyyaml>=6.0"]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_bounded_audit_writes_machine_and_human_reports(tmp_path: Path) -> None:
    """The public audit path verifies the bounded reviewer-facing artifact."""
    manifest = _write_canonical_evidence(tmp_path)
    benchmark_manifest = _write_benchmark_gate_fixture(tmp_path)
    _write_reproducibility_files(tmp_path)
    output_dir = tmp_path / "audit"

    report = audit_publication_reproducibility(
        root=tmp_path,
        output_dir=output_dir,
        evidence_manifest=manifest,
        benchmark_manifest=benchmark_manifest,
        commands=(
            AuditCommand(
                name="core fixture check",
                category="core_package",
                command=(sys.executable, "-c", "print('core ok')"),
            ),
            AuditCommand(
                name="experiment smoke fixture",
                category="experiment_smoke",
                command=(sys.executable, "-c", "print('smoke ok')"),
            ),
        ),
    )

    assert report.ready is True
    assert report.machine_report == output_dir / "audit-report.json"
    assert report.human_report == output_dir / "audit-report.md"

    payload = json.loads(report.machine_report.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["dependency_readiness"]["ready"] is True
    assert payload["dependency_readiness"]["install_command"] == (
        "uv sync --locked --extra dev --extra experiments"
    )
    assert [check["status"] for check in payload["checks"]] == ["pass", "pass"]
    assert payload["evidence_manifest"]["claim_count"] == 1
    assert payload["benchmark_gate"]["ready"] is True
    assert payload["paper_artifacts"]["ready"] is True
    assert payload["bounded_scope"]["full_canonical_reruns"] == "manual"
    assert "Full canonical experiment reruns are manual" in payload["caveats"]

    human = report.human_report.read_text(encoding="utf-8")
    assert "# Bounded Reproducibility Audit" in human
    assert "core fixture check: pass" in human
    assert "experiment smoke fixture: pass" in human
    assert "Full canonical experiment reruns are manual" in human


def test_experiment_readme_documents_bounded_audit_workflow() -> None:
    """Fresh-clone audit instructions name reports and full-run caveats."""
    readme = README.read_text(encoding="utf-8")

    assert "experiments.publication.audit" in readme
    assert "uv sync --locked --extra dev --extra experiments" in readme
    assert "audit-report.json" in readme
    assert "audit-report.md" in readme
    assert "Full canonical experiment reruns are manual" in readme
