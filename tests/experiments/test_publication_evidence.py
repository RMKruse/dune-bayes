"""Publication evidence manifest boundary tests (PRD-0003, GitHub #129)."""

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import yaml

from experiments.publication.evidence import validate_evidence_manifest

MANIFEST = Path("experiments/publication/evidence-manifest.yaml")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifact(
    root: Path,
    *,
    smoke: bool = False,
    experiment: str = "disentanglement",
    result_name: str = "canonical",
) -> None:
    run = root / "experiments" / experiment / "results" / result_name
    _write_json(
        run / "run.json",
        {"experiment": experiment, "seed": 9901, "smoke": smoke},
    )
    (run / "config.yaml").write_text("seed: 9901\n", encoding="utf-8")
    (run / "arrays").mkdir()
    (run / "arrays" / "decomposition.npz").write_bytes(b"fixture array")
    _write_json(run / "metrics" / "regional_components.json", {"draws": 500})
    (run / "figures").mkdir()
    (run / "figures" / "disentanglement.pdf").write_bytes(b"%PDF-1.4\n")


def _write_manifest(root: Path) -> Path:
    path = root / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(_manifest_payload(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _manifest_payload() -> dict[str, object]:
    """Return a complete one-claim publication manifest fixture."""
    return {
        "version": 1,
        "claims": [
            {
                "id": "central-disentanglement",
                "family": "disentanglement",
                "statement": (
                    "Variance decomposition separates epistemic and aleatoric "
                    "components."
                ),
                "requires": "full",
                "evidence": {
                    "path": "experiments/disentanglement/results/canonical",
                    "artifact_class": "simulation",
                    "expected_files": [
                        "config.yaml",
                        "run.json",
                        "arrays/decomposition.npz",
                        "metrics/regional_components.json",
                        "figures/disentanglement.pdf",
                    ],
                    "provenance": {"experiment": "disentanglement", "seed": 9901},
                },
            }
        ],
    }


def test_present_full_artifact_reports_publication_ready(tmp_path: Path) -> None:
    """A complete promoted artifact satisfies its paper claim."""
    _write_artifact(tmp_path)
    manifest_path = _write_manifest(tmp_path)

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is True
    assert report.failures == ()
    assert report.claim_count == 1


def test_missing_artifact_reports_claim_and_file(tmp_path: Path) -> None:
    """A missing promoted file is actionable from the readiness report."""
    _write_artifact(tmp_path)
    missing = (
        tmp_path
        / "experiments"
        / "disentanglement"
        / "results"
        / "canonical"
        / "figures"
        / "disentanglement.pdf"
    )
    missing.unlink()
    manifest_path = _write_manifest(tmp_path)

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: missing artifact file figures/disentanglement.pdf",
    )


def test_malformed_run_metadata_is_reported_without_traceback(
    tmp_path: Path,
) -> None:
    """Broken promoted metadata fails readiness with a file-specific message."""
    _write_artifact(tmp_path)
    run_json = (
        tmp_path
        / "experiments"
        / "disentanglement"
        / "results"
        / "canonical"
        / "run.json"
    )
    run_json.write_text("{not json", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path)

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: malformed JSON metadata run.json",
    )


def test_smoke_artifact_cannot_satisfy_full_claim(tmp_path: Path) -> None:
    """CI-scale smoke output cannot back a full paper-evidence claim."""
    _write_artifact(tmp_path, smoke=True)
    manifest_path = _write_manifest(tmp_path)

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: requires full paper evidence but "
        "run.json marks the artifact as smoke",
    )


def test_malformed_manifest_reports_missing_artifact_class(
    tmp_path: Path,
) -> None:
    """Claim metadata must name the expected artifact class."""
    _write_artifact(tmp_path)
    manifest = _manifest_payload()
    claim = manifest["claims"][0]
    claim["evidence"].pop("artifact_class")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: manifest evidence.artifact_class is required",
    )


def test_claim_can_be_backed_by_multiple_evidence_locations(
    tmp_path: Path,
) -> None:
    """One paper claim may cite several promoted canonical artifacts."""
    _write_artifact(tmp_path, experiment="parameter_recovery")
    _write_artifact(tmp_path, experiment="jsu_showcase")
    manifest = _manifest_payload()
    claim = manifest["claims"][0]
    claim["id"] = "epistemic-band-family"
    claim["family"] = "epistemic_bands"
    claim["evidence"] = [
        {
            "path": "experiments/parameter_recovery/results/canonical",
            "artifact_class": "simulation",
            "expected_files": ["config.yaml", "run.json"],
            "provenance": {"experiment": "parameter_recovery", "seed": 9901},
        },
        {
            "path": "experiments/jsu_showcase/results/canonical",
            "artifact_class": "simulation",
            "expected_files": ["config.yaml", "run.json"],
            "provenance": {"experiment": "jsu_showcase", "seed": 9901},
        },
    ]
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is True
    assert report.failures == ()


def test_stale_artifact_hash_is_reported(tmp_path: Path) -> None:
    """Recorded hashes make later promoted-artifact edits visible."""
    _write_artifact(tmp_path)
    metric = (
        tmp_path
        / "experiments"
        / "disentanglement"
        / "results"
        / "canonical"
        / "metrics"
        / "regional_components.json"
    )
    expected_digest = sha256(metric.read_bytes()).hexdigest()
    manifest = _manifest_payload()
    claim = manifest["claims"][0]
    claim["evidence"]["file_hashes"] = {
        "metrics/regional_components.json": expected_digest
    }
    metric.write_text('{"draws": 501}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    report = validate_evidence_manifest(manifest_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: stale artifact file "
        "metrics/regional_components.json has sha256 "
        f"{sha256(metric.read_bytes()).hexdigest()}, expected {expected_digest}",
    )


def test_checked_in_manifest_covers_required_claim_families() -> None:
    """The publication ledger includes the PRD-0003 headline claim families."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    families = {claim["family"] for claim in manifest["claims"]}

    assert families == {
        "benchmark_comparator",
        "disentanglement",
        "epistemic_bands",
        "vi_vs_nuts_limitation",
    }


def test_checked_in_manifest_rejects_current_benchmark_smoke_artifact() -> None:
    """The real ledger is a gate: smoke UCI output cannot satisfy the paper."""
    report = validate_evidence_manifest(MANIFEST, root=Path("."))

    assert report.ready is False
    assert (
        "benchmark-comparator-panel: requires full paper evidence but "
        "run.json marks the artifact as smoke"
    ) in report.failures


def test_cli_prints_actionable_readiness_failures(tmp_path: Path) -> None:
    """The publication gate is usable as a human-facing check command."""
    _write_artifact(tmp_path)
    missing = (
        tmp_path
        / "experiments"
        / "disentanglement"
        / "results"
        / "canonical"
        / "figures"
        / "disentanglement.pdf"
    )
    missing.unlink()
    manifest_path = _write_manifest(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.publication.evidence",
            str(manifest_path),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Publication evidence: NOT READY" in completed.stdout
    assert (
        "central-disentanglement: missing artifact file figures/disentanglement.pdf"
    ) in completed.stdout
