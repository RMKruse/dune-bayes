"""Claim-to-evidence manifest validator (PRD-0003, GitHub #129)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvidenceReport:
    """Readiness report for a publication evidence manifest.

    Attributes:
        ready: Whether all claims have complete supporting evidence.
        failures: Actionable validation failures, empty when ready.
        claim_count: Number of claims inspected from the manifest.
    """

    ready: bool
    failures: tuple[str, ...]
    claim_count: int


def validate_evidence_manifest(
    manifest_path: Path | str,
    *,
    root: Path | str | None = None,
) -> EvidenceReport:
    """Validate a claim-to-evidence manifest against promoted artifacts.

    Args:
        manifest_path: YAML manifest path.
        root: Repository root used to resolve relative artifact locations.

    Returns:
        Publication readiness report with actionable failures.
    """
    manifest_file = Path(manifest_path)
    base = Path(root) if root is not None else manifest_file.parent
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))

    failures: list[str] = []
    claims = manifest.get("claims", []) if isinstance(manifest, dict) else []
    for claim in claims:
        claim_id = str(claim.get("id", "<missing id>"))
        for evidence in _evidence_entries(claim.get("evidence", {})):
            _validate_evidence_entry(
                claim=claim,
                claim_id=claim_id,
                evidence=evidence,
                root=base,
                failures=failures,
            )

    return EvidenceReport(
        ready=not failures,
        failures=tuple(failures),
        claim_count=len(claims),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the publication evidence validator as a CLI.

    Args:
        argv: Optional argument vector for tests.

    Returns:
        Process exit code: 0 when ready, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Validate the dune-bayes publication evidence manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    report = validate_evidence_manifest(args.manifest, root=args.root)
    status = "READY" if report.ready else "NOT READY"
    print(f"Publication evidence: {status} ({report.claim_count} claims)")
    for failure in report.failures:
        print(f"- {failure}")
    return 0 if report.ready else 1


def _validate_evidence_entry(
    *,
    claim: dict[str, Any],
    claim_id: str,
    evidence: dict[str, Any],
    root: Path,
    failures: list[str],
) -> None:
    """Validate one promoted artifact referenced by a paper claim."""
    if not evidence.get("artifact_class"):
        failures.append(f"{claim_id}: manifest evidence.artifact_class is required")
    artifact = root / str(evidence.get("path", ""))
    expected_files = evidence.get("expected_files", [])

    for expected_file in expected_files:
        if not (artifact / str(expected_file)).is_file():
            failures.append(f"{claim_id}: missing artifact file {expected_file}")

    file_hashes = evidence.get("file_hashes", {})
    for expected_file, expected_digest in file_hashes.items():
        artifact_file = artifact / str(expected_file)
        if artifact_file.is_file():
            actual_digest = sha256(artifact_file.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                failures.append(
                    f"{claim_id}: stale artifact file {expected_file} has "
                    f"sha256 {actual_digest}, expected {expected_digest}"
                )

    run_metadata, metadata_error = _read_json(artifact / "run.json")
    if metadata_error is not None:
        failures.append(f"{claim_id}: {metadata_error}")
    else:
        if claim.get("requires") == "full" and run_metadata.get("smoke") is True:
            failures.append(
                f"{claim_id}: requires full paper evidence but "
                "run.json marks the artifact as smoke"
            )
        provenance = evidence.get("provenance", {})
        for key, expected in provenance.items():
            if run_metadata.get(key) != expected:
                failures.append(
                    f"{claim_id}: run.json {key} is {run_metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )


def _evidence_entries(evidence: Any) -> tuple[dict[str, Any], ...]:
    """Normalize one evidence mapping or a list of evidence mappings."""
    if isinstance(evidence, dict):
        return (evidence,)
    if isinstance(evidence, list):
        return tuple(entry for entry in evidence if isinstance(entry, dict))
    return ({},)


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read JSON metadata when present, otherwise return an empty mapping."""
    if not path.is_file():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return {}, f"malformed JSON metadata {path.name}"
    if not isinstance(data, dict):
        return {}, f"malformed JSON metadata {path.name}"
    return data, None


if __name__ == "__main__":
    raise SystemExit(main())
