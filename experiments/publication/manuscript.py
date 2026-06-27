"""Manuscript claim-ledger validation (GitHub #143, parent PRD #142)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ManuscriptLedgerReport:
    """Readiness report for the manuscript-facing claim ledger.

    Attributes:
        ready: Whether the ledger satisfies the manuscript evidence contract.
        failures: Actionable ledger failures, empty when ready.
        claim_count: Number of manuscript claims declared in the ledger.
    """

    ready: bool
    failures: tuple[str, ...]
    claim_count: int


def validate_claim_ledger(
    ledger_path: Path | str,
    *,
    root: Path | str | None = None,
    evidence_manifest_path: Path | str | None = None,
) -> ManuscriptLedgerReport:
    """Validate manuscript claims against promoted publication evidence.

    Args:
        ledger_path: YAML manuscript claim-ledger path.
        root: Repository root used to resolve relative manifest and artifact
            locations. Defaults to the ledger's parent directory.
        evidence_manifest_path: Optional override for the promoted evidence
            manifest. When omitted, the ledger's ``evidence_manifest`` value is
            used.

    Returns:
        Ledger readiness report with actionable failures.
    """
    ledger_file = Path(ledger_path)
    base = Path(root) if root is not None else ledger_file.parent
    ledger = yaml.safe_load(ledger_file.read_text(encoding="utf-8"))
    manifest_reference = _manifest_reference(
        ledger,
        evidence_manifest_path=evidence_manifest_path,
    )
    manifest_file = _resolve_from_base(manifest_reference, base=base)
    evidence_index = _evidence_index(
        yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    )

    failures: list[str] = []
    claims = ledger.get("claims", []) if isinstance(ledger, dict) else []
    if not isinstance(claims, list):
        return ManuscriptLedgerReport(
            ready=False,
            failures=("claim-ledger: claims must be a list",),
            claim_count=0,
        )

    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            failures.append("claim-ledger: each claim must be a mapping")
            continue
        claim_id = str(claim.get("id", "<missing id>"))
        if claim_id in seen:
            failures.append(f"{claim_id}: duplicate manuscript claim id")
        seen.add(claim_id)
        if claim_id not in evidence_index:
            failures.append(
                f"{claim_id}: claim id is not present in {manifest_reference}"
            )
            continue
        _validate_claim_fields(claim, claim_id=claim_id, failures=failures)
        _validate_claim_evidence(
            claim,
            claim_id=claim_id,
            promoted_entries=evidence_index[claim_id],
            failures=failures,
        )

    return ManuscriptLedgerReport(
        ready=not failures,
        failures=tuple(failures),
        claim_count=len(claims),
    )


def main(argv: list[str] | None = None) -> int:
    """Run manuscript claim-ledger validation as a CLI.

    Args:
        argv: Optional argument vector for tests.

    Returns:
        Process exit code: 0 when the ledger is ready, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Validate the dune-bayes manuscript claim ledger."
    )
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--evidence-manifest", type=Path)
    args = parser.parse_args(argv)

    report = validate_claim_ledger(
        args.ledger,
        root=args.root,
        evidence_manifest_path=args.evidence_manifest,
    )
    status = "READY" if report.ready else "NOT READY"
    print(f"Manuscript claim ledger: {status} ({report.claim_count} claims)")
    for failure in report.failures:
        print(f"- {failure}")
    return 0 if report.ready else 1


def _manifest_reference(
    ledger: Any,
    *,
    evidence_manifest_path: Path | str | None,
) -> str:
    """Return the evidence manifest path recorded for this manuscript ledger."""
    if evidence_manifest_path is not None:
        return str(evidence_manifest_path)
    if isinstance(ledger, dict):
        reference = ledger.get("evidence_manifest")
        if reference:
            return str(reference)
    return "experiments/publication/evidence-manifest.yaml"


def _resolve_from_base(path: str, *, base: Path) -> Path:
    """Resolve a possibly-relative manuscript ledger path."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _evidence_index(manifest: Any) -> dict[str, set[tuple[str, str]]]:
    """Index promoted evidence entries by claim id."""
    index: dict[str, set[tuple[str, str]]] = {}
    claims = manifest.get("claims", []) if isinstance(manifest, dict) else []
    if not isinstance(claims, list):
        return index
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id", "<missing id>"))
        entries = index.setdefault(claim_id, set())
        for evidence in _evidence_entries(claim.get("evidence", {})):
            entries.add(
                (
                    str(evidence.get("artifact_class", "")),
                    str(evidence.get("path", "")),
                )
            )
    return index


def _evidence_entries(evidence: Any) -> tuple[dict[str, Any], ...]:
    """Normalize one evidence mapping or a list of evidence mappings."""
    if isinstance(evidence, dict):
        return (evidence,)
    if isinstance(evidence, list):
        return tuple(entry for entry in evidence if isinstance(entry, dict))
    return ({},)


def _validate_claim_fields(
    claim: dict[str, Any],
    *,
    claim_id: str,
    failures: list[str],
) -> None:
    """Validate manuscript-facing fields that are not in the evidence manifest."""
    if not str(claim.get("paper_claim", "")).strip():
        failures.append(f"{claim_id}: paper_claim is required")
    if not str(claim.get("limitation_note", "")).strip():
        failures.append(f"{claim_id}: limitation_note is required")

    outputs = claim.get("intended_outputs", [])
    if not isinstance(outputs, list) or not outputs:
        failures.append(f"{claim_id}: intended_outputs must list table/figure output")
        return
    for output in outputs:
        output_path = str(output)
        if not (
            output_path.startswith("figures/") or output_path.startswith("tables/")
        ):
            failures.append(
                f"{claim_id}: intended output {output_path} must be under "
                "figures/ or tables/"
            )


def _validate_claim_evidence(
    claim: dict[str, Any],
    *,
    claim_id: str,
    promoted_entries: set[tuple[str, str]],
    failures: list[str],
) -> None:
    """Validate evidence rows against the promoted evidence manifest."""
    evidence_rows = claim.get("evidence", [])
    if not isinstance(evidence_rows, list) or not evidence_rows:
        failures.append(f"{claim_id}: evidence must list promoted artifacts")
        return

    for evidence in evidence_rows:
        if not isinstance(evidence, dict):
            failures.append(f"{claim_id}: each evidence row must be a mapping")
            continue
        evidence_class = str(evidence.get("evidence_class", ""))
        promoted_artifact = str(evidence.get("promoted_artifact", ""))
        if not evidence_class:
            failures.append(f"{claim_id}: evidence.evidence_class is required")
        if not promoted_artifact:
            failures.append(f"{claim_id}: evidence.promoted_artifact is required")
        if "runs" in Path(promoted_artifact).parts:
            failures.append(
                f"{claim_id}: scratch artifact paths under runs/ cannot be used"
            )
        if (evidence_class, promoted_artifact) not in promoted_entries:
            failures.append(
                f"{claim_id}: evidence row does not match promoted manifest entry "
                f"({evidence_class}, {promoted_artifact})"
            )


if __name__ == "__main__":
    raise SystemExit(main())
