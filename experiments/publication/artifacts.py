"""Paper artifact builder for promoted evidence (PRD-0003, GitHub #132/#146)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from experiments.publication.evidence import validate_evidence_manifest


@dataclass(frozen=True)
class ArtifactBuildReport:
    """Result of building paper-facing artifacts from canonical evidence.

    Attributes:
        ready: Whether all requested artifacts were written.
        failures: Actionable build failures, empty when ready.
        outputs: Deterministic output paths written for manuscript use.
    """

    ready: bool
    failures: tuple[str, ...]
    outputs: tuple[Path, ...]


def build_paper_artifacts(
    manifest_path: Path | str,
    *,
    output_dir: Path | str,
    root: Path | str | None = None,
) -> ArtifactBuildReport:
    """Build paper-facing tables, figures, and provenance from promoted evidence.

    Args:
        manifest_path: Publication evidence manifest path.
        output_dir: Directory where stable paper artifacts are written.
        root: Repository root used to resolve relative evidence paths.

    Returns:
        Build report with stable output paths or actionable failures.
    """
    manifest_file = Path(manifest_path)
    base = Path(root) if root is not None else manifest_file.parent
    destination = Path(output_dir)

    readiness = validate_evidence_manifest(manifest_file, root=base)
    failures = list(readiness.failures)
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    entries = _manifest_entries(manifest)
    failures.extend(_canonical_path_failures(entries))
    if failures:
        return ArtifactBuildReport(
            ready=False,
            failures=tuple(failures),
            outputs=(),
        )

    figures_dir = destination / "figures"
    tables_dir = destination / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_outputs = _copy_declared_figures(entries, base=base, figures_dir=figures_dir)
    table_outputs = _copy_declared_tables(entries, base=base, tables_dir=tables_dir)
    summary_table = tables_dir / "evidence-summary.csv"
    _write_summary_table(summary_table, entries, base=base)
    provenance = destination / "provenance.json"
    _write_provenance(
        provenance,
        manifest_path=manifest_file,
        entries=entries,
        base=base,
    )
    artifact_metadata = destination / "artifact-metadata.yaml"
    _write_artifact_metadata(artifact_metadata, entries=entries)
    appendix = destination / "reviewer-evidence-appendix.md"
    _write_reviewer_appendix(
        appendix,
        entries=entries,
    )

    return ArtifactBuildReport(
        ready=True,
        failures=(),
        outputs=(
            *figure_outputs,
            *table_outputs,
            summary_table,
            provenance,
            artifact_metadata,
            appendix,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the paper artifact builder as a CLI.

    Args:
        argv: Optional argument vector for tests.

    Returns:
        Process exit code: 0 when artifacts are ready, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Build dune-bayes paper artifacts from promoted evidence."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_paper_artifacts(
        args.manifest,
        output_dir=args.output_dir,
        root=args.root,
    )
    status = "READY" if report.ready else "NOT READY"
    print(f"Paper artifacts: {status} ({len(report.outputs)} outputs)")
    for output in report.outputs:
        print(f"- {output}")
    for failure in report.failures:
        print(f"- {failure}")
    return 0 if report.ready else 1


def _manifest_entries(manifest: Any) -> tuple[dict[str, Any], ...]:
    """Flatten manifest claims into claim/evidence entries."""
    if not isinstance(manifest, dict):
        return ()
    entries: list[dict[str, Any]] = []
    claims = manifest.get("claims", [])
    if not isinstance(claims, list):
        return ()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        for evidence in _evidence_entries(claim.get("evidence", {})):
            entries.append({"claim": claim, "evidence": evidence})
    return tuple(entries)


def _evidence_entries(evidence: Any) -> tuple[dict[str, Any], ...]:
    """Normalize one evidence mapping or a list of evidence mappings."""
    if isinstance(evidence, dict):
        return (evidence,)
    if isinstance(evidence, list):
        return tuple(entry for entry in evidence if isinstance(entry, dict))
    return ({},)


def _canonical_path_failures(entries: tuple[dict[str, Any], ...]) -> list[str]:
    """Reject scratch output before it can be copied into paper artifacts."""
    failures: list[str] = []
    for entry in entries:
        claim = entry["claim"]
        evidence = entry["evidence"]
        claim_id = str(claim.get("id", "<missing id>"))
        evidence_path = str(evidence.get("path", ""))
        parts = Path(evidence_path).parts
        if "runs" in parts:
            failures.append(
                f"{claim_id}: scratch artifact paths under runs/ cannot be used"
            )
    return failures


def _copy_declared_figures(
    entries: tuple[dict[str, Any], ...],
    *,
    base: Path,
    figures_dir: Path,
) -> tuple[Path, ...]:
    """Copy manifest-declared figures to stable claim-prefixed filenames."""
    records = _declared_files(entries, parent="figures", suffixes=None)
    counts = _candidate_counts(records)
    outputs: list[Path] = []
    used: set[str] = set()
    for record in records:
        output = figures_dir / _stable_output_name(record, counts=counts, used=used)
        shutil.copyfile(base / record["evidence_path"] / record["relative"], output)
        outputs.append(output)
    return tuple(outputs)


def _copy_declared_tables(
    entries: tuple[dict[str, Any], ...],
    *,
    base: Path,
    tables_dir: Path,
) -> tuple[Path, ...]:
    """Copy manifest-declared CSV metrics to stable claim-prefixed filenames."""
    records = _declared_files(entries, parent="metrics", suffixes={".csv"})
    counts = _candidate_counts(records)
    outputs: list[Path] = []
    used: set[str] = set()
    for record in records:
        output = tables_dir / _stable_output_name(record, counts=counts, used=used)
        shutil.copyfile(base / record["evidence_path"] / record["relative"], output)
        outputs.append(output)
    return tuple(outputs)


def _declared_files(
    entries: tuple[dict[str, Any], ...],
    *,
    parent: str,
    suffixes: set[str] | None,
) -> tuple[dict[str, Any], ...]:
    """Return manifest-declared files that should become paper artifacts."""
    records: list[dict[str, Any]] = []
    for entry in entries:
        claim = entry["claim"]
        evidence = entry["evidence"]
        for expected_file in evidence.get("expected_files", []):
            relative = Path(str(expected_file))
            if relative.parts[:1] != (parent,):
                continue
            if suffixes is not None and relative.suffix not in suffixes:
                continue
            records.append(
                {
                    "claim_id": str(claim.get("id", "<missing id>")),
                    "artifact_class": str(evidence.get("artifact_class", "")),
                    "evidence_path": Path(str(evidence.get("path", ""))),
                    "relative": relative,
                    "artifact_metadata": evidence.get("artifact_metadata", {}),
                }
            )
    return tuple(records)


def _candidate_counts(records: tuple[dict[str, Any], ...]) -> dict[str, int]:
    """Count default output names so only collisions are disambiguated."""
    counts: dict[str, int] = {}
    for record in records:
        candidate = _default_output_name(record)
        counts[candidate] = counts.get(candidate, 0) + 1
    return counts


def _stable_output_name(
    record: dict[str, Any],
    *,
    counts: dict[str, int],
    used: set[str],
) -> str:
    """Return a deterministic, non-overwriting paper artifact filename."""
    candidate = _default_output_name(record)
    if counts[candidate] > 1:
        candidate = (
            f"{record['claim_id']}__{_disambiguator(record)}__{record['relative'].name}"
        )
    unique = candidate
    index = 2
    while unique in used:
        stem = Path(candidate).stem
        suffix = Path(candidate).suffix
        unique = f"{stem}__{index}{suffix}"
        index += 1
    used.add(unique)
    return unique


def _default_output_name(record: dict[str, Any]) -> str:
    """Return the stable default claim-prefixed filename."""
    relative_parent = record["relative"].parts[1:-1]
    if relative_parent:
        return (
            f"{record['claim_id']}__{_slug('__'.join(relative_parent))}__"
            f"{record['relative'].name}"
        )
    return f"{record['claim_id']}__{record['relative'].name}"


def _disambiguator(record: dict[str, Any]) -> str:
    """Prefer dataset/subtable names, falling back to the canonical run name."""
    relative_parent = record["relative"].parts[1:-1]
    if relative_parent:
        return _slug("__".join(relative_parent))
    return _slug(record["evidence_path"].name)


def _slug(value: str) -> str:
    """Make a path component safe for deterministic artifact filenames."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "artifact"


def _write_summary_table(
    path: Path,
    entries: tuple[dict[str, Any], ...],
    *,
    base: Path,
) -> None:
    """Write a compact claim-to-evidence table for release review."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id",
                "family",
                "requires",
                "artifact_class",
                "path",
                "experiment",
                "seed",
            ],
        )
        writer.writeheader()
        for entry in entries:
            claim = entry["claim"]
            evidence = entry["evidence"]
            run_metadata = _read_json(base / str(evidence.get("path", "")) / "run.json")
            writer.writerow(
                {
                    "claim_id": str(claim.get("id", "")),
                    "family": str(claim.get("family", "")),
                    "requires": str(claim.get("requires", "")),
                    "artifact_class": str(evidence.get("artifact_class", "")),
                    "path": str(evidence.get("path", "")),
                    "experiment": str(run_metadata.get("experiment", "")),
                    "seed": str(run_metadata.get("seed", "")),
                }
            )


def _write_provenance(
    path: Path,
    *,
    manifest_path: Path,
    entries: tuple[dict[str, Any], ...],
    base: Path,
) -> None:
    """Record source artifacts and generation metadata for the build."""
    payload = {
        "manifest": str(manifest_path),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "builder_version": _builder_version(),
        "inputs": [
            {
                "claim_id": str(entry["claim"].get("id", "")),
                "artifact_class": str(entry["evidence"].get("artifact_class", "")),
                "path": str(entry["evidence"].get("path", "")),
                "run_metadata": _read_json(
                    base / str(entry["evidence"].get("path", "")) / "run.json"
                ),
            }
            for entry in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _write_artifact_metadata(
    path: Path,
    *,
    entries: tuple[dict[str, Any], ...],
) -> None:
    """Write manuscript-facing captions and uncertainty-component labels."""
    payload = {
        "version": 1,
        "artifacts": list(_artifact_metadata_records(entries)),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_reviewer_appendix(
    path: Path,
    *,
    entries: tuple[dict[str, Any], ...],
) -> None:
    """Write the reviewer-facing claim-to-evidence appendix."""
    output_records = _artifact_output_records(entries)
    lines = [
        "# Reviewer Evidence Appendix",
        "",
        "This appendix is generated from the promoted publication evidence manifest.",
        "",
    ]
    _append_reviewer_notes(lines)
    for artifact_class in _artifact_classes(entries):
        lines.extend([f"## {_artifact_class_heading(artifact_class)}", ""])
        for entry in entries:
            evidence = entry["evidence"]
            if str(evidence.get("artifact_class", "")) != artifact_class:
                continue
            _append_claim_evidence(lines, entry, output_records=output_records)
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_reviewer_notes(lines: list[str]) -> None:
    """Append ADR-backed conventions that reviewers need beside the evidence."""
    lines.extend(
        [
            "## Reviewer Conventions",
            "",
            "- epistemic effect ribbons are centered, epistemic-only credible "
            "ribbons for per-feature effects. They are not response-level "
            "predictive bands.",
            "- response-level predictive bands combine epistemic + aleatoric "
            "uncertainty and are interpreted as prediction intervals.",
            "- centered effect recovery is evaluated against centered posterior "
            "draws because additive shape functions are identified only up to "
            "level; intercept coverage is reported separately.",
            "- Simulation coverage is measured and reported rather than asserted "
            "correct, preserving the mean-field VI narrowness limitation in "
            "ADR-0001.",
            "- VI-vs-NUTS evidence is validation-only NUTS evidence from "
            "experiments/. dune-bayes does not ship an MCMC backend; ADR-0006 "
            "keeps JAX/NumPyro behind a future inference seam.",
            "- Family parameterizations follow the package glossary: positivity "
            "uses softplus(x) + EPS, and Johnson's SU uses the scipy johnsonsu "
            "parameterization.",
            "",
        ]
    )


def _artifact_classes(entries: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Return artifact classes in manifest order."""
    classes: list[str] = []
    for entry in entries:
        artifact_class = str(entry["evidence"].get("artifact_class", ""))
        if artifact_class not in classes:
            classes.append(artifact_class)
    return tuple(classes)


def _artifact_class_heading(artifact_class: str) -> str:
    """Return reviewer-facing evidence class labels."""
    headings = {
        "simulation": "Simulation Evidence",
        "real_data_benchmark": "Real-Data Benchmark Evidence",
        "validation": "Validation Evidence",
    }
    return headings.get(
        artifact_class, f"{artifact_class.replace('_', ' ').title()} Evidence"
    )


def _append_claim_evidence(
    lines: list[str],
    entry: dict[str, Any],
    *,
    output_records: tuple[dict[str, str], ...],
) -> None:
    """Append one claim-to-evidence mapping to appendix lines."""
    claim = entry["claim"]
    evidence = entry["evidence"]
    claim_id = str(claim.get("id", ""))
    evidence_path = str(evidence.get("path", ""))
    claim_outputs = [
        record["output"]
        for record in output_records
        if record["claim_id"] == claim_id and record["evidence_path"] == evidence_path
    ]
    lines.extend(
        [
            f"### {claim_id}",
            "",
            str(claim.get("statement", "")).strip(),
            "",
            f"- Evidence class: {evidence.get('artifact_class', '')}",
            f"- Promoted evidence: {evidence.get('path', '')}",
            "- Artifact-builder outputs:",
        ]
    )
    lines.extend(f"  - {output}" for output in claim_outputs)
    lines.append("")


def _artifact_output_records(
    entries: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    """Return stable appendix output records for each manifest-declared artifact."""
    return (
        *_declared_output_records(
            entries,
            parent="figures",
            output_parent="figures",
            suffixes=None,
        ),
        *_declared_output_records(
            entries,
            parent="metrics",
            output_parent="tables",
            suffixes={".csv"},
        ),
    )


def _artifact_metadata_records(
    entries: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    """Return metadata rows for copied manuscript artifacts."""
    return (
        *_declared_metadata_records(
            entries,
            parent="figures",
            output_parent="figures",
            suffixes=None,
        ),
        *_declared_metadata_records(
            entries,
            parent="metrics",
            output_parent="tables",
            suffixes={".csv"},
        ),
    )


def _declared_output_records(
    entries: tuple[dict[str, Any], ...],
    *,
    parent: str,
    output_parent: str,
    suffixes: set[str] | None,
) -> tuple[dict[str, str], ...]:
    """Return appendix records using the same naming rules as copied artifacts."""
    records = _declared_files(entries, parent=parent, suffixes=suffixes)
    counts = _candidate_counts(records)
    used: set[str] = set()
    output_records: list[dict[str, str]] = []
    for record in records:
        output_name = _stable_output_name(record, counts=counts, used=used)
        output_records.append(
            {
                "claim_id": str(record["claim_id"]),
                "evidence_path": str(record["evidence_path"]),
                "output": f"{output_parent}/{output_name}",
            }
        )
    return tuple(output_records)


def _declared_metadata_records(
    entries: tuple[dict[str, Any], ...],
    *,
    parent: str,
    output_parent: str,
    suffixes: set[str] | None,
) -> tuple[dict[str, str], ...]:
    """Return artifact metadata using the copied-output naming rules."""
    records = _declared_files(entries, parent=parent, suffixes=suffixes)
    counts = _candidate_counts(records)
    used: set[str] = set()
    metadata_records: list[dict[str, str]] = []
    for record in records:
        relative = str(record["relative"])
        output_name = _stable_output_name(record, counts=counts, used=used)
        manifest_metadata = record["artifact_metadata"]
        metadata = (
            manifest_metadata.get(relative, {})
            if isinstance(manifest_metadata, dict)
            else {}
        )
        metadata_records.append(
            {
                "output": f"{output_parent}/{output_name}",
                "claim_id": str(record["claim_id"]),
                "evidence_path": str(record["evidence_path"]),
                "source_file": relative,
                "artifact_class": str(record["artifact_class"]),
                "uncertainty_component": str(
                    metadata.get("uncertainty_component", "")
                ).strip(),
                "caption": str(metadata.get("caption", "")).strip(),
            }
        )
    return tuple(metadata_records)


def _read_json(path: Path) -> dict[str, Any]:
    """Read validated JSON metadata from a promoted artifact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _builder_version() -> str:
    """Return installed package version metadata when available."""
    try:
        return version("dune-bayes")
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
