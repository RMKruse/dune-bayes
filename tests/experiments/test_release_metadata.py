"""Release and citation metadata tests (PRD-0003, GitHub #135)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_citation_metadata_parses_with_release_identity() -> None:
    """Citation metadata names the citable repository and pending paper fields."""
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "dune-bayes"
    assert citation["repository-code"] == "https://github.com/RMKruse/dune-bayes"
    assert citation["license"] == "MIT"
    assert citation["authors"] == [{"family-names": "Kruse", "given-names": "R.-M."}]

    preferred = citation["preferred-citation"]
    assert preferred["type"] == "article"
    assert preferred["title"] == (
        "DUNE: Distributional Uncertainty in Neural-additive Estimation"
    )
    assert preferred["notes"] == (
        "Paper DOI/preprint citation pending author approval; replace this "
        "preferred-citation before tagging the paper artifact release."
    )
    assert preferred["url"] == "https://github.com/RMKruse/dune-bayes"


def test_release_metadata_ties_tag_to_audit_and_publication_docs() -> None:
    """Release metadata links the paper tag/version to reviewer artifacts."""
    metadata_path = Path("experiments/publication/release-metadata.yaml")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))

    release = metadata["release"]
    assert release["paper_artifact_version"] == "0.1.0-paper"
    assert release["package_version"] == "0.1.0"
    assert release["planned_tag"] == "v0.1.0-paper"
    assert release["current_development_version"] == pyproject["project"]["version"]
    assert citation["version"] == release["current_development_version"]
    assert release["reproducibility_audit"]["human_report"] == (
        "experiments/publication/reproducibility-audit/audit-report.md"
    )
    assert release["reproducibility_audit"]["machine_report"] == (
        "experiments/publication/reproducibility-audit/audit-report.json"
    )

    for key in (
        "citation_metadata",
        "release_notes",
        "artifact_archiving_instructions",
        "evidence_manifest",
    ):
        assert Path(release[key]).is_file()

    release_notes = Path(release["release_notes"]).read_text(encoding="utf-8")
    archiving = Path(release["artifact_archiving_instructions"]).read_text(
        encoding="utf-8"
    )
    assert release["planned_tag"] in release_notes
    assert "Bounded Reproducibility Audit" in release_notes
    assert "Zenodo" in archiving
    assert "audit-report.json" in archiving


def test_readme_citation_text_matches_release_metadata_state() -> None:
    """README citation guidance agrees with the pending paper release state."""
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "There is no dune-bayes paper yet" not in readme
    assert "CITATION.cff" in readme
    assert "v0.1.0-paper" in readme
    assert "pending author approval" in readme
    assert "Zenodo" in readme
