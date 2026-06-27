"""Documentation drift checks for the paper artifact state (GitHub #149)."""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.support import concrete_families

PUBLIC_DOCS = (
    Path("README.md"),
    Path("docs/architecture.md"),
    Path("docs/tutorials/reader_workflow.md"),
)


def test_public_docs_list_the_shipped_response_families() -> None:
    """Public docs name every concrete response family exported by the package."""
    family_names = {family.__name__ for family in concrete_families()}

    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        missing = sorted(name for name in family_names if name not in text)
        assert not missing, f"{path} is missing shipped families: {missing}"


def test_public_status_matches_release_metadata_and_ci_state() -> None:
    """README status should agree with paper-release metadata and CI activation."""
    readme = Path("README.md").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    metadata = yaml.safe_load(
        Path("experiments/publication/release-metadata.yaml").read_text(
            encoding="utf-8"
        )
    )

    release = metadata["release"]
    assert release["target_branch"] == "main"
    assert release["planned_tag"] in readme
    assert release["current_development_version"] in readme
    assert "target branch is `main`" in readme

    manual_only = "workflow_dispatch" in workflow and "#   push:" in workflow
    assert manual_only
    assert "manual-only CI" in readme
    assert "public release activation" in readme


def test_overview_docs_distinguish_effect_ribbons_from_response_bands() -> None:
    """Overview docs preserve the manuscript's uncertainty terminology."""
    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8").lower()
        assert "effect ribbon" in text
        assert "response-level" in text
        assert "epistemic" in text
        assert "aleatoric" in text
