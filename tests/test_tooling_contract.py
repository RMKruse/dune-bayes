"""Tooling contract tests (ADR-0008, GitHub #108)."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path


def _pyproject() -> dict[str, object]:
    """Read the project metadata that defines the public tooling contract."""
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted call target for simple attribute/name expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def test_mypy_public_signature_rule_is_ci_gating() -> None:
    """Package code is checked with disallow_untyped_defs in the CI gate."""
    project = _pyproject()
    mypy_config = project["tool"]["mypy"]
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    package_overrides = [
        override
        for override in mypy_config["overrides"]
        if override.get("module") == "dune_bayes.*"
    ]

    assert package_overrides == [
        {"module": "dune_bayes.*", "disallow_untyped_defs": True}
    ]
    assert "strict = true" not in Path("pyproject.toml").read_text(encoding="utf-8")
    assert "uv run mypy src/dune_bayes" in workflow


def test_slow_suites_are_registered_opt_in_and_ci_covered() -> None:
    """The default pytest gate excludes slow suites; CI runs opt-in suites."""
    project = _pyproject()
    pytest_config = project["tool"]["pytest"]["ini_options"]
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert pytest_config["addopts"] == "-m 'not hmc and not experiment'"
    assert any(marker.startswith("hmc:") for marker in pytest_config["markers"])
    assert any(marker.startswith("experiment:") for marker in pytest_config["markers"])
    assert any(
        marker.startswith("full_experiment:") for marker in pytest_config["markers"]
    )
    assert "uv run pytest -q" in workflow
    assert "uv run pytest -q -m hmc" in workflow
    assert 'uv run pytest -q -m "experiment and not full_experiment"' in workflow


def test_pytest_marker_selection_makes_experiments_opt_in() -> None:
    """Pytest's public marker selection keeps experiment tests out by default."""
    default = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/experiments"],
        check=False,
        capture_output=True,
        text=True,
    )
    experiment = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "experiment",
            "tests/experiments",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    hmc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "hmc",
            "tests/experiments",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert default.returncode == 5
    assert "no tests collected" in default.stdout
    assert "deselected" in default.stdout
    assert experiment.returncode == 0
    assert "tests/experiments/test_" in experiment.stdout
    assert hmc.returncode == 0
    assert "tests/experiments/test_hmc_agreement.py" in hmc.stdout


def test_correctness_tests_cannot_gain_skipif_or_xfail() -> None:
    """Numerical correctness gates are unconditional unless marked slow opt-in."""
    banned_targets = {"pytest.mark.skipif", "pytest.mark.xfail"}
    violations: list[str] = []

    for path in sorted(Path("tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = _dotted_name(node.func)
                if target in banned_targets:
                    violations.append(f"{path}:{node.lineno}: {target}")

    assert violations == []


def test_claude_tooling_section_records_issue_108_decisions() -> None:
    """CLAUDE.md is consistent with the mechanized tooling policy."""
    claude = Path("CLAUDE.md").read_text(encoding="utf-8")

    assert "disallow_untyped_defs = true" in claude
    assert "CI-gating" in claude
    assert "Full `--strict` is deliberately NOT the target" in claude
    assert "Correctness tests are never skippable" in claude
    assert "(`hmc`, `experiment` markers) are opt-**in**" in claude
