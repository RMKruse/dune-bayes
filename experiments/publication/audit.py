"""Bounded reproducibility audit workflow (PRD-0003, GitHub #133)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.publication.artifacts import build_paper_artifacts
from experiments.publication.evidence import validate_evidence_manifest
from experiments.uci_benchmark import publication_gate


@dataclass(frozen=True)
class AuditCommand:
    """One bounded command run by the publication reproducibility audit.

    Args:
        name: Human-readable check name.
        category: Report category, such as ``core_package`` or
            ``experiment_smoke``.
        command: Argument vector executed from the repository root.
    """

    name: str
    category: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class AuditReport:
    """Result of a bounded reproducibility audit run.

    Attributes:
        ready: Whether every bounded check passed.
        failures: Actionable failures collected across the audit.
        machine_report: Path to the JSON report.
        human_report: Path to the Markdown report.
    """

    ready: bool
    failures: tuple[str, ...]
    machine_report: Path
    human_report: Path


def audit_publication_reproducibility(
    *,
    root: Path | str = Path("."),
    output_dir: Path | str,
    evidence_manifest: Path | str | None = None,
    benchmark_manifest: Path | str | None = None,
    commands: tuple[AuditCommand, ...] | None = None,
) -> AuditReport:
    """Run the bounded reviewer-facing reproducibility audit.

    Args:
        root: Repository root for commands and relative artifact paths.
        output_dir: Directory receiving reports and regenerated paper artifacts.
        evidence_manifest: Claim-to-evidence manifest; defaults to the checked-in
            publication manifest under ``experiments/publication``.
        benchmark_manifest: Benchmark claim manifest; defaults to the promoted
            canonical UCI benchmark manifest.
        commands: Optional reduced command set for tests or local dry runs.

    Returns:
        Paths and readiness status for the emitted audit reports.
    """
    repo = Path(root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(evidence_manifest)
        if evidence_manifest is not None
        else repo / "experiments" / "publication" / "evidence-manifest.yaml"
    )
    benchmark = (
        Path(benchmark_manifest)
        if benchmark_manifest is not None
        else repo
        / "experiments"
        / "uci_benchmark"
        / "results"
        / "canonical"
        / "benchmark-claims.yaml"
    )
    command_specs = commands if commands is not None else _default_commands(repo)

    dependency_readiness = _dependency_readiness(repo)
    command_results = tuple(
        _run_command(command, cwd=repo) for command in command_specs
    )
    evidence_report = validate_evidence_manifest(manifest, root=repo)
    benchmark_report = _run_benchmark_gate(benchmark)
    artifact_report = build_paper_artifacts(
        manifest,
        root=repo,
        output_dir=destination / "paper-artifacts",
    )

    failures = _audit_failures(
        dependency_readiness=dependency_readiness,
        command_results=command_results,
        evidence_failures=evidence_report.failures,
        benchmark_failures=benchmark_report["failures"],
        artifact_failures=artifact_report.failures,
    )
    ready = not failures

    payload = {
        "status": "ready" if ready else "not_ready",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "root": str(repo),
        "dependency_readiness": dependency_readiness,
        "bounded_scope": {
            "core_package_checks": "automated",
            "experiment_smokes": "automated",
            "canonical_evidence_validation": "automated",
            "benchmark_publication_gate": "automated",
            "paper_artifact_assembly": "automated",
            "full_canonical_reruns": "manual",
        },
        "checks": [result for result in command_results],
        "evidence_manifest": {
            "ready": evidence_report.ready,
            "claim_count": evidence_report.claim_count,
            "failures": list(evidence_report.failures),
            "manifest": str(manifest),
        },
        "benchmark_gate": benchmark_report,
        "paper_artifacts": {
            "ready": artifact_report.ready,
            "outputs": [str(path) for path in artifact_report.outputs],
            "failures": list(artifact_report.failures),
        },
        "caveats": [
            "Full canonical experiment reruns are manual",
            "The bounded audit regenerates smoke outputs and paper artifacts only",
            "Promoted canonical evidence remains the reviewed source for paper claims",
        ],
        "failures": list(failures),
    }

    machine_report = destination / "audit-report.json"
    human_report = destination / "audit-report.md"
    machine_report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    human_report.write_text(_human_report(payload), encoding="utf-8")

    return AuditReport(
        ready=ready,
        failures=failures,
        machine_report=machine_report,
        human_report=human_report,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the bounded publication reproducibility audit as a CLI.

    Args:
        argv: Optional argument vector for tests.

    Returns:
        Process exit code, where 0 means the bounded audit is ready.
    """
    parser = argparse.ArgumentParser(
        description="Run the bounded dune-bayes publication reproducibility audit."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/publication/reproducibility-audit"),
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    report = audit_publication_reproducibility(
        root=args.root,
        output_dir=args.output_dir,
        evidence_manifest=args.evidence_manifest,
        benchmark_manifest=args.benchmark_manifest,
    )
    status = "READY" if report.ready else "NOT READY"
    print(f"Bounded reproducibility audit: {status}")
    print(f"- machine report: {report.machine_report}")
    print(f"- human report: {report.human_report}")
    for failure in report.failures:
        print(f"- {failure}")
    return 0 if report.ready else 1


def _default_commands(root: Path) -> tuple[AuditCommand, ...]:
    """Return the bounded command set used by the reviewer-facing audit."""
    core = (
        AuditCommand(
            name="ruff format",
            category="core_package",
            command=(
                "uv",
                "run",
                "ruff",
                "format",
                "--check",
                "src",
                "tests",
                "experiments",
            ),
        ),
        AuditCommand(
            name="ruff lint",
            category="core_package",
            command=("uv", "run", "ruff", "check", "src", "tests", "experiments"),
        ),
        AuditCommand(
            name="mypy",
            category="core_package",
            command=("uv", "run", "mypy", "src/dune_bayes"),
        ),
        AuditCommand(
            name="pytest core",
            category="core_package",
            command=("uv", "run", "pytest", "-q"),
        ),
    )
    smoke_commands = tuple(
        AuditCommand(
            name=f"{config.parent.name} smoke",
            category="experiment_smoke",
            command=(
                "uv",
                "run",
                "python",
                str((config.parent / "run.py").relative_to(root)),
                str(config.relative_to(root)),
                "--smoke",
            ),
        )
        for config in sorted((root / "experiments").glob("*/config*.yaml"))
        if (config.parent / "run.py").is_file()
    )
    smoke_tests = (
        AuditCommand(
            name="experiment harness tests",
            category="experiment_smoke",
            command=("uv", "run", "pytest", "-q", "-m", "experiment"),
        ),
        AuditCommand(
            name="HMC agreement smoke tests",
            category="experiment_smoke",
            command=("uv", "run", "pytest", "-q", "-m", "hmc"),
        ),
    )
    return (*core, *smoke_commands, *smoke_tests)


def _dependency_readiness(root: Path) -> dict[str, Any]:
    """Check lockfile-backed dependency groups for the bounded audit."""
    pyproject = root / "pyproject.toml"
    lockfile = root / "uv.lock"
    failures: list[str] = []
    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as error:
        return {
            "ready": False,
            "install_command": "uv sync --locked --extra dev --extra experiments",
            "failures": [f"pyproject.toml is not readable: {error}"],
        }

    optional = project.get("optional-dependencies", {})
    dev = _normalized_requirements(optional.get("dev", []))
    experiments = _normalized_requirements(optional.get("experiments", []))
    for group, present, required in (
        ("dev", dev, ("pytest", "ruff", "mypy", "pyyaml")),
        ("experiments", experiments, ("jax", "numpyro", "pyyaml")),
    ):
        missing = [name for name in required if name not in present]
        if missing:
            failures.append(f"{group} dependency group is missing {', '.join(missing)}")
    if not lockfile.is_file():
        failures.append("uv.lock is required for --locked reviewer installs")

    return {
        "ready": not failures,
        "install_command": "uv sync --locked --extra dev --extra experiments",
        "groups": {
            "dev": sorted(dev),
            "experiments": sorted(experiments),
        },
        "lockfile": str(lockfile),
        "failures": failures,
    }


def _normalized_requirements(requirements: object) -> set[str]:
    """Return lowercase package names without version specifiers or extras."""
    if not isinstance(requirements, list):
        return set()
    names: set[str] = set()
    for requirement in requirements:
        raw = str(requirement).lower()
        name = raw.split("[", 1)[0]
        for separator in ("<", ">", "=", "!", "~"):
            name = name.split(separator, 1)[0]
        names.add(name.strip())
    return names


def _run_command(command: AuditCommand, *, cwd: Path) -> dict[str, Any]:
    """Run one bounded command and return serializable status."""
    completed = subprocess.run(
        command.command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "name": command.name,
        "category": command.category,
        "command": list(command.command),
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _run_benchmark_gate(manifest: Path) -> dict[str, Any]:
    """Run the existing benchmark publication gate and capture its output."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        returncode = publication_gate.main([str(manifest)])
    return {
        "ready": returncode == 0,
        "manifest": str(manifest),
        "stdout": stdout.getvalue().splitlines(),
        "failures": stderr.getvalue().splitlines(),
    }


def _audit_failures(
    *,
    dependency_readiness: dict[str, Any],
    command_results: tuple[dict[str, Any], ...],
    evidence_failures: tuple[str, ...],
    benchmark_failures: list[str],
    artifact_failures: tuple[str, ...],
) -> tuple[str, ...]:
    """Collect high-level audit failures."""
    failures: list[str] = []
    failures.extend(str(item) for item in dependency_readiness.get("failures", []))
    failures.extend(
        f"{result['name']} failed with exit code {result['returncode']}"
        for result in command_results
        if result["status"] != "pass"
    )
    failures.extend(evidence_failures)
    failures.extend(benchmark_failures)
    failures.extend(artifact_failures)
    return tuple(failures)


def _human_report(payload: dict[str, Any]) -> str:
    """Render the reviewer-facing Markdown report."""
    lines = [
        "# Bounded Reproducibility Audit",
        "",
        f"Status: {payload['status']}",
        "",
        "## Dependency Readiness",
        "",
        f"Install command: `{payload['dependency_readiness']['install_command']}`",
        f"Ready: {payload['dependency_readiness']['ready']}",
        "",
        "## Automated Checks",
        "",
    ]
    lines.extend(f"- {check['name']}: {check['status']}" for check in payload["checks"])
    lines.extend(
        [
            "",
            "## Evidence And Artifacts",
            "",
            (
                "- Evidence manifest: "
                f"{_ready_text(payload['evidence_manifest']['ready'])} "
                f"({payload['evidence_manifest']['claim_count']} claims)"
            ),
            f"- Benchmark gate: {_ready_text(payload['benchmark_gate']['ready'])}",
            f"- Paper artifacts: {_ready_text(payload['paper_artifacts']['ready'])}",
            "",
            "## Bounded Scope",
            "",
        ]
    )
    lines.extend(
        f"- {name.replace('_', ' ')}: {status}"
        for name, status in payload["bounded_scope"].items()
    )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in payload["caveats"])
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in payload["failures"])
    return "\n".join(lines) + "\n"


def _ready_text(ready: bool) -> str:
    """Return stable human-facing readiness text."""
    return "ready" if ready else "not ready"


def _tail(output: str, *, lines: int = 40) -> str:
    """Keep command output useful but bounded in the machine report."""
    return "\n".join(output.splitlines()[-lines:])


if __name__ == "__main__":
    raise SystemExit(main())
