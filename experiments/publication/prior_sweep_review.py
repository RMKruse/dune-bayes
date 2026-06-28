"""Prior/smoothness sweep review helpers (PRD-0004, GitHub #160/#161)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIRMATORY_SEEDS = (9801, 9811, 9821)
_BASELINE_RUN_NAME = "prior-sweep-normal-fixed-1p0"
_CSV_FIELDS = (
    "rank",
    "config_file",
    "run_name",
    "seed",
    "smoke",
    "coverage_mae",
    "coverage_max_abs_error",
    "intercept_coverage_mae",
    "final_loss",
    "final_nll",
    "final_kl",
    "prior_scale_mode",
    "recovery_figure",
    "calibration_figure",
)


@dataclass(frozen=True)
class PriorSweepDecisionReport:
    """Readiness report for the reviewed prior/smoothness promotion decision.

    Attributes:
        ready: Whether the decision record is complete and internally coherent.
        failures: Actionable validation failures, empty when ready.
    """

    ready: bool
    failures: tuple[str, ...]


def prior_sweep_commands(root: Path | str) -> tuple[dict[str, Any], ...]:
    """List smoke and full commands for pre-registered Normal sweep candidates.

    Args:
        root: Repository root.

    Returns:
        One command row per committed Normal prior/smoothness sweep config.
    """
    repo_root = Path(root)
    config_dir = repo_root / "experiments" / "parameter_recovery" / "prior_sweep"
    rows: list[dict[str, Any]] = []
    for config_path in sorted(config_dir.glob("normal-*.yaml")):
        rows.append(
            {
                "candidate": config_path.stem,
                "config": str(config_path.relative_to(repo_root)),
                "smoke_command": _command_for_config(
                    repo_root, config_path, smoke=True
                ),
                "full_command": _command_for_config(
                    repo_root, config_path, smoke=False
                ),
            }
        )
    return tuple(rows)


def summarize_prior_sweep_outputs(root: Path | str) -> tuple[dict[str, Any], ...]:
    """Summarize available ignored prior/smoothness sweep outputs.

    Args:
        root: Repository root.

    Returns:
        One summary row per complete scratch sweep run. Incomplete run
        directories are skipped so a top-to-bottom notebook run stays safe while
        experiments are still in progress.
    """
    repo_root = Path(root)
    run_root = (
        repo_root
        / "experiments"
        / "parameter_recovery"
        / "runs"
        / "parameter_recovery_normal"
    )
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(run_root.glob("prior-sweep-normal-*")):
        calibration_path = run_dir / "metrics" / "calibration.csv"
        prior_path = run_dir / "metrics" / "prior_scale.json"
        config_path = run_dir / "config.yaml"
        if (
            not calibration_path.exists()
            or not prior_path.exists()
            or not config_path.exists()
        ):
            continue
        coverage = _coverage_errors(calibration_path)
        prior_state = _read_json(prior_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        architecture = config["architecture"]
        rows.append(
            {
                "candidate": run_dir.name,
                "configured_prior_scale": architecture["prior_scale"],
                "prior": architecture.get("prior", "fixed"),
                "mean_abs_coverage_error": coverage["mean_abs_error"],
                "max_abs_coverage_error": coverage["max_abs_error"],
                "min_empirical_coverage": coverage["min_empirical_coverage"],
                "prior_mode": prior_state.get("mode"),
                "scale": prior_state.get("scale"),
                "scale_median": prior_state.get("scale_median"),
                "scale_mean": prior_state.get("scale_mean"),
                "run_path": str(run_dir.relative_to(repo_root)),
            }
        )
    return tuple(rows)


def validate_prior_sweep_decision(
    decision_path: Path | str,
    *,
    root: Path | str | None = None,
) -> PriorSweepDecisionReport:
    """Validate the human-reviewed prior/smoothness publication decision.

    Args:
        decision_path: YAML decision record for GitHub #163.
        root: Repository root used to resolve relative paths. Defaults to the
            current working directory because the checked-in decision stores
            repository-relative paths.

    Returns:
        Readiness report for the decision record.
    """
    decision_file = Path(decision_path)
    repo_root = Path(root) if root is not None else Path(".")
    decision = yaml.safe_load(decision_file.read_text(encoding="utf-8"))
    failures: list[str] = []

    if not isinstance(decision, dict):
        return PriorSweepDecisionReport(
            ready=False,
            failures=("decision record must be a YAML mapping",),
        )

    if decision.get("github_issue") != 163:
        failures.append("decision record must reference GitHub issue 163")
    if decision.get("status") != "no_promotion":
        failures.append("decision status must record the no_promotion outcome")

    for field in (
        "reviewed_against",
        "paper_results_notebook",
        "artifact_builder",
        "publication_evidence_manifest",
    ):
        path = repo_root / str(decision.get(field, ""))
        if not path.is_file():
            failures.append(f"{field} does not point to a checked-in file: {path}")

    manifest_path = repo_root / str(decision.get("publication_evidence_manifest", ""))
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        normal_evidence_path = _normal_publication_evidence_path(manifest)
        if normal_evidence_path is None:
            failures.append("publication manifest is missing Normal recovery evidence")
        elif decision.get("canonical_artifact") != normal_evidence_path:
            failures.append(
                "canonical_artifact must match the Normal publication evidence path"
            )
    else:
        normal_evidence_path = None

    canonical_path = repo_root / str(decision.get("canonical_artifact", ""))
    normal_config_path = canonical_path / "config.yaml"
    if normal_config_path.is_file():
        normal_config = yaml.safe_load(normal_config_path.read_text(encoding="utf-8"))
        prior_scale = normal_config.get("architecture", {}).get("prior_scale")
        if prior_scale != 1.0:
            failures.append("canonical Normal evidence must remain at prior_scale=1.0")
    else:
        failures.append(f"canonical artifact config is missing: {normal_config_path}")

    _validate_no_promotion_policy(decision, failures)
    _validate_review_summary(decision, failures)

    follow_up = str(decision.get("follow_up", ""))
    if "last-layer" not in follow_up or "low-rank covariance" not in follow_up:
        failures.append(
            "no-promotion decision must name the richer posterior follow-up"
        )

    if normal_evidence_path is not None and not canonical_path.is_dir():
        failures.append(f"canonical artifact directory is missing: {canonical_path}")

    return PriorSweepDecisionReport(
        ready=not failures,
        failures=tuple(failures),
    )


def materialize_confirmatory_configs(
    root: Path | str,
    candidate_config: Path | str | None,
) -> tuple[dict[str, Any], ...]:
    """Write baseline-versus-candidate confirmatory configs to ignored scratch.

    Args:
        root: Repository root.
        candidate_config: Selected screening candidate config, or ``None`` when
            no candidate has been chosen.

    Returns:
        Command rows for the generated confirmatory configs. No files are written
        until a candidate config is provided.
    """
    if candidate_config is None:
        return ()

    repo_root = Path(root)
    config_dir = repo_root / "experiments" / "parameter_recovery" / "prior_sweep"
    baseline_config = config_dir / "normal-fixed-1p0.yaml"
    selected_config = Path(candidate_config)
    if not selected_config.is_absolute():
        selected_config = repo_root / selected_config
    output_dir = (
        repo_root
        / "experiments"
        / "parameter_recovery"
        / "runs"
        / "confirmatory-configs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source_config in (baseline_config, selected_config):
        base = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        label = source_config.stem
        for seed in CONFIRMATORY_SEEDS:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["artifacts"] = {
                "root": "../runs",
                "run_name": f"confirmatory-{label}-seed-{seed}",
            }
            target = output_dir / f"{label}-seed-{seed}.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            rows.append(
                {
                    "candidate": label,
                    "seed": seed,
                    "config": str(target.relative_to(repo_root)),
                    "smoke_command": _command_for_config(repo_root, target, smoke=True),
                    "full_command": _command_for_config(repo_root, target, smoke=False),
                }
            )
    return tuple(rows)


def write_review(
    *,
    config_dir: Path,
    runs_root: Path,
    output_dir: Path,
    decision: str = "undecided",
    decision_notes: str = "",
) -> dict[str, Any]:
    """Write JSON and CSV review summaries for completed sweep runs.

    Args:
        config_dir: Directory containing the pre-registered Normal sweep configs.
        runs_root: Ignored experiment scratch root containing completed runs.
        output_dir: Ignored output directory for review summaries.
        decision: ``undecided``, ``no-candidate``, or ``nominate:<run_name>``.
        decision_notes: Human inspection notes to persist with the decision.

    Returns:
        The JSON-serializable review payload.
    """
    candidates = [
        _summarize_candidate(path, config, runs_root=runs_root)
        for path, config in _candidate_configs(config_dir)
    ]
    if len(candidates) != 5:
        raise ValueError(
            f"expected five Normal sweep candidates, found {len(candidates)}"
        )
    run_names = {str(candidate["run_name"]) for candidate in candidates}
    if _BASELINE_RUN_NAME not in run_names:
        raise ValueError(f"missing baseline candidate: {_BASELINE_RUN_NAME}")
    ranking = _rank(candidates)
    baseline = next(
        candidate
        for candidate in candidates
        if candidate["run_name"] == _BASELINE_RUN_NAME
    )
    payload = {
        "baseline_run_name": _BASELINE_RUN_NAME,
        "baseline_coverage_mae": baseline["coverage"]["mean_absolute_error"],
        "ranking": ranking,
        "decision": _decision(decision, decision_notes, run_names),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screening_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_review_csv(output_dir / "screening_summary.csv", ranking)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the prior/smoothness screening review CLI.

    Args:
        argv: Optional CLI arguments; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/parameter_recovery/prior_sweep"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/parameter_recovery/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/parameter_recovery/runs/prior-sweep-review-9801"),
    )
    parser.add_argument("--decision", default="undecided")
    parser.add_argument("--decision-notes", default="")
    args = parser.parse_args(argv)

    payload = write_review(
        config_dir=args.config_dir,
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        decision=args.decision,
        decision_notes=args.decision_notes,
    )
    best = payload["ranking"][0]
    best_mae = float(best["coverage"]["mean_absolute_error"])
    if not math.isfinite(best_mae):
        raise ValueError("best candidate has non-finite coverage error")
    print(
        f"wrote {args.output_dir} with best={best['run_name']} "
        f"coverage_mae={best_mae:.6f}"
    )


def _command_for_config(repo_root: Path, config_path: Path, *, smoke: bool) -> str:
    """Return the shell command for one parameter-recovery config."""
    parts = [
        "uv",
        "run",
        "--extra",
        "experiments",
        "python",
        "experiments/parameter_recovery/run.py",
        str(config_path.relative_to(repo_root)),
    ]
    if smoke:
        parts.append("--smoke")
    return " ".join(shlex.quote(part) for part in parts)


def _coverage_errors(path: Path) -> dict[str, float]:
    """Compute calibration review metrics from one coverage CSV."""
    rows = _read_metric_rows(path)
    abs_errors = [float(row["absolute_error"]) for row in rows]
    empirical_coverages = [float(row["empirical_coverage"]) for row in rows]
    return {
        "mean_abs_error": sum(abs_errors) / len(abs_errors),
        "max_abs_error": max(abs_errors),
        "min_empirical_coverage": min(empirical_coverages),
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object artifact: {path}")
    return payload


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    """Read one long-form calibration table with numeric fields parsed."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        nominal = float(row["nominal"])
        empirical = float(row["empirical_coverage"])
        parsed.append(
            {
                "parameter": row["parameter"],
                "nominal": nominal,
                "empirical_coverage": empirical,
                "absolute_error": abs(empirical - nominal),
            }
        )
    return parsed


def _coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize absolute coverage error over a long-form metrics table."""
    errors = [float(row["absolute_error"]) for row in rows]
    return {
        "mean_absolute_error": float(sum(errors) / len(errors)),
        "max_absolute_error": float(max(errors)),
    }


def _candidate_configs(config_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load the pre-registered Normal sweep config files."""
    candidates = []
    for path in sorted(config_dir.glob("normal-*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config["family"] == "normal":
            candidates.append((path, config))
    return candidates


def _required_artifact(path: Path) -> Path:
    """Return a required artifact path or raise an actionable error."""
    if not path.is_file():
        raise FileNotFoundError(f"missing required sweep artifact: {path}")
    return path


def _prior_scale_mode(diagnostics: Mapping[str, Any]) -> str:
    """Return the review-facing prior-scale mode."""
    return str(diagnostics.get("mode", "unknown"))


def _run_root(config: Mapping[str, Any], runs_root: Path) -> Path:
    """Resolve the configured run directory under the supplied scratch root."""
    return runs_root / str(config["experiment"]) / str(config["artifacts"]["run_name"])


def _normal_publication_evidence_path(manifest: object) -> str | None:
    """Return the paper-facing Normal recovery evidence path, if present."""
    if not isinstance(manifest, dict):
        return None
    for claim in manifest.get("claims", []):
        if not isinstance(claim, dict):
            continue
        if claim.get("id") != "per-feature-per-parameter-epistemic-bands":
            continue
        evidence_entries = claim.get("evidence", [])
        if isinstance(evidence_entries, dict):
            evidence_entries = [evidence_entries]
        for evidence in evidence_entries:
            if not isinstance(evidence, dict):
                continue
            path = str(evidence.get("path", ""))
            if path.endswith("canonical-normal"):
                return path
    return None


def _validate_no_promotion_policy(
    decision: Mapping[str, Any],
    failures: list[str],
) -> None:
    """Validate the decision fields that prevent accidental promotion."""
    policy = decision.get("evidence_policy", {})
    if not isinstance(policy, dict):
        failures.append("evidence_policy must be a mapping")
        return

    expected_false_fields = (
        "canonical_artifact_path_changed",
        "publication_manifest_changed",
        "artifact_building_path_changed",
        "paper_results_notebook_promotes_candidate",
    )
    if policy.get("promoted_candidate") is not None:
        failures.append("no-promotion decision must not name a promoted candidate")
    for field in expected_false_fields:
        if policy.get(field) is not False:
            failures.append(f"evidence_policy.{field} must be false")

    unchanged_defaults = decision.get("unchanged_defaults", {})
    if not isinstance(unchanged_defaults, dict):
        failures.append("unchanged_defaults must be a mapping")
        return
    for field in ("package_defaults", "benchmark_configs"):
        if unchanged_defaults.get(field) is not True:
            failures.append(f"unchanged_defaults.{field} must be true")


def _validate_review_summary(
    decision: Mapping[str, Any],
    failures: list[str],
) -> None:
    """Validate that the no-promotion rationale is self-contained."""
    summary = decision.get("review_summary", {})
    if not isinstance(summary, dict):
        failures.append("review_summary must be a mapping")
        return

    baseline = summary.get("baseline", {})
    if not isinstance(baseline, dict):
        failures.append("review_summary.baseline must be a mapping")
    else:
        if baseline.get("run_name") != _BASELINE_RUN_NAME:
            failures.append("review_summary.baseline must name the fixed 1.0 baseline")
        if "coverage_mae" not in baseline:
            failures.append("review_summary.baseline must record coverage_mae")
        if "intercept_coverage_mae" not in baseline:
            failures.append(
                "review_summary.baseline must record intercept_coverage_mae"
            )

    rejected = summary.get("rejected_candidates", [])
    if not isinstance(rejected, list) or len(rejected) != 4:
        failures.append("review_summary must record four rejected sweep candidates")
        return
    rejected_names = {
        str(candidate.get("run_name", ""))
        for candidate in rejected
        if isinstance(candidate, dict)
    }
    expected_names = {
        "prior-sweep-normal-fixed-0p3",
        "prior-sweep-normal-fixed-3p0",
        "prior-sweep-normal-empirical-bayes",
        "prior-sweep-normal-hierarchical-ig",
    }
    if rejected_names != expected_names:
        failures.append("review_summary rejected candidates do not match the sweep")
    for candidate in rejected:
        if not isinstance(candidate, dict):
            failures.append("each rejected candidate must be a mapping")
            continue
        if "coverage_mae" not in candidate:
            failures.append("each rejected candidate must record coverage_mae")
        reason = str(candidate.get("reason", "")).strip()
        if not reason:
            failures.append("each rejected candidate must record a rejection reason")
        if candidate.get("run_name") == "prior-sweep-normal-fixed-3p0":
            normalized_reason = reason.lower()
            if "intercept coverage" not in normalized_reason:
                failures.append(
                    "review_summary must record the fixed 3.0 "
                    "intercept-coverage rejection"
                )


def _summarize_candidate(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    runs_root: Path,
) -> dict[str, Any]:
    """Summarize one completed sweep candidate from public artifacts."""
    root = _run_root(config, runs_root)
    calibration_rows = _read_metric_rows(
        _required_artifact(root / "metrics" / "calibration.csv")
    )
    intercept_rows = _read_metric_rows(
        _required_artifact(root / "metrics" / "intercept_coverage.csv")
    )
    prior_scale = _read_json(_required_artifact(root / "metrics" / "prior_scale.json"))
    training = _read_json(_required_artifact(root / "metrics" / "training.json"))
    run = _read_json(_required_artifact(root / "run.json"))
    recovery_figure = _required_artifact(root / "figures" / "recovery.pdf")
    calibration_figure = _required_artifact(root / "figures" / "calibration.pdf")
    coverage = _coverage_summary(calibration_rows)
    intercept = _coverage_summary(intercept_rows)

    return {
        "config_file": str(config_path),
        "run_name": str(config["artifacts"]["run_name"]),
        "run_root": str(root),
        "seed": int(run["seed"]),
        "smoke": bool(run["smoke"]),
        "coverage": {
            **coverage,
            "rows": calibration_rows,
        },
        "intercept_coverage": {
            **intercept,
            "rows": intercept_rows,
        },
        "training": training,
        "prior_scale": prior_scale,
        "artifacts": {
            "recovery_figure": str(recovery_figure),
            "calibration_figure": str(calibration_figure),
        },
    }


def _rank(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates by lower mean absolute coverage error."""
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["coverage"]["mean_absolute_error"]),
            str(row["run_name"]),
        ),
    )
    return [dict(row, rank=index) for index, row in enumerate(ranked, start=1)]


def _decision(decision: str, notes: str, run_names: set[str]) -> dict[str, str | None]:
    """Parse and validate the human screening decision."""
    if decision == "undecided":
        return {"status": "undecided", "candidate": None, "notes": notes}
    if decision == "no-candidate":
        return {"status": "no_candidate", "candidate": None, "notes": notes}
    prefix = "nominate:"
    if decision.startswith(prefix):
        candidate = decision.removeprefix(prefix)
        if candidate not in run_names:
            raise ValueError(f"nominated candidate is not in the sweep: {candidate}")
        return {"status": "nominate", "candidate": candidate, "notes": notes}
    raise ValueError(
        "--decision must be 'undecided', 'no-candidate', or 'nominate:<run_name>'"
    )


def _csv_row(rank: int, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one candidate summary for spreadsheet inspection."""
    training = candidate["training"]
    return {
        "rank": rank,
        "config_file": candidate["config_file"],
        "run_name": candidate["run_name"],
        "seed": candidate["seed"],
        "smoke": candidate["smoke"],
        "coverage_mae": candidate["coverage"]["mean_absolute_error"],
        "coverage_max_abs_error": candidate["coverage"]["max_absolute_error"],
        "intercept_coverage_mae": candidate["intercept_coverage"][
            "mean_absolute_error"
        ],
        "final_loss": training["final_loss"],
        "final_nll": training["final_nll"],
        "final_kl": training["final_kl"],
        "prior_scale_mode": _prior_scale_mode(candidate["prior_scale"]),
        "recovery_figure": candidate["artifacts"]["recovery_figure"],
        "calibration_figure": candidate["artifacts"]["calibration_figure"],
    }


def _write_review_csv(path: Path, ranking: Sequence[Mapping[str, Any]]) -> None:
    """Write the compact machine-readable review table."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, candidate in enumerate(ranking, start=1):
            writer.writerow(_csv_row(index, candidate))


if __name__ == "__main__":
    main()
