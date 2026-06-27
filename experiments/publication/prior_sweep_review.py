"""Prior/smoothness sweep review helpers (PRD-0004, GitHub #160)."""

from __future__ import annotations

import copy
import csv
import json
import shlex
from pathlib import Path
from typing import Any

import yaml

CONFIRMATORY_SEEDS = (9801, 9811, 9821)


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
        prior_state = json.loads(prior_path.read_text(encoding="utf-8"))
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
    abs_errors: list[float] = []
    empirical_coverages: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            nominal = float(row["nominal"])
            empirical = float(row["empirical_coverage"])
            abs_errors.append(abs(empirical - nominal))
            empirical_coverages.append(empirical)
    return {
        "mean_abs_error": sum(abs_errors) / len(abs_errors),
        "max_abs_error": max(abs_errors),
        "min_empirical_coverage": min(empirical_coverages),
    }
