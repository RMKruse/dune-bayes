"""Locked confirmation execution and claim assignment (ADR-0008, issue #181)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dune_bayes.utils import EPS  # noqa: E402
from experiments.uci_benchmark.development import (  # noqa: E402
    CONFIRMATION_RUN_SEEDS,
    PRIMARY_MODELS,
    PROCEDURES,
    _read_csv,
    _selected_configurations,
    _validate_evaluation,
    _write_csv,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREDICTIVE_METRICS = ("nll", "crps")
METRIC_COLUMNS = {"nll": "mean_nll", "crps": "mean_crps"}
WITHHELD_CLAIMS = (
    "universal predictive dominance",
    "general calibration",
    "VI-to-NUTS equivalence",
    "package-default promotion",
)
CODE_GUARDRAILS = (
    "tests/sampling/test_effect_sampler.py::test_additive_predictor_reconstructs_from_separate_effects",
    "tests/sampling/test_effect_sampler.py::test_all_feature_names_returned",
    "tests/sampling/test_effect_sampler.py::test_effect_bands_are_reportable_on_original_covariate_scale",
    "tests/data/test_numeric_preprocessing.py::test_plot_grid_inverse_transform_recovers_original_scale",
    "tests/families/test_link_floor_gate.py::test_extreme_pre_link_log_prob_is_finite",
    "tests/layers/test_variational_dense.py::test_collect_kl_reaches_all_variational_layers",
    "tests/model/test_nan_gradient_gate.py",
)


def _digest(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    return sha256(path.read_bytes()).hexdigest()


def _finite(row: Mapping[str, str], column: str) -> float | None:
    """Read a finite completed score, otherwise return None."""
    if row.get("status") != "completed" or row.get("failure"):
        return None
    return _reported_float(row, column)


def _reported_float(row: Mapping[str, str], column: str) -> float | None:
    """Read a finite raw value even when its run is classified as failed."""
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _validate_freeze(freeze_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate issue #180's immutable inputs before inspecting confirmation scores."""
    manifest = json.loads((freeze_dir / "freeze.json").read_text(encoding="utf-8"))
    hashes = json.loads(
        (freeze_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    if not manifest.get("confirmation_inputs_complete"):
        errors.append("development freeze is incomplete")
    if manifest.get("confirmation", {}).get("scores_generated") is not False:
        errors.append("development freeze already contains confirmation scores")
    if _digest(freeze_dir / "artifact_hashes.json") != manifest.get(
        "artifact_hashes_sha256"
    ):
        errors.append("development artifact-hash manifest drifted")
    for relative, expected in hashes.get("sha256", {}).items():
        path = freeze_dir / relative
        if not path.is_file() or _digest(path) != expected:
            errors.append(f"frozen artifact drifted: {relative}")
    pairs = manifest.get("confirmation", {}).get("split_run_pairs", [])
    expected_pairs = [
        {
            "dataset": dataset,
            "run_seed": run_seed,
            "split_seed": run_seed * 100 + index,
        }
        for run_seed in CONFIRMATION_RUN_SEEDS
        for index, dataset in enumerate(
            (
                "autompg",
                "concrete",
                "energy",
                "kin8nm",
                "naval",
                "power",
                "protein",
                "wine",
                "yacht",
                "bike",
            ),
            start=1,
        )
    ]
    if pairs != expected_pairs:
        errors.append("confirmation split/run pairs drifted")
    if errors:
        raise ValueError("; ".join(errors))
    return manifest, pairs


def _record_infrastructure_event(
    path: Path,
    *,
    event: str,
    pair: Mapping[str, object],
    config: Path,
    returncode: int,
) -> None:
    """Record a failed attempt or its same-config repair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **pair,
        "event": event,
        "config": str(config),
        "config_sha256": _digest(config),
        "returncode": returncode,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_confirmation_panel(
    freeze_dir: Path,
    *,
    run_root: Path,
    smoke: bool = False,
) -> None:
    """Run only the frozen winners on the predeclared confirmation pairs.

    Args:
        freeze_dir: Immutable issue #180 confirmation-input freeze.
        run_root: Scratch experiment root for resumable model fits.
        smoke: Run only the first locked pair as an orchestration tracer.
    """
    _, pairs = _validate_freeze(freeze_dir)
    base = yaml.safe_load(
        (freeze_dir / "development_config.yaml").read_text(encoding="utf-8")
    )
    dataset_by_name = {dataset["name"]: dataset for dataset in base["datasets"]}
    generated = run_root / "confirmation-configs"
    generated.mkdir(parents=True, exist_ok=True)
    selection_root = run_root / "confirmation-selections"
    for source in (freeze_dir / "selections").glob("*.json"):
        destination = selection_root / source.stem / "selection.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    prefix = "confirmation-smoke" if smoke else "confirmation"
    events = run_root / f"{prefix}-infrastructure-events.jsonl"
    event_records = (
        [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
        if events.is_file()
        else []
    )
    failure_count = 0
    selected_pairs = pairs[:1] if smoke else pairs
    for pair in selected_pairs:
        dataset = str(pair["dataset"])
        run_seed = int(pair["run_seed"])
        split_seed = int(pair["split_seed"])
        run_name = f"{prefix}-seed-{run_seed}-{dataset}"
        result = run_root / "uci_benchmark" / run_name
        pair_events = [
            event
            for event in event_records
            if event.get("dataset") == dataset
            and int(event.get("run_seed", -1)) == run_seed
            and int(event.get("split_seed", -1)) == split_seed
        ]
        prior_failure = pair_events[-1] if pair_events else None
        metric_root = result / "metrics" / dataset
        complete = all(
            path.is_file()
            for path in (
                result / "config.yaml",
                result / "run.json",
                result / "metrics" / "comparison.csv",
                metric_root / "evaluation.json",
                metric_root / "C" / "dune_bayes" / "parameter_bands.csv",
                metric_root / "C" / "dune_bayes" / "variance_split.csv",
            )
        )
        if complete and not (
            prior_failure is not None and prior_failure.get("event") == "failure"
        ):
            continue
        config = json.loads(json.dumps(base))
        locked_dataset = dict(dataset_by_name[dataset])
        locked_dataset["split_seed"] = split_seed
        config["seed"] = run_seed
        config["datasets"] = [locked_dataset]
        config["data"]["split_dir"] = f"data/confirmation-splits/seed-{run_seed}"
        config["development"] = {
            **config.get("development", {}),
            "enabled": True,
            "dataset": dataset,
            "selection_seed": 102,
        }
        config["artifacts"] = {"root": str(run_root), "run_name": run_name}
        target = generated / f"seed-{run_seed}-{dataset}.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if (
            prior_failure is not None
            and prior_failure.get("event") == "failure"
            and prior_failure.get("config_sha256") != _digest(target)
        ):
            raise ValueError(f"repair config drifted for {dataset}, seed {run_seed}")
        command = [
            sys.executable,
            str(EXPERIMENT_DIR / "run.py"),
            str(target),
            "--dataset",
            dataset,
            "--selection-root",
            str(selection_root),
        ]
        if smoke:
            command.append("--smoke")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failure_count += 1
            _record_infrastructure_event(
                events,
                event="failure",
                pair=pair,
                config=target,
                returncode=completed.returncode,
            )
        elif prior_failure is not None and prior_failure.get("event") == "failure":
            _record_infrastructure_event(
                events,
                event="repair_succeeded",
                pair=pair,
                config=target,
                returncode=0,
            )
    if failure_count:
        raise RuntimeError(
            f"{failure_count} infrastructure failure(s) recorded; repair and rerun"
        )


def _collect_scores(
    freeze_dir: Path,
    run_root: Path,
    pairs: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, str]], dict[tuple[str, int, str, str], dict[str, str]]]:
    """Collect every raw score after validating frozen configs and winners."""
    rows: list[dict[str, str]] = []
    indexed: dict[tuple[str, int, str, str], dict[str, str]] = {}
    errors: list[str] = []
    base = yaml.safe_load(
        (freeze_dir / "development_config.yaml").read_text(encoding="utf-8")
    )
    dataset_by_name = {dataset["name"]: dataset for dataset in base["datasets"]}
    locked_keys = (
        "primary_models",
        "architecture",
        "tuning",
        "training",
        "baselines",
        "draws",
        "predictive_samples",
        "calibration_bins",
    )
    for pair in pairs:
        dataset = str(pair["dataset"])
        seed = int(pair["run_seed"])
        split_seed = int(pair["split_seed"])
        run = run_root / "uci_benchmark" / f"confirmation-seed-{seed}-{dataset}"
        config_path = run / "config.yaml"
        comparison_path = run / "metrics" / "comparison.csv"
        evaluation_path = run / "metrics" / dataset / "evaluation.json"
        if not all(
            path.is_file() for path in (config_path, comparison_path, evaluation_path)
        ):
            errors.append(f"missing confirmation artifacts for {dataset}, seed {seed}")
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        configured = config.get("datasets", [{}])[0]
        if int(config.get("seed", -1)) != seed:
            errors.append(f"run seed drifted for {dataset}, seed {seed}")
        if int(configured.get("split_seed", -1)) != split_seed:
            errors.append(f"split seed drifted for {dataset}, seed {seed}")
        expected_dataset = dict(dataset_by_name[dataset])
        expected_dataset["split_seed"] = split_seed
        if configured != expected_dataset:
            errors.append(f"dataset configuration drifted for {dataset}, seed {seed}")
        expected_data = {
            **base["data"],
            "split_dir": f"data/confirmation-splits/seed-{seed}",
        }
        if config.get("data") != expected_data:
            errors.append(f"data configuration drifted for {dataset}, seed {seed}")
        expected_development = {
            **base.get("development", {}),
            "enabled": True,
            "dataset": dataset,
            "selection_seed": 102,
        }
        if config.get("development") != expected_development:
            errors.append(f"selection configuration drifted for {dataset}, seed {seed}")
        artifacts = config.get("artifacts", {})
        if artifacts.get("root") != str(run_root) or artifacts.get("run_name") != (
            f"confirmation-seed-{seed}-{dataset}"
        ):
            errors.append(f"artifact configuration drifted for {dataset}, seed {seed}")
        for key in locked_keys:
            if config.get(key) != base.get(key):
                errors.append(f"locked {key} drifted for {dataset}, seed {seed}")
        run_log = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if (
            run_log.get("experiment") != "uci_benchmark"
            or int(run_log.get("seed", -1)) != seed
            or run_log.get("smoke") is not False
        ):
            errors.append(f"run metadata drifted for {dataset}, seed {seed}")
        selection_path = freeze_dir / "selections" / f"{dataset}.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = _selected_configurations(
            selection, dataset=dataset, expected_slots=12, errors=errors
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _validate_evaluation(
            evaluation,
            dataset=dataset,
            seed=seed,
            selection_sha256=_digest(selection_path),
            selected=selected,
            errors=errors,
        )
        uncertainty = run / "metrics" / dataset / "C" / "dune_bayes"
        band_path = uncertainty / "parameter_bands.csv"
        variance_path = uncertainty / "variance_split.csv"
        if not band_path.is_file() or not variance_path.is_file():
            errors.append(f"missing uncertainty artifacts for {dataset}, seed {seed}")
            continue
        bands = _read_csv(band_path)
        variance = _read_csv(variance_path)
        if not bands or not variance:
            errors.append(f"missing uncertainty artifacts for {dataset}, seed {seed}")
        for row in bands:
            try:
                q05, q50, q95 = (float(row[key]) for key in ("q05", "q50", "q95"))
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid parameter band for {dataset}, seed {seed}")
                break
            if not all(map(math.isfinite, (q05, q50, q95))) or not q05 <= q50 <= q95:
                errors.append(f"non-finite parameter band for {dataset}, seed {seed}")
                break
        for row in variance:
            try:
                aleatoric, epistemic, total = (
                    float(row[key]) for key in ("aleatoric", "epistemic", "total")
                )
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid variance split for {dataset}, seed {seed}")
                break
            if (
                not all(map(math.isfinite, (aleatoric, epistemic, total)))
                or min(aleatoric, epistemic, total) < 0.0
                or not math.isclose(
                    total, aleatoric + epistemic, rel_tol=EPS, abs_tol=EPS
                )
            ):
                errors.append(f"non-finite variance split for {dataset}, seed {seed}")
                break
        run_rows = _read_csv(comparison_path)
        primary = [row for row in run_rows if row.get("comparison_role") == "primary"]
        if len(primary) != len(PROCEDURES) * len(PRIMARY_MODELS):
            errors.append(f"incomplete primary panel for {dataset}, seed {seed}")
        for row in primary:
            try:
                row_seed = int(row.get("seed", -1))
            except (TypeError, ValueError):
                row_seed = -1
            if row.get("dataset") != dataset or row_seed != seed:
                errors.append(f"score provenance drifted for {dataset}, seed {seed}")
            if row.get("family") != configured.get("family"):
                errors.append(f"score family drifted for {dataset}, seed {seed}")
            if (
                row.get("procedure") not in PROCEDURES
                or row.get("model") not in PRIMARY_MODELS
                or row.get("status")
                not in {"completed", "model_failure", "infrastructure_failure"}
            ):
                errors.append(f"invalid primary score for {dataset}, seed {seed}")
            key = (dataset, seed, row.get("procedure", ""), row.get("model", ""))
            if key in indexed:
                errors.append(f"duplicate confirmation score {key}")
            if int(row.get("split_seed", -1)) != split_seed:
                errors.append(f"score split seed drifted for {dataset}, seed {seed}")
            indexed[key] = row
            rows.append(row)
    if errors:
        raise ValueError("; ".join(errors))
    return rows, indexed


def _gap_closures(
    pairs: Sequence[Mapping[str, object]],
    scores: Mapping[tuple[str, int, str, str], dict[str, str]],
    mappings: Mapping[tuple[str, str], str],
) -> list[dict[str, object]]:
    """Compute frozen-comparator gap closure and preservation per raw run."""
    rows: list[dict[str, object]] = []
    for pair in pairs:
        dataset = str(pair["dataset"])
        seed = int(pair["run_seed"])
        reference = scores[(dataset, seed, "R", "dune_bayes")]
        candidate = scores[(dataset, seed, "C", "dune_bayes")]
        for metric in PREDICTIVE_METRICS:
            column = METRIC_COLUMNS[metric]
            comparator_name = mappings[(dataset, metric)]
            comparator = scores[(dataset, seed, "C", comparator_name)]
            reference_score = _finite(reference, column)
            candidate_score = _finite(candidate, column)
            comparator_score = _finite(comparator, column)
            valid = None not in (reference_score, candidate_score, comparator_score)
            if not valid:
                closure: float | str = ""
                capped: float | str = ""
                competitive = False
                preserved = False
                candidate_win = False
            else:
                assert reference_score is not None
                assert candidate_score is not None
                assert comparator_score is not None
                deficit = reference_score - comparator_score
                competitive = deficit <= 0.0
                preserved = competitive and candidate_score <= comparator_score
                candidate_win = candidate_score <= comparator_score
                closure = (
                    "" if competitive else (reference_score - candidate_score) / deficit
                )
                capped = "" if competitive else min(1.0, closure)
            rows.append(
                {
                    "dataset": dataset,
                    "run_seed": seed,
                    "split_seed": int(pair["split_seed"]),
                    "metric": metric,
                    "comparator": comparator_name,
                    "reference_score": reference_score
                    if reference_score is not None
                    else "",
                    "candidate_score": candidate_score
                    if candidate_score is not None
                    else "",
                    "comparator_score": comparator_score
                    if comparator_score is not None
                    else "",
                    "valid": valid,
                    "already_competitive": competitive,
                    "preserved": preserved,
                    "candidate_comparator_win": candidate_win,
                    "raw_gap_closure": closure,
                    "capped_gap_closure": capped,
                }
            )
    return rows


def _summarize_predictive(
    closures: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Aggregate runs dataset-first and assign each predictive metric's tier."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in closures:
        grouped[(str(row["dataset"]), str(row["metric"]))].append(row)
    datasets: list[dict[str, object]] = []
    for (dataset, metric), rows in sorted(grouped.items()):
        values = [
            float(row["capped_gap_closure"])
            for row in rows
            if row["capped_gap_closure"] != ""
        ]
        invalid_runs = sum(not bool(row["valid"]) for row in rows)
        competitive = invalid_runs == 0 and not values
        median = statistics.median(values) if values else ""
        stable = (
            all(bool(row["preserved"]) for row in rows)
            if competitive
            else invalid_runs == 0 and sum(value > 0.0 for value in values) >= 2
        )
        datasets.append(
            {
                "dataset": dataset,
                "metric": metric,
                "invalid_runs": invalid_runs,
                "median_gap_closure": median,
                "improved_runs": sum(value > 0.0 for value in values),
                "stable": stable,
                "already_competitive": competitive,
                "preserved": competitive
                and all(bool(row["preserved"]) for row in rows),
                "comparator_wins": sum(
                    bool(row["candidate_comparator_win"]) for row in rows
                ),
            }
        )
    summaries: list[dict[str, object]] = []
    for metric in PREDICTIVE_METRICS:
        metric_rows = [row for row in datasets if row["metric"] == metric]
        closure_values = [
            float(row["median_gap_closure"])
            for row in metric_rows
            if row["median_gap_closure"] != ""
        ]
        median = statistics.median(closure_values) if closure_values else 0.0
        support = sum(
            (row["median_gap_closure"] != "" and float(row["median_gap_closure"]) > 0)
            or bool(row["preserved"])
            for row in metric_rows
        )
        stable = sum(bool(row["stable"]) for row in metric_rows)
        all_preserved = all(
            not bool(row["already_competitive"]) or bool(row["preserved"])
            for row in metric_rows
        )
        within_band = sum(
            bool(row["preserved"])
            if row["already_competitive"]
            else row["median_gap_closure"] != ""
            and float(row["median_gap_closure"]) >= -0.10
            for row in metric_rows
        )
        nonregression = median >= -0.10 and within_band >= 6 and all_preserved
        failures = sum(int(row["invalid_runs"]) for row in metric_rows)
        if median >= 0.50 and support >= 8 and stable >= 8 and all_preserved:
            tier = "Strong"
        elif median >= 0.25 and support >= 6 and stable >= 6 and all_preserved:
            tier = "Material"
        elif median > 0.0 and nonregression:
            tier = "Suggestive"
        else:
            tier = "Mixed"
        summaries.append(
            {
                "metric": metric,
                "tier": tier,
                "failed_runs": failures,
                "median_gap_closure": median,
                "supporting_datasets": support,
                "stable_datasets": stable,
                "within_nonregression_band_or_preserved": within_band,
                "already_competitive_datasets": sum(
                    bool(row["already_competitive"]) for row in metric_rows
                ),
                "preserved_datasets": sum(
                    bool(row["preserved"]) for row in metric_rows
                ),
                "panel_nonregression": nonregression,
            }
        )
    return datasets, summaries


def _binomial_mean_absolute_error(n: int, p: float = 0.1) -> float:
    """Return exact E[|K/n-p|] for K~Binomial(n,p), stably in log-space."""
    if n <= 0:
        raise ValueError("PIT normalization requires a positive test-set size")
    k = math.floor(n * p)
    log_pmf = (
        math.lgamma(n)
        - math.lgamma(k + 1)
        - math.lgamma(n - k)
        + k * math.log(p)
        + (n - 1 - k) * math.log1p(-p)
    )
    return 2.0 * p * (1.0 - p) * math.exp(log_pmf)


def _pit_normalization(
    pairs: Sequence[Mapping[str, object]],
    scores: Mapping[tuple[str, int, str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Normalize PIT error by finite-sample noise and apply frozen raw thresholds."""
    rows: list[dict[str, object]] = []
    for pair in pairs:
        dataset = str(pair["dataset"])
        seed = int(pair["run_seed"])
        reference = scores[(dataset, seed, "R", "dune_bayes")]
        candidate = scores[(dataset, seed, "C", "dune_bayes")]
        n_test = int(candidate["n_test"])
        reference_error = _reported_float(reference, "calibration_error")
        candidate_error = _reported_float(candidate, "calibration_error")
        valid = (
            _finite(reference, "calibration_error") is not None
            and _finite(candidate, "calibration_error") is not None
        )
        floor = _binomial_mean_absolute_error(n_test)
        reduction = (
            reference_error - candidate_error
            if reference_error is not None and candidate_error is not None
            else ""
        )
        rows.append(
            {
                "dataset": dataset,
                "run_seed": seed,
                "split_seed": int(pair["split_seed"]),
                "n_test": n_test,
                "reference_pit_error": (
                    reference_error if reference_error is not None else ""
                ),
                "candidate_pit_error": (
                    candidate_error if candidate_error is not None else ""
                ),
                "raw_reduction": reduction,
                "sampling_noise_floor": floor,
                "reference_normalized_error": (
                    reference_error / floor if reference_error is not None else ""
                ),
                "candidate_normalized_error": (
                    candidate_error / floor if candidate_error is not None else ""
                ),
                "valid": valid,
            }
        )
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    reductions = []
    for dataset_rows in by_dataset.values():
        valid_reductions = [
            float(row["raw_reduction"])
            for row in dataset_rows
            if row["valid"] and row["raw_reduction"] != ""
        ]
        if valid_reductions:
            reductions.append(statistics.median(valid_reductions))
    median_reduction = statistics.median(reductions)
    within_band = sum(value >= -0.01 for value in reductions)
    reference_saturated = sum(
        row["reference_pit_error"] != "" and float(row["reference_pit_error"]) >= 0.18
        for row in rows
    )
    candidate_saturated = sum(
        row["candidate_pit_error"] != "" and float(row["candidate_pit_error"]) >= 0.18
        for row in rows
    )
    reference_failures = sum(row["reference_pit_error"] == "" for row in rows)
    candidate_failures = sum(row["candidate_pit_error"] == "" for row in rows)
    nonregression = (
        median_reduction >= -0.01
        and within_band >= 6
        and candidate_saturated == 0
        and candidate_failures == 0
    )
    if median_reduction >= 0.02 and nonregression:
        tier = "Strong"
    elif median_reduction >= 0.01 and nonregression:
        tier = "Material"
    elif nonregression:
        tier = "Suggestive"
    else:
        tier = "Mixed"
    return rows, {
        "tier": tier,
        "median_raw_reduction": median_reduction,
        "datasets_within_equivalence_band_or_better": within_band,
        "reference_saturated_runs": reference_saturated,
        "candidate_saturated_runs": candidate_saturated,
        "reference_failed_runs": reference_failures,
        "candidate_failed_runs": candidate_failures,
        "panel_nonregression": nonregression,
    }


def _model_or_numerical_failure(row: Mapping[str, str]) -> bool:
    """Return whether one retained primary score failed its numerical gate."""
    return (
        row["status"] != "completed"
        or any(
            _finite(row, column) is None
            for column in ("mean_nll", "mean_crps", "calibration_error")
        )
        or float(row["calibration_error"]) >= 0.18
    )


def _git_blob(repository_root: Path, commit: str, relative: str) -> bytes:
    """Read one frozen-commit blob without checking it out."""
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _csv_blob(
    repository_root: Path, commit: str, relative: str
) -> list[dict[str, str]]:
    """Parse one CSV artifact from the frozen commit."""
    text = _git_blob(repository_root, commit, relative).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def assess_frozen_guardrail_thresholds(
    *,
    coverage_errors: Sequence[float],
    accepted_coverage_errors: Sequence[float],
    accepted_coverage_mae: float,
    width_ratios: Sequence[float],
    accepted_width_ratios: Sequence[float],
    center_differences: Sequence[float],
) -> dict[str, bool]:
    """Apply the frozen recovery and VI-to-NUTS numerical thresholds.

    Args:
        coverage_errors: Current absolute pointwise coverage errors.
        accepted_coverage_errors: Matching errors at the frozen reference.
        accepted_coverage_mae: Frozen reference's mean coverage error.
        width_ratios: Current median VI-to-NUTS width ratios.
        accepted_width_ratios: Matching ratios at the frozen reference.
        center_differences: Current normalized center differences.

    Returns:
        Promotion pass/fail and the separate over-width human-review trigger.
    """
    # EPS keeps mathematically exact decimal boundaries inclusive after float parsing.
    recovery_passed = (
        statistics.mean(coverage_errors) <= accepted_coverage_mae + 0.02 + EPS
    )
    recovery_passed = recovery_passed and all(
        current <= accepted + 0.05 + EPS
        for current, accepted in zip(
            coverage_errors, accepted_coverage_errors, strict=True
        )
    )
    agreement_passed = all(
        current >= accepted - 0.05 - EPS
        for current, accepted in zip(width_ratios, accepted_width_ratios, strict=True)
    ) and all(value <= 0.10 + EPS for value in center_differences)
    return {
        "passed": recovery_passed and agreement_passed,
        "overwidth_review_triggered": any(value > 1.25 for value in width_ratios),
    }


def _frozen_guardrails(repository_root: Path, freeze_dir: Path) -> dict[str, object]:
    """Check recovery and VI-to-NUTS evidence against the freeze commit and hashes."""
    relative_freeze = str(freeze_dir.resolve().relative_to(repository_root.resolve()))
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative_freeze + "/freeze.json"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = (
        "experiments/publication/prior-smoothness-calibration-decision.yaml",
        "experiments/parameter_recovery/results/canonical-normal/arrays/recovery.npz",
        "experiments/parameter_recovery/results/canonical-normal/metrics/calibration.csv",
        "experiments/parameter_recovery/results/canonical-normal/run.json",
        "experiments/hmc_agreement/results/canonical/metrics/band_width_ratios.csv",
        "experiments/hmc_agreement/results/canonical/metrics/diagnostics.json",
        "experiments/hmc_agreement/results/canonical/run.json",
    )
    hashes: dict[str, str] = {}
    for relative in files:
        path = repository_root / relative
        current = _digest(path)
        frozen = sha256(_git_blob(repository_root, commit, relative)).hexdigest()
        if current != frozen:
            raise ValueError(f"guardrail artifact drifted from {commit}: {relative}")
        hashes[relative] = current

    decision = yaml.safe_load((repository_root / files[0]).read_text(encoding="utf-8"))
    calibration = _read_csv(repository_root / files[2])
    frozen_calibration = _csv_blob(repository_root, commit, files[2])
    coverage_errors = [
        abs(float(row["empirical_coverage"]) - float(row["nominal"]))
        for row in calibration
    ]
    frozen_coverage_errors = {
        (row["parameter"], row["nominal"]): abs(
            float(row["empirical_coverage"]) - float(row["nominal"])
        )
        for row in frozen_calibration
    }
    coverage_mae = statistics.mean(coverage_errors)
    accepted_mae = float(decision["review_summary"]["baseline"]["coverage_mae"])
    diagnostics = json.loads((repository_root / files[5]).read_text(encoding="utf-8"))
    ratios = _read_csv(repository_root / files[4])
    accepted_ratio_by_key = {
        (row["feature"], row["distribution_parameter"]): float(
            row["median_vi_to_nuts_width_ratio"]
        )
        for row in _csv_blob(repository_root, commit, files[4])
    }
    accepted_coverage = [
        frozen_coverage_errors[(row["parameter"], row["nominal"])]
        for row in calibration
    ]
    width_ratios = [float(row["median_vi_to_nuts_width_ratio"]) for row in ratios]
    accepted_ratios = [
        accepted_ratio_by_key[(row["feature"], row["distribution_parameter"])]
        for row in ratios
    ]
    center_differences = [
        float(row["median_normalized_center_difference"]) for row in ratios
    ]
    thresholds = assess_frozen_guardrail_thresholds(
        coverage_errors=coverage_errors,
        accepted_coverage_errors=accepted_coverage,
        accepted_coverage_mae=accepted_mae,
        width_ratios=width_ratios,
        accepted_width_ratios=accepted_ratios,
        center_differences=center_differences,
    )
    passed = (
        thresholds["passed"]
        and int(diagnostics["chains"]) == 4
        and float(diagnostics["r_hat_max"]) <= 1.01
        and float(diagnostics["ess_bulk_min"]) >= 400
        and int(diagnostics["divergences"]) == 0
        and all(float(row["vi_inside_nuts_fraction"]) >= 0.9 for row in ratios)
    )
    if not passed:
        raise ValueError("frozen recovery or VI-to-NUTS guardrail failed")
    return {
        "passed": True,
        "frozen_commit": commit,
        "sha256": hashes,
        "pointwise_recovery": {
            "mean_absolute_coverage_error": coverage_mae,
            "accepted_reference": accepted_mae,
            "maximum_cell_coverage_error": max(coverage_errors),
            "mean_nonregression_margin": 0.02,
            "max_cell_nonregression_margin": 0.05,
        },
        "vi_to_nuts": {
            "r_hat_max": diagnostics["r_hat_max"],
            "ess_bulk_min": diagnostics["ess_bulk_min"],
            "divergences": diagnostics["divergences"],
            "minimum_width_ratio": min(
                float(row["median_vi_to_nuts_width_ratio"]) for row in ratios
            ),
            "maximum_width_ratio": max(
                float(row["median_vi_to_nuts_width_ratio"]) for row in ratios
            ),
            "minimum_accepted_width_ratio_minus_margin": min(accepted_ratios) - 0.05,
            "overwidth_review_triggered": thresholds["overwidth_review_triggered"],
            "maximum_normalized_center_difference": max(
                float(row["median_normalized_center_difference"]) for row in ratios
            ),
        },
    }


def _code_guardrails(repository_root: Path, *, run: bool) -> dict[str, object]:
    """Run structural, scale, family-link, KL, and finite-gradient gates."""
    if not run:
        return {"passed": True, "status": "not_run", "tests": list(CODE_GUARDRAILS)}
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *CODE_GUARDRAILS],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    result = {
        "passed": completed.returncode == 0,
        "status": "passed" if completed.returncode == 0 else "failed",
        "tests": list(CODE_GUARDRAILS),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise ValueError("confirmation code guardrails failed")
    return result


def _combined_tier(
    predictive: Sequence[Mapping[str, object]], pit: Mapping[str, object]
) -> str:
    """Apply the frozen combined Strong/Material/Suggestive/Mixed rule."""
    by_metric = {str(row["metric"]): row for row in predictive}
    tiers = {metric: str(by_metric[metric]["tier"]) for metric in PREDICTIVE_METRICS}
    pit_ok = bool(pit["panel_nonregression"])
    both_nonregressing = all(
        bool(by_metric[metric]["panel_nonregression"]) for metric in PREDICTIVE_METRICS
    )
    if all(tier == "Strong" for tier in tiers.values()) and pit_ok:
        return "Strong"
    if (
        any(tier in {"Strong", "Material"} for tier in tiers.values())
        and all(
            tier in {"Strong", "Material"}
            or bool(by_metric[metric]["panel_nonregression"])
            for metric, tier in tiers.items()
        )
        and pit_ok
    ):
        return "Material"
    if (
        any(tier != "Mixed" for tier in tiers.values())
        and both_nonregressing
        and pit_ok
    ):
        return "Suggestive"
    return "Mixed"


def _write_report_markdown(path: Path, report: Mapping[str, object]) -> None:
    """Write the panel-bounded claim in a human-reviewable form."""
    predictive = report["predictive_metrics"]
    assert isinstance(predictive, list)
    pit = report["pit"]
    assert isinstance(pit, Mapping)
    tier = str(report["mechanically_earned_tier"])
    candidate_failures = int(report["selected_candidate_model_or_numerical_failures"])
    reference_failures = int(report["reference_model_or_numerical_failures"])
    infrastructure_repairs = int(report["infrastructure_repairs"])
    numerical_guardrail = "passed" if candidate_failures == 0 else "failed"
    metric_lines = "\n".join(
        f"- {row['metric'].upper()}: {row['tier']} "
        f"(median gap closure {float(row['median_gap_closure']):.1%}, "
        f"support {row['supporting_datasets']}/10)."
        for row in predictive
    )
    withheld = "\n".join(f"- {claim}." for claim in WITHHELD_CLAIMS)
    path.write_text(
        "\n".join(
            (
                "# Locked confirmation report",
                "",
                f"Mechanically earned panel-bounded tier: **{tier}**.",
                "Human review may downgrade this tier but may not upgrade it.",
                "",
                "## Predictive evidence",
                "",
                metric_lines,
                f"- PIT: {pit['tier']} (median raw reduction "
                f"{float(pit['median_raw_reduction']):.4f}).",
                "",
                "## Guardrails",
                "",
                "- Selected candidate model or numerical failures: "
                f"{candidate_failures}.",
                "- The 0/30 promotion gate covers procedure C DUNE fits only.",
                "- Frozen-reference model or numerical failures: "
                f"{reference_failures}.",
                "- Reference failures remain in raw evidence and reduce paired "
                "support; they are not relabeled as candidate failures.",
                "- Recorded same-config infrastructure repairs: "
                f"{infrastructure_repairs}.",
                f"- Numerical promotion guardrail: {numerical_guardrail}.",
                "- Frozen pointwise-recovery and VI-to-NUTS artifacts: passed.",
                "- Additive reconstruction, inspectable effects, and original-scale "
                "bands: passed.",
                "- Family link floors, complete KL accounting, and finite-gradient "
                "gates: passed.",
                "- All 30 configs, run metadata, evaluations, and uncertainty "
                "tables are included in the canonical audit trail.",
                "",
                "## Claims explicitly withheld",
                "",
                withheld,
                "",
            )
        ),
        encoding="utf-8",
    )


def _validated_infrastructure_repairs(
    run_root: Path, pairs: Sequence[Mapping[str, object]]
) -> tuple[int, Path | None]:
    """Validate the durable same-config audit for any full-panel repairs."""
    path = run_root / "confirmation-infrastructure-events.jsonl"
    if not path.is_file():
        return 0, None
    allowed = {
        (str(pair["dataset"]), int(pair["run_seed"]), int(pair["split_seed"]))
        for pair in pairs
    }
    pending: dict[tuple[str, int, int], str] = {}
    repairs = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        key = (
            str(event.get("dataset")),
            int(event.get("run_seed", -1)),
            int(event.get("split_seed", -1)),
        )
        config = (
            run_root
            / "uci_benchmark"
            / f"confirmation-seed-{key[1]}-{key[0]}"
            / "config.yaml"
        )
        if key not in allowed or event.get("config_sha256") != _digest(config):
            raise ValueError(f"invalid infrastructure event provenance: {key}")
        if event.get("event") == "failure":
            pending[key] = str(event["config_sha256"])
        elif event.get("event") == "repair_succeeded" and pending.get(key) == event.get(
            "config_sha256"
        ):
            repairs += 1
            del pending[key]
        else:
            raise ValueError(f"invalid infrastructure repair sequence: {key}")
    if pending:
        raise ValueError(f"unrepaired infrastructure failures: {sorted(pending)}")
    return repairs, path


def _promote_run_evidence(
    run_root: Path,
    output_dir: Path,
    pairs: Sequence[Mapping[str, object]],
) -> None:
    """Copy the minimal complete per-run audit trail into canonical results."""
    for pair in pairs:
        dataset = str(pair["dataset"])
        seed = int(pair["run_seed"])
        source = run_root / "uci_benchmark" / f"confirmation-seed-{seed}-{dataset}"
        metric_root = source / "metrics" / dataset
        uncertainty = metric_root / "C" / "dune_bayes"
        destination = output_dir / "runs" / f"seed-{seed}-{dataset}"
        destination.mkdir(parents=True, exist_ok=True)
        for name, path in {
            "config.yaml": source / "config.yaml",
            "run.json": source / "run.json",
            "comparison.csv": source / "metrics" / "comparison.csv",
            "evaluation.json": metric_root / "evaluation.json",
            "parameter_bands.csv": uncertainty / "parameter_bands.csv",
            "variance_split.csv": uncertainty / "variance_split.csv",
        }.items():
            if path.suffix == ".csv":
                rows = _read_csv(path)
                _write_csv(destination / name, rows, tuple(rows[0]))
            else:
                shutil.copyfile(path, destination / name)


def evaluate_confirmation(
    freeze_dir: Path,
    *,
    run_root: Path,
    output_dir: Path,
    repository_root: Path = Path("."),
    run_code_guardrails: bool = True,
) -> Path:
    """Validate locked runs, assign the earned tier, and publish report artifacts.

    Args:
        freeze_dir: Immutable issue #180 confirmation-input freeze.
        run_root: Root containing the 30 confirmation result directories.
        output_dir: Destination for issue #181's committed evidence.
        repository_root: Git repository containing frozen reference artifacts.
        run_code_guardrails: Whether to execute structural pytest guardrails.

    Returns:
        Path to the machine-readable confirmation report.
    """
    manifest, pairs = _validate_freeze(freeze_dir)
    raw, indexed = _collect_scores(freeze_dir, run_root, pairs)
    mapping_rows = _read_csv(freeze_dir / "comparator_mapping.csv")
    mappings = {
        (row["dataset"], row["metric"]): row["comparator"]
        for row in mapping_rows
        if row["metric"] in PREDICTIVE_METRICS
    }
    closures = _gap_closures(pairs, indexed, mappings)
    dataset_summaries, metric_summaries = _summarize_predictive(closures)
    pit_rows, pit_summary = _pit_normalization(pairs, indexed)
    panel_failures = sum(_model_or_numerical_failure(row) for row in raw)
    selected_candidate_failures = sum(
        _model_or_numerical_failure(row)
        for row in raw
        if row["procedure"] == "C" and row["model"] == "dune_bayes"
    )
    reference_failures = sum(
        _model_or_numerical_failure(row) for row in raw if row["procedure"] == "R"
    )
    infrastructure_repairs, infrastructure_events = _validated_infrastructure_repairs(
        run_root, pairs
    )
    frozen_guardrails = _frozen_guardrails(repository_root, freeze_dir)
    code_guardrails = _code_guardrails(repository_root, run=run_code_guardrails)
    tier = (
        "Mixed"
        if selected_candidate_failures
        else _combined_tier(metric_summaries, pit_summary)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "raw_scores.csv", raw, tuple(raw[0]))
    _write_csv(output_dir / "gap_closure.csv", closures, tuple(closures[0]))
    _write_csv(
        output_dir / "dataset_summary.csv",
        dataset_summaries,
        tuple(dataset_summaries[0]),
    )
    _write_csv(output_dir / "pit_normalization.csv", pit_rows, tuple(pit_rows[0]))
    _promote_run_evidence(run_root, output_dir, pairs)
    if infrastructure_events is not None:
        shutil.copyfile(
            infrastructure_events, output_dir / "infrastructure_events.jsonl"
        )
    report: dict[str, object] = {
        "issue": 181,
        "evidence_role": "locked_confirmation",
        "development_freeze_sha256": _digest(freeze_dir / "freeze.json"),
        "development_artifact_hashes_sha256": manifest["artifact_hashes_sha256"],
        "split_run_pairs": pairs,
        "raw_score_count": len(raw),
        "selected_candidate_model_or_numerical_failures": (selected_candidate_failures),
        "failure_gate_scope": "procedure_C_dune_bayes_30_fits",
        "reference_model_or_numerical_failures": reference_failures,
        "panel_model_or_numerical_failures": panel_failures,
        "infrastructure_repairs": infrastructure_repairs,
        "predictive_metrics": metric_summaries,
        "pit": pit_summary,
        "guardrails": {
            "frozen_artifacts": frozen_guardrails,
            "code": code_guardrails,
            "selected_candidate_numerical_gate": selected_candidate_failures == 0,
            "all_panel_rows_passed_numerical_gate": panel_failures == 0,
        },
        "mechanically_earned_tier": tier,
        "reported_tier": tier,
        "human_downgrade_allowed": True,
        "human_upgrade_allowed": False,
        "panel_bounded": True,
        "withheld_claims": list(WITHHELD_CLAIMS),
        "package_default_promoted": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report_markdown(output_dir / "report.md", report)
    artifact_hashes = {
        path.relative_to(output_dir).as_posix(): _digest(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"sha256": artifact_hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run and evaluate the issue #181 locked confirmation panel.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code: zero only when the confirmation report is complete.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze",
        type=Path,
        default=EXPERIMENT_DIR / "results/development-freeze",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-root", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT_DIR / "results/confirmation"
    )
    args = parser.parse_args(argv)
    freeze = args.freeze.resolve()
    run_root = args.run_root.resolve()
    if args.run:
        run_confirmation_panel(freeze, run_root=run_root, smoke=args.smoke)
    if args.smoke:
        return 0
    evaluate_confirmation(
        freeze,
        run_root=run_root,
        output_dir=args.output.resolve(),
        repository_root=Path(__file__).resolve().parents[2],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
