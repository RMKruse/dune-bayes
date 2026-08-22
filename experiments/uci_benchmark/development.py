"""Development-panel execution and confirmation freeze (ADR-0008, issue #180)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_SEEDS = (102, 112, 122)
CONFIRMATION_RUN_SEEDS = (202, 212, 222)
PROCEDURES = ("R", "C")
PRIMARY_MODELS = (
    "dune_bayes",
    "deterministic_namlss",
    "plain_mlp",
    "deep_ensemble",
)
METRICS = {
    "nll": "mean_nll",
    "crps": "mean_crps",
    "pit_error": "calibration_error",
}
SELECTED_CANDIDATE_FAILURE_LIMIT = 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one benchmark table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: tuple[str, ...]
) -> None:
    """Write a deterministic table, including its header when empty."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _score_index(
    rows: list[dict[str, str]], errors: list[str]
) -> dict[tuple[str, int, int, str, str], dict[str, str]]:
    """Index raw selected-fit rows and report duplicate keys."""
    indexed: dict[tuple[str, int, int, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["dataset"],
            int(row["seed"]),
            int(row["split_seed"]),
            row["procedure"],
            row["model"],
        )
        if key in indexed:
            errors.append(f"duplicate raw score key {key}")
        indexed[key] = row
    return indexed


def _selected_configurations(
    selection: Mapping[str, object],
    *,
    dataset: str,
    expected_slots: int,
    errors: list[str],
) -> dict[tuple[str, str], object]:
    """Validate one complete tuning trace and return its frozen winners."""
    selected: dict[tuple[str, str], object] = {}
    if selection.get("dataset") != dataset or selection.get("tuning_seed") != 102:
        errors.append(f"selection provenance drifted for {dataset}")
    if selection.get("executed_candidate_count") != expected_slots:
        errors.append(f"incomplete selection trace for {dataset}")
    procedures = selection.get("procedures")
    if not isinstance(procedures, Mapping) or set(procedures) != set(PROCEDURES):
        errors.append(f"incomplete selection procedures for {dataset}")
        return selected
    for procedure in PROCEDURES:
        procedure_evidence = procedures[procedure]
        models = (
            procedure_evidence.get("models")
            if isinstance(procedure_evidence, Mapping)
            else None
        )
        if not isinstance(models, Mapping) or set(models) != set(PRIMARY_MODELS):
            errors.append(f"incomplete selection models for {dataset}, {procedure}")
            continue
        for model in PRIMARY_MODELS:
            evidence = models[model]
            if not isinstance(evidence, Mapping):
                errors.append(
                    f"invalid selection evidence for {dataset}, {procedure}, {model}"
                )
                continue
            trials = evidence.get("trials")
            configuration = evidence.get("selected_configuration")
            candidate_id = evidence.get("selected_candidate_id")
            complete = (
                evidence.get("candidate_slots") == expected_slots
                and isinstance(trials, list)
                and len(trials) == expected_slots
                and configuration is not None
                and candidate_id is not None
                and any(
                    isinstance(trial, Mapping)
                    and trial.get("candidate_id") == candidate_id
                    and trial.get("configuration") == configuration
                    for trial in trials or []
                )
            )
            if not complete:
                errors.append(
                    f"incomplete selection trace for {dataset}, {procedure}, {model}"
                )
                continue
            selected[(procedure, model)] = configuration
    return selected


def _validate_evaluation(
    evaluation: Mapping[str, object],
    *,
    dataset: str,
    seed: int,
    selection_sha256: str,
    selected: Mapping[tuple[str, str], object],
    errors: list[str],
) -> None:
    """Require one exact frozen-winner refit trace per procedure and model."""
    if (
        evaluation.get("dataset") != dataset
        or evaluation.get("seed") != seed
        or evaluation.get("selection_sha256") != selection_sha256
    ):
        errors.append(f"evaluation provenance drifted for {dataset}, seed {seed}")
    procedures = evaluation.get("procedures")
    if not isinstance(procedures, Mapping) or set(procedures) != set(PROCEDURES):
        errors.append(f"incomplete evaluation trace for {dataset}, seed {seed}")
        return
    for procedure in PROCEDURES:
        procedure_evidence = procedures[procedure]
        models = (
            procedure_evidence.get("models")
            if isinstance(procedure_evidence, Mapping)
            else None
        )
        if not isinstance(models, Mapping) or set(models) != set(PRIMARY_MODELS):
            errors.append(
                f"incomplete evaluation trace for {dataset}, seed {seed}, {procedure}"
            )
            continue
        for model in PRIMARY_MODELS:
            evidence = models[model]
            trials = evidence.get("trials") if isinstance(evidence, Mapping) else None
            expected = selected.get((procedure, model))
            complete = (
                isinstance(evidence, Mapping)
                and evidence.get("candidate_slots") == 1
                and isinstance(trials, list)
                and len(trials) == 1
                and isinstance(trials[0], Mapping)
                and trials[0].get("configuration") == expected
                and expected is not None
            )
            if not complete:
                errors.append(
                    f"incomplete evaluation trace for {dataset}, seed {seed}, "
                    f"{procedure}, {model}"
                )


def _value(row: dict[str, str] | None, metric: str) -> float | None:
    """Return one finite reported score, including model-failure diagnostics."""
    if row is None or row.get("status") == "infrastructure_failure":
        return None
    try:
        value = float(row[METRICS[metric]])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _gap(
    left: dict[str, str] | None,
    right: dict[str, str] | None,
    metric: str,
) -> float | None:
    """Return a lower-is-better paired gap when both scores are usable."""
    left_value = _value(left, metric)
    right_value = _value(right, metric)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _summarize_gaps(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize paired gaps within each dataset without pooling raw scores."""
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["gap"] != "":
            grouped[
                (
                    str(row["dataset"]),
                    str(row["metric"]),
                    str(row["comparison"]),
                    str(row["procedure"]),
                    str(row["comparator"]),
                )
            ].append(float(str(row["gap"])))

    summaries: list[dict[str, object]] = []
    for key, gaps in sorted(grouped.items()):
        dataset, metric, comparison, procedure, comparator = key
        summaries.append(
            {
                "dataset": dataset,
                "metric": metric,
                "comparison": comparison,
                "procedure": procedure,
                "comparator": comparator,
                "n_pairs": len(gaps),
                "wins": sum(gap < 0.0 for gap in gaps),
                "losses": sum(gap > 0.0 for gap in gaps),
                "ties": sum(gap == 0.0 for gap in gaps),
                "median_gap": statistics.median(gaps),
                "min_gap": min(gaps),
                "max_gap": max(gaps),
            }
        )
    return summaries


def freeze_development_panel(
    config_path: Path,
    *,
    run_root: Path,
    output_dir: Path,
    smoke: bool = False,
) -> Path:
    """Freeze development scores, selections, comparator identities, and hashes.

    Args:
        config_path: Locked benchmark configuration declaring datasets and splits.
        run_root: Root containing one dataset-level run for every development seed.
        output_dir: Destination for the immutable confirmation inputs.
        smoke: Whether to accept only the first dataset as an orchestration tracer.

    Returns:
        The written freeze manifest path.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = config["datasets"][:1] if smoke else config["datasets"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    (output_dir / "evaluations").mkdir(exist_ok=True)
    (output_dir / "run_metadata").mkdir(exist_ok=True)
    (output_dir / "selections").mkdir(exist_ok=True)
    shutil.copyfile(config_path, output_dir / "development_config.yaml")

    errors: list[str] = []
    development = config.get("development", {})
    if tuple(development.get("run_seeds", ())) != RUN_SEEDS:
        errors.append("development run seeds drifted from 102, 112, 122")
    if tuple(development.get("confirmation_run_seeds", ())) != CONFIRMATION_RUN_SEEDS:
        errors.append("confirmation run seeds drifted from 202, 212, 222")
    if development.get("comparator_candidates") != [
        "plain_mlp",
        "deep_ensemble",
    ]:
        errors.append("comparator candidates drifted from the frozen pair")
    if development.get("comparator_tie_winner") != "deep_ensemble":
        errors.append("comparator exact-tie winner drifted from deep_ensemble")
    if (
        int(development.get("selected_candidate_failure_limit", -1))
        != SELECTED_CANDIDATE_FAILURE_LIMIT
    ):
        errors.append("selected candidate failure limit drifted from 1/30")
    split_dir = Path(str(config["data"]["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = config_path.parent / split_dir
    split_rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(datasets, start=1):
        name = str(dataset["name"])
        if int(dataset["split_seed"]) != 10200 + dataset_index:
            errors.append(f"development split seed drifted for {name}")
        split_path = split_dir / f"{name}.npz"
        if not split_path.is_file():
            errors.append(f"missing persisted split for {name}")
            continue
        split_rows.append(
            {
                "dataset": name,
                "split_seed": int(dataset["split_seed"]),
                "file": split_path.name,
                "sha256": sha256(split_path.read_bytes()).hexdigest(),
            }
        )
    _write_csv(
        output_dir / "split_hashes.csv",
        split_rows,
        ("dataset", "split_seed", "file", "sha256"),
    )
    infrastructure_events: list[dict[str, object]] = []
    raw_rows: list[dict[str, str]] = []
    selected_rows: list[dict[str, object]] = []
    expected_slots = (
        int(config["tuning"]["smoke_candidate_count"])
        if smoke
        else len(config["tuning"]["hidden_dims"])
        * len(config["tuning"]["learning_rates"])
    )
    for dataset in datasets:
        name = str(dataset["name"])
        expected_split_seed = int(dataset["split_seed"])
        selection_path = (
            run_root
            / "uci_benchmark"
            / f"development-seed-102-{name}"
            / "metrics"
            / name
            / "selection.json"
        )
        selected: dict[tuple[str, str], object] = {}
        selection_digest = ""
        if not selection_path.is_file():
            errors.append(f"missing complete selection trace for {name}")
        else:
            selection_bytes = selection_path.read_bytes()
            selection_digest = sha256(selection_bytes).hexdigest()
            selection = json.loads(selection_bytes)
            selected = _selected_configurations(
                selection,
                dataset=name,
                expected_slots=expected_slots,
                errors=errors,
            )
            shutil.copyfile(selection_path, output_dir / "selections" / f"{name}.json")
            for procedure in PROCEDURES:
                for model in PRIMARY_MODELS:
                    evidence = selection["procedures"][procedure]["models"][model]
                    selected_rows.append(
                        {
                            "dataset": name,
                            "procedure": procedure,
                            "model": model,
                            "candidate_id": evidence.get("selected_candidate_id"),
                            "configuration": json.dumps(
                                evidence.get("selected_configuration"), sort_keys=True
                            ),
                        }
                    )
        for seed in RUN_SEEDS:
            run = run_root / "uci_benchmark" / f"development-seed-{seed}-{name}"
            comparison_path = run / "metrics" / "comparison.csv"
            run_config = run / "config.yaml"
            run_metadata = run / "run.json"
            if (
                not comparison_path.is_file()
                or not run_config.is_file()
                or not run_metadata.is_file()
            ):
                failure_path = run / "infrastructure_failure.json"
                event = (
                    json.loads(failure_path.read_text(encoding="utf-8"))
                    if failure_path.is_file()
                    else {"dataset": name, "seed": seed, "failure": "missing_artifact"}
                )
                infrastructure_events.append(event)
                errors.append(
                    f"missing completed development run for {name}, seed {seed}"
                )
                continue
            frozen_config = yaml.safe_load(run_config.read_text(encoding="utf-8"))
            metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
            configured_dataset = next(
                (
                    item
                    for item in frozen_config.get("datasets", [])
                    if item.get("name") == name
                ),
                None,
            )
            if (
                frozen_config.get("seed") != seed
                or frozen_config.get("development", {}).get("dataset") != name
                or frozen_config.get("development", {}).get("selection_seed") != 102
                or configured_dataset is None
                or int(configured_dataset.get("split_seed", -1)) != expected_split_seed
            ):
                errors.append(f"run config provenance drifted for {name}, seed {seed}")
            if (
                metadata.get("experiment") != "uci_benchmark"
                or metadata.get("seed") != seed
            ):
                errors.append(
                    f"run metadata provenance drifted for {name}, seed {seed}"
                )
            shutil.copyfile(
                run_config, output_dir / "configs" / f"seed-{seed}-{name}.yaml"
            )
            shutil.copyfile(
                run_metadata,
                output_dir / "run_metadata" / f"seed-{seed}-{name}.json",
            )
            if seed != RUN_SEEDS[0]:
                evaluation_path = run / "metrics" / name / "evaluation.json"
                if evaluation_path.is_file():
                    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                    _validate_evaluation(
                        evaluation,
                        dataset=name,
                        seed=seed,
                        selection_sha256=selection_digest,
                        selected=selected,
                        errors=errors,
                    )
                    shutil.copyfile(
                        evaluation_path,
                        output_dir / "evaluations" / f"seed-{seed}-{name}.json",
                    )
                else:
                    errors.append(f"missing evaluation trace for {name}, seed {seed}")
            for row in _read_csv(comparison_path):
                try:
                    row_seed = int(row["seed"])
                    row_split_seed = int(row["split_seed"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"malformed raw score for {name}, seed {seed}")
                    continue
                if row.get("dataset") != name or row_seed != seed:
                    errors.append(
                        f"raw score provenance drifted for {name}, seed {seed}"
                    )
                if row_split_seed != expected_split_seed:
                    errors.append(f"wrong split seed for {name}, seed {seed}")
                if (
                    row.get("procedure") not in PROCEDURES
                    or row.get("model") not in PRIMARY_MODELS
                    or row.get("comparison_role") != "primary"
                    or row.get("status")
                    not in {"completed", "model_failure", "infrastructure_failure"}
                ):
                    errors.append(
                        f"invalid raw score classification for {name}, seed {seed}"
                    )
                if row.get("status") == "completed" and any(
                    _value(row, metric) is None for metric in METRICS
                ):
                    errors.append(f"invalid completed metrics for {name}, seed {seed}")
                if (
                    row.get("status") != "completed"
                    and not row.get("failure", "").strip()
                ):
                    errors.append(
                        f"missing failure classification for {name}, seed {seed}"
                    )
                raw_rows.append(row)

    raw_rows.sort(
        key=lambda row: (
            row["dataset"],
            int(row["seed"]),
            row["procedure"],
            row["model"],
        )
    )
    raw_fields = (
        "comparison_role",
        "dataset",
        "family",
        "model",
        "procedure",
        "seed",
        "split_seed",
        "uncertainty_scope",
        "n_test",
        "mean_nll",
        "mean_crps",
        "calibration_error",
        "status",
        "failure",
    )
    _write_csv(output_dir / "raw_scores.csv", raw_rows, raw_fields)
    _write_csv(
        output_dir / "selected_configurations.csv",
        selected_rows,
        ("dataset", "procedure", "model", "candidate_id", "configuration"),
    )

    indexed = _score_index(raw_rows, errors)
    expected_keys = {
        (
            str(dataset["name"]),
            seed,
            int(dataset["split_seed"]),
            procedure,
            model,
        )
        for dataset in datasets
        for seed in RUN_SEEDS
        for procedure in PROCEDURES
        for model in PRIMARY_MODELS
    }
    missing_keys = expected_keys - set(indexed)
    if missing_keys:
        errors.append(f"missing {len(missing_keys)} selected-fit score rows")

    mappings: list[dict[str, object]] = []
    comparator_by_metric: dict[tuple[str, str], str] = {}
    for dataset in datasets:
        name = str(dataset["name"])
        split_seed = int(dataset["split_seed"])
        for metric in METRICS:
            medians: dict[str, float] = {}
            for model in ("plain_mlp", "deep_ensemble"):
                scores = [
                    _value(indexed.get((name, seed, split_seed, "C", model)), metric)
                    for seed in RUN_SEEDS
                ]
                finite = [score for score in scores if score is not None]
                if len(finite) != len(RUN_SEEDS):
                    errors.append(
                        f"incomplete comparator scores for {name}, {metric}, {model}"
                    )
                if finite:
                    medians[model] = statistics.median(finite)
            if len(medians) != 2:
                continue
            comparator = (
                "deep_ensemble"
                if medians["deep_ensemble"] <= medians["plain_mlp"]
                else "plain_mlp"
            )
            comparator_by_metric[(name, metric)] = comparator
            mappings.append(
                {
                    "dataset": name,
                    "metric": metric,
                    "comparator": comparator,
                    "plain_mlp_development_median": medians["plain_mlp"],
                    "deep_ensemble_development_median": medians["deep_ensemble"],
                    "tie_rule": "deep_ensemble",
                    "procedure": "C",
                }
            )
    _write_csv(
        output_dir / "comparator_mapping.csv",
        mappings,
        (
            "dataset",
            "metric",
            "comparator",
            "plain_mlp_development_median",
            "deep_ensemble_development_median",
            "tie_rule",
            "procedure",
        ),
    )

    paired: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    for dataset in datasets:
        name = str(dataset["name"])
        split_seed = int(dataset["split_seed"])
        for seed in RUN_SEEDS:
            reference = indexed.get((name, seed, split_seed, "R", "dune_bayes"))
            candidate = indexed.get((name, seed, split_seed, "C", "dune_bayes"))
            for metric in METRICS:
                comparator = comparator_by_metric.get((name, metric), "")
                baseline = indexed.get((name, seed, split_seed, "C", comparator))
                for pair_name, left, right in (
                    ("candidate_minus_reference", candidate, reference),
                    ("candidate_minus_comparator_b", candidate, baseline),
                    ("reference_minus_comparator_b", reference, baseline),
                ):
                    gap = _gap(left, right, metric)
                    paired.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "metric": metric,
                            "comparison": pair_name,
                            "procedure": "",
                            "comparator": comparator
                            if "comparator" in pair_name
                            else "",
                            "gap": "" if gap is None else gap,
                        }
                    )
                for procedure in PROCEDURES:
                    dune = indexed.get(
                        (name, seed, split_seed, procedure, "dune_bayes")
                    )
                    for model in PRIMARY_MODELS[1:]:
                        gap = _gap(
                            dune,
                            indexed.get((name, seed, split_seed, procedure, model)),
                            metric,
                        )
                        contrasts.append(
                            {
                                "dataset": name,
                                "seed": seed,
                                "metric": metric,
                                "comparison": "dune_minus_model",
                                "procedure": procedure,
                                "comparator": model,
                                "gap": "" if gap is None else gap,
                            }
                        )
    gap_fields = (
        "dataset",
        "seed",
        "metric",
        "comparison",
        "procedure",
        "comparator",
        "gap",
    )
    _write_csv(output_dir / "paired_gaps.csv", paired, gap_fields)
    _write_csv(output_dir / "model_contrasts.csv", contrasts, gap_fields)
    _write_csv(
        output_dir / "win_loss_summary.csv",
        _summarize_gaps(paired + contrasts),
        (
            "dataset",
            "metric",
            "comparison",
            "procedure",
            "comparator",
            "n_pairs",
            "wins",
            "losses",
            "ties",
            "median_gap",
            "min_gap",
            "max_gap",
        ),
    )

    failure_rows: list[dict[str, object]] = []
    for procedure in PROCEDURES:
        for model in PRIMARY_MODELS:
            group = [
                row
                for row in raw_rows
                if row["procedure"] == procedure and row["model"] == model
            ]
            failure_rows.append(
                {
                    "procedure": procedure,
                    "model": model,
                    "selected_fits": len(group),
                    "completed": sum(row["status"] == "completed" for row in group),
                    "model_failures": sum(
                        row["status"] == "model_failure" for row in group
                    ),
                    "infrastructure_failures": sum(
                        row["status"] == "infrastructure_failure" for row in group
                    ),
                }
            )
    failure_rows.append(
        {
            "procedure": "",
            "model": "artifact_production",
            "selected_fits": 0,
            "completed": 0,
            "model_failures": 0,
            "infrastructure_failures": len(infrastructure_events),
        }
    )
    _write_csv(
        output_dir / "failure_incidence.csv",
        failure_rows,
        (
            "procedure",
            "model",
            "selected_fits",
            "completed",
            "model_failures",
            "infrastructure_failures",
        ),
    )
    candidate_failures = sum(
        row["status"] == "model_failure"
        for row in raw_rows
        if row["procedure"] == "C" and row["model"] == "dune_bayes"
    )
    infrastructure_failures = len(infrastructure_events) + sum(
        row["status"] == "infrastructure_failure" for row in raw_rows
    )
    if candidate_failures > SELECTED_CANDIDATE_FAILURE_LIMIT:
        errors.append(
            f"candidate has {candidate_failures} model failures; limit is "
            f"{SELECTED_CANDIDATE_FAILURE_LIMIT}"
        )
    if infrastructure_failures:
        errors.append(
            f"development panel has {infrastructure_failures} infrastructure failures"
        )

    split_run_pairs = [
        {
            "dataset": str(dataset["name"]),
            "run_seed": run_seed,
            "split_seed": run_seed * 100 + index,
        }
        for run_seed in CONFIRMATION_RUN_SEEDS
        for index, dataset in enumerate(config["datasets"], start=1)
    ]
    hashed_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"artifact_hashes.json", "freeze.json"}
    )
    hashes = {
        "algorithm": "sha256",
        "sha256": {
            path.relative_to(output_dir).as_posix(): sha256(
                path.read_bytes()
            ).hexdigest()
            for path in hashed_files
        },
    }
    hash_path = output_dir / "artifact_hashes.json"
    hash_path.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "issue": 180,
        "evidence_role": "bounded_smoke_only" if smoke else "development_freeze",
        "development_run_seeds": list(RUN_SEEDS),
        "development_split_seeds": {
            str(dataset["name"]): int(dataset["split_seed"]) for dataset in datasets
        },
        "selected_candidate_failure_limit": SELECTED_CANDIDATE_FAILURE_LIMIT,
        "selected_candidate_model_failures": candidate_failures,
        "confirmation_inputs_complete": not errors and not smoke,
        "completeness_errors": errors,
        "artifact_hashes_sha256": sha256(hash_path.read_bytes()).hexdigest(),
        "split_hashes_sha256": sha256(
            (output_dir / "split_hashes.csv").read_bytes()
        ).hexdigest(),
        "confirmation": {
            "run_seeds": list(CONFIRMATION_RUN_SEEDS),
            "split_run_pairs": split_run_pairs,
            "scores_generated": False,
        },
        "package_default_promoted": False,
        "paper_claim_promoted": False,
    }
    freeze_path = output_dir / "freeze.json"
    freeze_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return freeze_path


def _run_panel(config_path: Path, *, run_root: Path, smoke: bool) -> None:
    """Run each dataset separately so a long panel can resume safely."""
    run_root = run_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = config["datasets"][:1] if smoke else config["datasets"]
    generated = run_root / "development-configs"
    generated.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        name = str(dataset["name"])
        selection_root: Path | None = None
        for seed in RUN_SEEDS:
            run_name = f"development-seed-{seed}-{name}"
            result = run_root / "uci_benchmark" / run_name
            if (result / "run.json").is_file() and (
                result / "metrics" / "comparison.csv"
            ).is_file():
                if seed == 102:
                    selection_root = result / "metrics"
                continue
            run_config = dict(config)
            run_config["seed"] = seed
            run_config["development"] = {
                **config.get("development", {}),
                "enabled": True,
                "dataset": name,
                "selection_seed": 102,
            }
            run_config["artifacts"] = {"root": str(run_root), "run_name": run_name}
            target = generated / f"seed-{seed}-{name}.yaml"
            target.write_text(
                yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8"
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("run.py")),
                str(target),
            ]
            if smoke:
                command.append("--smoke")
            command.extend(("--dataset", name))
            if selection_root is not None:
                command.extend(("--selection-root", str(selection_root)))
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                result.mkdir(parents=True, exist_ok=True)
                (result / "infrastructure_failure.json").write_text(
                    json.dumps(
                        {
                            "classification": "infrastructure_failure",
                            "dataset": name,
                            "seed": seed,
                            "returncode": completed.returncode,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if seed == 102:
                    break
                continue
            if seed == 102:
                selection_root = result / "metrics"


def main(argv: list[str] | None = None) -> int:
    """Run and freeze the issue #180 development panel.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code: zero only when the requested freeze is complete.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-root", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT_DIR / "results/development-freeze"
    )
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if args.run:
        _run_panel(args.config, run_root=run_root, smoke=args.smoke)
    freeze_path = freeze_development_panel(
        args.config,
        run_root=run_root,
        output_dir=output,
        smoke=args.smoke,
    )
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    return 0 if manifest["confirmation_inputs_complete"] or args.smoke else 1


if __name__ == "__main__":
    raise SystemExit(main())
