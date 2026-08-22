"""Locked confirmation-panel boundaries (GitHub #181)."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from scipy import stats

from experiments.uci_benchmark.confirmation import (
    assess_frozen_guardrail_thresholds,
    evaluate_confirmation,
    run_confirmation_panel,
)

FREEZE = Path("experiments/uci_benchmark/results/development-freeze")
MODELS = ("dune_bayes", "deterministic_namlss", "plain_mlp", "deep_ensemble")
PROCEDURES = ("R", "C")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one synthetic public metric table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_confirmation_runs(root: Path) -> None:
    """Materialize strong evidence for every frozen dataset/split/run pair."""
    freeze = json.loads((FREEZE / "freeze.json").read_text(encoding="utf-8"))
    mappings = list(csv.DictReader((FREEZE / "comparator_mapping.csv").open()))
    comparators = {
        (row["dataset"], row["metric"]): row["comparator"] for row in mappings
    }
    base = yaml.safe_load(
        (FREEZE / "development_config.yaml").read_text(encoding="utf-8")
    )
    dataset_by_name = {dataset["name"]: dataset for dataset in base["datasets"]}
    for pair in freeze["confirmation"]["split_run_pairs"]:
        dataset = pair["dataset"]
        seed = pair["run_seed"]
        split_seed = pair["split_seed"]
        run = root / "uci_benchmark" / f"confirmation-seed-{seed}-{dataset}"
        metric_dir = run / "metrics" / dataset
        selection = json.loads(
            (FREEZE / "selections" / f"{dataset}.json").read_text(encoding="utf-8")
        )
        rows: list[dict[str, object]] = []
        for procedure in PROCEDURES:
            for model in MODELS:
                is_dune = model == "dune_bayes"
                score = 3.0 if is_dune and procedure == "R" else 1.5 if is_dune else 1.0
                rows.append(
                    {
                        "comparison_role": "primary",
                        "dataset": dataset,
                        "family": dataset_by_name[dataset]["family"],
                        "model": model,
                        "procedure": procedure,
                        "seed": seed,
                        "split_seed": split_seed,
                        "uncertainty_scope": "distributional_parameter_bands",
                        "n_test": 50 + split_seed % 10,
                        "mean_nll": score,
                        "mean_crps": score,
                        "calibration_error": 0.05 if procedure == "R" else 0.025,
                        "status": "completed",
                        "failure": "",
                    }
                )
        # Keep the frozen comparator identity while making both eligible choices tie.
        for metric in ("nll", "crps"):
            comparator = comparators[(dataset, metric)]
            assert comparator in {"plain_mlp", "deep_ensemble"}
        _write_csv(run / "metrics" / "comparison.csv", rows)
        _write_csv(
            metric_dir / "C" / "dune_bayes" / "parameter_bands.csv",
            [
                {
                    "dataset": dataset,
                    "model": "dune_bayes",
                    "observation": 0,
                    "parameter": "loc",
                    "q05": 0.5,
                    "q50": 1.0,
                    "q95": 1.5,
                }
            ],
        )
        _write_csv(
            metric_dir / "C" / "dune_bayes" / "variance_split.csv",
            [
                {
                    "dataset": dataset,
                    "model": "dune_bayes",
                    "observation": 0,
                    "aleatoric": 1.0,
                    "epistemic": 0.5,
                    "total": 1.5,
                }
            ],
        )
        run.mkdir(parents=True, exist_ok=True)
        (run / "run.json").write_text(
            json.dumps({"experiment": "uci_benchmark", "seed": seed, "smoke": False}),
            encoding="utf-8",
        )
        config = json.loads(json.dumps(base))
        configured_dataset = dict(dataset_by_name[dataset])
        configured_dataset["split_seed"] = split_seed
        config["seed"] = seed
        config["datasets"] = [configured_dataset]
        config["data"]["split_dir"] = f"data/confirmation-splits/seed-{seed}"
        config["development"] = {
            **config.get("development", {}),
            "enabled": True,
            "dataset": dataset,
            "selection_seed": 102,
        }
        config["artifacts"] = {
            "root": str(root),
            "run_name": f"confirmation-seed-{seed}-{dataset}",
        }
        (run / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        evaluation = {
            "dataset": dataset,
            "seed": seed,
            "selection_sha256": sha256(
                (FREEZE / "selections" / f"{dataset}.json").read_bytes()
            ).hexdigest(),
            "procedures": {
                procedure: {
                    "models": {
                        model: {
                            "candidate_slots": 1,
                            "failures": 0,
                            "selected_configuration": selection["procedures"][
                                procedure
                            ]["models"][model]["selected_configuration"],
                            "trials": [
                                {
                                    "status": "completed",
                                    "configuration": selection["procedures"][procedure][
                                        "models"
                                    ][model]["selected_configuration"],
                                }
                            ],
                        }
                        for model in MODELS
                    }
                }
                for procedure in PROCEDURES
            },
        }
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "evaluation.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )


def test_confirmation_assigns_the_mechanically_earned_panel_claim(
    tmp_path: Path,
) -> None:
    """Locked strong evidence produces a reviewable Strong panel-bounded claim."""
    run_root = tmp_path / "runs"
    _write_confirmation_runs(run_root)
    output = tmp_path / "confirmation"

    report_path = evaluate_confirmation(
        FREEZE,
        run_root=run_root,
        output_dir=output,
        repository_root=Path("."),
        run_code_guardrails=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw = list(csv.DictReader((output / "raw_scores.csv").open()))
    closures = list(csv.DictReader((output / "gap_closure.csv").open()))
    pit = list(csv.DictReader((output / "pit_normalization.csv").open()))
    assert len(raw) == 10 * 3 * 2 * 4
    assert len(closures) == 10 * 3 * 2
    assert len(pit) == 10 * 3
    assert {float(row["capped_gap_closure"]) for row in closures} == {0.75}
    assert all(float(row["sampling_noise_floor"]) > 0.0 for row in pit)
    n_test = int(pit[0]["n_test"])
    expected_floor = sum(
        abs(k / n_test - 0.1) * stats.binom.pmf(k, n_test, 0.1)
        for k in range(n_test + 1)
    )
    assert float(pit[0]["sampling_noise_floor"]) == pytest.approx(
        # Closed form and scipy's float64 sum agree to double-precision rounding.
        expected_floor,
        abs=1e-15,
    )
    assert report["mechanically_earned_tier"] == "Strong"
    assert report["human_downgrade_allowed"] is True
    assert report["human_upgrade_allowed"] is False
    assert report["selected_candidate_model_or_numerical_failures"] == 0
    assert report["failure_gate_scope"] == "procedure_C_dune_bayes_30_fits"
    assert report["reference_model_or_numerical_failures"] == 0
    assert report["infrastructure_repairs"] == 0
    assert report["withheld_claims"] == [
        "universal predictive dominance",
        "general calibration",
        "VI-to-NUTS equivalence",
        "package-default promotion",
    ]
    assert report["guardrails"]["frozen_artifacts"]["passed"] is True
    assert (output / "report.md").is_file()
    promoted_runs = sorted((output / "runs").iterdir())
    assert len(promoted_runs) == 30
    assert all(
        {
            "config.yaml",
            "run.json",
            "comparison.csv",
            "evaluation.json",
            "parameter_bands.csv",
            "variance_split.csv",
        }
        <= {path.name for path in run.iterdir()}
        for run in promoted_runs
    )
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "runs/seed-202-autompg/config.yaml" in hashes["sha256"]


def test_confirmation_rejects_a_non_predeclared_split(tmp_path: Path) -> None:
    """A changed split cannot be laundered into confirmatory evidence."""
    run_root = tmp_path / "runs"
    _write_confirmation_runs(run_root)
    config = next(run_root.glob("uci_benchmark/confirmation-*/config.yaml"))
    drifted = yaml.safe_load(config.read_text(encoding="utf-8"))
    drifted["datasets"][0]["split_seed"] += 1
    config.write_text(yaml.safe_dump(drifted), encoding="utf-8")

    with pytest.raises(ValueError, match="split seed"):
        evaluate_confirmation(
            FREEZE,
            run_root=run_root,
            output_dir=tmp_path / "confirmation",
            repository_root=Path("."),
            run_code_guardrails=False,
        )


def test_confirmation_rejects_raw_score_provenance_drift(tmp_path: Path) -> None:
    """A score row cannot claim a different dataset or run seed."""
    run_root = tmp_path / "runs"
    _write_confirmation_runs(run_root)
    comparison = next(
        run_root.glob("uci_benchmark/confirmation-*/metrics/comparison.csv")
    )
    rows = list(csv.DictReader(comparison.open()))
    rows[0]["seed"] = "999"
    _write_csv(comparison, rows)

    with pytest.raises(ValueError, match="score provenance"):
        evaluate_confirmation(
            FREEZE,
            run_root=run_root,
            output_dir=tmp_path / "confirmation",
            repository_root=Path("."),
            run_code_guardrails=False,
        )


def test_confirmation_failure_forces_mixed_without_dropping_raw_scores(
    tmp_path: Path,
) -> None:
    """A saturated selected fit is retained and blocks claim-tier promotion."""
    run_root = tmp_path / "runs"
    _write_confirmation_runs(run_root)
    comparison = (
        run_root
        / "uci_benchmark"
        / "confirmation-seed-202-power"
        / "metrics"
        / "comparison.csv"
    )
    rows = list(csv.DictReader(comparison.open()))
    failed = next(
        row for row in rows if row["procedure"] == "C" and row["model"] == "dune_bayes"
    )
    failed["calibration_error"] = 0.18
    failed["status"] = "model_failure"
    failed["failure"] = "saturated_pit"
    _write_csv(comparison, rows)

    report_path = evaluate_confirmation(
        FREEZE,
        run_root=run_root,
        output_dir=tmp_path / "confirmation",
        repository_root=Path("."),
        run_code_guardrails=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw = list(csv.DictReader((report_path.parent / "raw_scores.csv").open()))
    assert report["mechanically_earned_tier"] == "Mixed"
    assert report["selected_candidate_model_or_numerical_failures"] == 1
    assert len(raw) == 240
    retained = next(
        row
        for row in raw
        if row["dataset"] == "power"
        and row["seed"] == "202"
        and row["procedure"] == "C"
        and row["model"] == "dune_bayes"
    )
    assert retained["failure"] == "saturated_pit"


def test_confirmation_reference_failure_reduces_breadth_without_blocking(
    tmp_path: Path,
) -> None:
    """A frozen-reference failure remains visible but is not a candidate failure."""
    run_root = tmp_path / "runs"
    _write_confirmation_runs(run_root)
    comparison = (
        run_root
        / "uci_benchmark"
        / "confirmation-seed-202-power"
        / "metrics"
        / "comparison.csv"
    )
    rows = list(csv.DictReader(comparison.open()))
    for row in rows:
        if row["procedure"] == "R":
            row["calibration_error"] = 0.18
            row["status"] = "model_failure"
            row["failure"] = "saturated_pit"
    _write_csv(comparison, rows)

    report_path = evaluate_confirmation(
        FREEZE,
        run_root=run_root,
        output_dir=tmp_path / "confirmation",
        repository_root=Path("."),
        run_code_guardrails=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mechanically_earned_tier"] == "Strong"
    assert report["selected_candidate_model_or_numerical_failures"] == 0
    assert report["reference_model_or_numerical_failures"] == 4
    assert report["panel_model_or_numerical_failures"] == 4
    assert report["pit"]["reference_saturated_runs"] == 1
    assert report["pit"]["candidate_saturated_runs"] == 0


def test_confirmation_publishes_validated_infrastructure_repairs(
    tmp_path: Path,
) -> None:
    """A full-panel repair audit is validated and copied into canonical evidence."""
    run_root = tmp_path / "runs"
    _write_confirmation_runs(run_root)
    config = run_root / "uci_benchmark/confirmation-seed-202-autompg/config.yaml"
    record = {
        "dataset": "autompg",
        "run_seed": 202,
        "split_seed": 20201,
        "config_sha256": sha256(config.read_bytes()).hexdigest(),
        "returncode": 1,
    }
    events = run_root / "confirmation-infrastructure-events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(
                {**record, "event": event, "returncode": 1 if event == "failure" else 0}
            )
            for event in ("failure", "repair_succeeded")
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = evaluate_confirmation(
        FREEZE,
        run_root=run_root,
        output_dir=tmp_path / "confirmation",
        repository_root=Path("."),
        run_code_guardrails=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["infrastructure_repairs"] == 1
    assert (report_path.parent / "infrastructure_events.jsonl").read_bytes() == (
        events.read_bytes()
    )


def test_confirmation_records_repair_with_the_same_locked_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An infrastructure retry records failure and successful same-config repair."""
    returncodes = iter((1, 0))
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            partial = (
                tmp_path / "runs/uci_benchmark/confirmation-smoke-seed-202-autompg"
            )
            (partial / "metrics").mkdir(parents=True)
            (partial / "run.json").write_text("{}", encoding="utf-8")
            (partial / "metrics/comparison.csv").write_text(
                "partial\n", encoding="utf-8"
            )
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(
        "experiments.uci_benchmark.confirmation.subprocess.run", fake_run
    )
    run_root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        run_confirmation_panel(FREEZE, run_root=run_root, smoke=True)
    run_confirmation_panel(FREEZE, run_root=run_root, smoke=True)

    events = [
        json.loads(line)
        for line in (run_root / "confirmation-smoke-infrastructure-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events] == ["failure", "repair_succeeded"]
    assert events[0]["config_sha256"] == events[1]["config_sha256"]


@pytest.mark.parametrize(
    ("coverage", "accepted_coverage", "width", "accepted_width", "center"),
    [
        ([0.121], [0.10], [0.40], [0.40], [0.05]),
        ([0.151, 0.089], [0.10, 0.10], [0.40], [0.40], [0.05]),
        ([0.10], [0.10], [0.349], [0.40], [0.05]),
        ([0.10], [0.10], [0.40], [0.40], [0.101]),
    ],
)
def test_frozen_guardrail_threshold_crossings_fail(
    coverage: list[float],
    accepted_coverage: list[float],
    width: list[float],
    accepted_width: list[float],
    center: list[float],
) -> None:
    """Each frozen recovery/agreement boundary independently blocks promotion."""
    result = assess_frozen_guardrail_thresholds(
        coverage_errors=coverage,
        accepted_coverage_errors=accepted_coverage,
        accepted_coverage_mae=0.10,
        width_ratios=width,
        accepted_width_ratios=accepted_width,
        center_differences=center,
    )
    assert result["passed"] is False


def test_frozen_guardrail_boundaries_pass_and_overwidth_only_triggers_review() -> None:
    """Exact margins pass; ratio above 1.25 requests review without rejection."""
    result = assess_frozen_guardrail_thresholds(
        coverage_errors=[0.15, 0.09],
        accepted_coverage_errors=[0.10, 0.10],
        accepted_coverage_mae=0.10,
        width_ratios=[0.35, 1.26],
        accepted_width_ratios=[0.40, 1.26],
        center_differences=[0.10, 0.05],
    )
    assert result == {"passed": True, "overwidth_review_triggered": True}
