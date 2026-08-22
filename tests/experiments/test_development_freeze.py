"""Development-panel freeze boundaries (GitHub #180)."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

from experiments.uci_benchmark.development import freeze_development_panel

_MODELS = ("dune_bayes", "deterministic_namlss", "plain_mlp", "deep_ensemble")
_PROCEDURES = ("R", "C")
_SEEDS = (102, 112, 122)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one synthetic selected-fit table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(
    root: Path,
    *,
    dataset: dict[str, object],
    dataset_index: int,
    seed: int,
) -> None:
    """Write one complete dataset/seed result at the public artifact seam."""
    name = str(dataset["name"])
    run = root / "uci_benchmark" / f"development-seed-{seed}-{name}"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": seed,
                "datasets": [dataset],
                "development": {"dataset": name, "selection_seed": 102},
            }
        ),
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps({"experiment": "uci_benchmark", "seed": seed, "smoke": False}),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for procedure in _PROCEDURES:
        for model in _MODELS:
            model_offset = {
                "dune_bayes": 3.0 if procedure == "R" else 2.0,
                "deterministic_namlss": 1.5,
                "plain_mlp": 1.0,
                "deep_ensemble": 0.5,
            }[model]
            # Exercise the frozen exact-tie rule on one dataset and metric.
            ensemble_nll = model_offset
            if name == "autompg" and model == "deep_ensemble":
                ensemble_nll = 1.0
            failed_reference = (
                name == "autompg"
                and seed == 102
                and procedure == "R"
                and model == "dune_bayes"
            )
            rows.append(
                {
                    "comparison_role": "primary",
                    "dataset": name,
                    "family": dataset["family"],
                    "model": model,
                    "procedure": procedure,
                    "seed": seed,
                    "split_seed": dataset["split_seed"],
                    "uncertainty_scope": "predictive",
                    "n_test": 20 + dataset_index,
                    "mean_nll": dataset_index + ensemble_nll + seed / 10000,
                    "mean_crps": dataset_index / 10 + model_offset / 10,
                    "calibration_error": 0.18
                    if failed_reference
                    else model_offset / 100,
                    "status": "model_failure" if failed_reference else "completed",
                    "failure": "saturated_pit" if failed_reference else "",
                }
            )
    _write_csv(run / "metrics" / "comparison.csv", rows)

    if seed == 102:
        candidates = [
            {
                "candidate_id": f"candidate-{index:02d}",
                "hidden_dims": [16],
                "learning_rate": 0.001,
            }
            for index in range(1, 13)
        ]
        selected = {
            "dataset": name,
            "objective": "validation_nll",
            "tuning_seed": 102,
            "executed_candidate_count": 12,
            "procedures": {
                procedure: {
                    "models": {
                        model: {
                            "candidate_slots": 12,
                            "selected_candidate_id": "candidate-01",
                            "selected_configuration": candidates[0],
                            "trials": [
                                {
                                    "candidate_id": candidate["candidate_id"],
                                    "configuration": candidate,
                                    "status": "completed",
                                    "validation_nll": 1.0,
                                }
                                for candidate in candidates
                            ],
                        }
                        for model in _MODELS
                    }
                }
                for procedure in _PROCEDURES
            },
        }
        selection = run / "metrics" / name / "selection.json"
        selection.parent.mkdir(parents=True)
        selection.write_text(json.dumps(selected), encoding="utf-8")
    else:
        source = (
            root
            / "uci_benchmark"
            / f"development-seed-102-{name}"
            / "metrics"
            / name
            / "selection.json"
        )
        selected = json.loads(source.read_text(encoding="utf-8"))
        evaluation = run / "metrics" / name / "evaluation.json"
        evaluation.parent.mkdir(parents=True)
        evaluation.write_text(
            json.dumps(
                {
                    "dataset": name,
                    "seed": seed,
                    "selection_sha256": sha256(source.read_bytes()).hexdigest(),
                    "procedures": {
                        procedure: {
                            "models": {
                                model: {
                                    "candidate_slots": 1,
                                    "selected_candidate_id": "candidate-01",
                                    "selected_configuration": selected["procedures"][
                                        procedure
                                    ]["models"][model]["selected_configuration"],
                                    "trials": [
                                        {
                                            "candidate_id": "candidate-01",
                                            "configuration": selected["procedures"][
                                                procedure
                                            ]["models"][model][
                                                "selected_configuration"
                                            ],
                                            "status": "completed",
                                        }
                                    ],
                                }
                                for model in _MODELS
                            }
                        }
                        for procedure in _PROCEDURES
                    },
                }
            ),
            encoding="utf-8",
        )


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a generated development table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_full_development_panel_freezes_complete_confirmation_inputs(
    tmp_path: Path,
) -> None:
    """The freeze is dataset-first, hashed, and never promotes a claim."""
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    config["data"]["split_dir"] = str(split_dir)
    for dataset in config["datasets"]:
        np.savez(
            split_dir / f"{dataset['name']}.npz",
            train_indices=np.asarray([0, 1]),
            test_indices=np.asarray([2]),
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_root = tmp_path / "runs"
    for dataset_index, dataset in enumerate(config["datasets"], start=1):
        for seed in _SEEDS:
            _write_run(
                run_root,
                dataset=dataset,
                dataset_index=dataset_index,
                seed=seed,
            )

    output = tmp_path / "development-freeze"
    freeze_development_panel(
        config_path,
        run_root=run_root,
        output_dir=output,
    )

    raw = _read_rows(output / "raw_scores.csv")
    mappings = _read_rows(output / "comparator_mapping.csv")
    paired = _read_rows(output / "paired_gaps.csv")
    contrasts = _read_rows(output / "model_contrasts.csv")
    summaries = _read_rows(output / "win_loss_summary.csv")
    selected = _read_rows(output / "selected_configurations.csv")
    manifest = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))

    assert len(raw) == 10 * 3 * 2 * 4
    assert len(mappings) == 10 * 3
    tie = next(
        row
        for row in mappings
        if row["dataset"] == "autompg" and row["metric"] == "nll"
    )
    assert tie["comparator"] == "deep_ensemble"
    assert len(paired) == 10 * 3 * 3 * 3
    assert len(contrasts) == 10 * 3 * 3 * 3 * 2
    assert len(summaries) == 10 * 3 * (3 + 3 * 2)
    assert len(selected) == 10 * 2 * 4
    assert "pooled_mean" not in {key for row in summaries for key in row}
    assert manifest["confirmation_inputs_complete"] is True
    assert manifest["selected_candidate_model_failures"] == 0
    assert manifest["selected_candidate_failure_limit"] == 1
    assert manifest["package_default_promoted"] is False
    assert manifest["paper_claim_promoted"] is False
    assert manifest["confirmation"]["run_seeds"] == [202, 212, 222]
    assert len(manifest["confirmation"]["split_run_pairs"]) == 30
    assert "raw_scores.csv" in hashes["sha256"]
    assert "split_hashes.csv" in hashes["sha256"]
    assert "selections/autompg.json" in hashes["sha256"]
    assert "evaluations/seed-112-autompg.json" in hashes["sha256"]
    assert "run_metadata/seed-122-bike.json" in hashes["sha256"]


def test_checked_in_development_freeze_is_complete_and_hashed() -> None:
    """The executed panel remains reviewable without rerunning model fits."""
    root = Path("experiments/uci_benchmark/results/development-freeze")
    manifest = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))

    assert manifest["confirmation_inputs_complete"] is True
    assert manifest["selected_candidate_model_failures"] == 0
    assert manifest["confirmation"]["scores_generated"] is False
    assert manifest["package_default_promoted"] is False
    assert manifest["paper_claim_promoted"] is False
    assert len(_read_rows(root / "raw_scores.csv")) == 240
    assert len(_read_rows(root / "paired_gaps.csv")) == 270
    assert len(_read_rows(root / "comparator_mapping.csv")) == 30
    assert len(_read_rows(root / "split_hashes.csv")) == 10
    assert (
        sha256((root / "artifact_hashes.json").read_bytes()).hexdigest()
        == manifest["artifact_hashes_sha256"]
    )
    for relative, digest in hashes["sha256"].items():
        assert sha256((root / relative).read_bytes()).hexdigest() == digest


def test_freeze_rejects_drifted_split_and_incomplete_evaluation(
    tmp_path: Path,
) -> None:
    """Wrong-split scores and empty refit traces cannot certify completeness."""
    config = yaml.safe_load(
        Path("experiments/uci_benchmark/config.yaml").read_text(encoding="utf-8")
    )
    dataset = config["datasets"][0]
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    np.savez(
        split_dir / f"{dataset['name']}.npz",
        train_indices=np.asarray([0, 1]),
        test_indices=np.asarray([2]),
    )
    config["data"]["split_dir"] = str(split_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_root = tmp_path / "runs"
    for seed in _SEEDS:
        _write_run(run_root, dataset=dataset, dataset_index=1, seed=seed)

    comparison = (
        run_root
        / "uci_benchmark"
        / "development-seed-112-autompg"
        / "metrics"
        / "comparison.csv"
    )
    rows = _read_rows(comparison)
    rows[0]["split_seed"] = "999"
    rows[1]["mean_nll"] = ""
    rows[2]["status"] = "model_failure"
    rows[2]["failure"] = ""
    _write_csv(comparison, rows)
    evaluation = (
        run_root
        / "uci_benchmark"
        / "development-seed-122-autompg"
        / "metrics"
        / "autompg"
        / "evaluation.json"
    )
    evaluation.write_text(
        json.dumps({"dataset": "autompg", "seed": 122, "procedures": {}}),
        encoding="utf-8",
    )

    output = tmp_path / "freeze"
    freeze_development_panel(
        config_path,
        run_root=run_root,
        output_dir=output,
        smoke=True,
    )

    errors = json.loads((output / "freeze.json").read_text(encoding="utf-8"))[
        "completeness_errors"
    ]
    assert any("wrong split seed" in error for error in errors)
    assert any("incomplete evaluation trace" in error for error in errors)
    assert any("invalid completed metrics" in error for error in errors)
    assert any("missing failure classification" in error for error in errors)
