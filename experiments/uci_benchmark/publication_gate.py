"""Benchmark publication gate for paper evidence (ADR-0008, GitHub #130)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class _Gap:
    claim: str
    message: str

    def format(self) -> str:
        """Return the actionable publication-gate message."""
        return f"{self.claim}: {self.message}"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    """Read one YAML mapping from disk."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _load_json(path: Path) -> Mapping[str, Any]:
    """Read one JSON mapping from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one promoted result table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_manifest_path(manifest: Path, value: object) -> Path:
    """Resolve result paths relative to the manifest that declares them."""
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def _configured_families(config: Mapping[str, Any]) -> dict[str, str]:
    """Map configured dataset names to response families."""
    datasets = config.get("datasets", [])
    if not isinstance(datasets, Sequence):
        raise ValueError("Benchmark config must contain a datasets list.")
    families: dict[str, str] = {}
    for dataset in datasets:
        if isinstance(dataset, Mapping):
            families[str(dataset["name"])] = str(dataset["family"])
    return families


def _comparison_rows(result: Path) -> dict[tuple[str, str], Mapping[str, str]]:
    """Index the canonical comparison table by dataset and model."""
    rows = _read_csv(result / "metrics" / "comparison.csv")
    return {(row["dataset"], row["model"]): row for row in rows}


def _exclusion_keys(
    claim: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    """Collect documented dataset/baseline/metric exclusions."""
    keys: set[tuple[str, str, str]] = set()
    for source in (
        claim.get("exclusions", []),
        run_metadata.get("benchmark_exclusions", []),
    ):
        if not isinstance(source, Sequence):
            raise ValueError("Benchmark exclusions must be lists.")
        for exclusion in source:
            if not isinstance(exclusion, Mapping):
                raise ValueError("Benchmark exclusions must be mappings.")
            reason = str(exclusion.get("reason", "")).strip()
            if not reason:
                raise ValueError("Benchmark exclusions require a non-empty reason.")
            keys.add(
                (
                    str(exclusion["dataset"]),
                    str(exclusion["baseline"]),
                    str(exclusion["metric"]),
                )
            )
    return keys


def _metric_path(result: Path, dataset: str, baseline: str, metric: str) -> Path:
    """Return the public metric table path written by the UCI harness."""
    metric_dir = result / "metrics" / dataset
    if baseline != "dune_bayes":
        metric_dir = metric_dir / baseline
    return metric_dir / f"{metric}.csv"


def _is_excluded(
    exclusions: set[tuple[str, str, str]],
    dataset: str,
    baseline: str,
    metric: str,
) -> bool:
    """Return whether one metric or its whole comparator is excluded."""
    return (dataset, baseline, metric) in exclusions or (
        dataset,
        baseline,
        "*",
    ) in exclusions


def _validate_claim(manifest: Path, claim: Mapping[str, Any]) -> list[_Gap]:
    """Validate one benchmark claim against promoted canonical artifacts."""
    claim_id = str(claim["id"])
    result = _resolve_manifest_path(manifest, claim["result"])
    gaps: list[_Gap] = []

    run_metadata = _load_json(result / "run.json")
    if str(claim.get("evidence", "full")) == "full" and bool(
        run_metadata.get("smoke", False)
    ):
        gaps.append(
            _Gap(
                claim_id,
                f"{result} is smoke evidence; full benchmark evidence is required",
            )
        )

    config = _load_yaml(result / "config.yaml")
    configured_families = _configured_families(config)
    comparison = _comparison_rows(result)
    exclusions = _exclusion_keys(claim, run_metadata)
    baselines = [str(item) for item in claim.get("baselines", [])]
    metrics = [str(item) for item in claim.get("metrics", [])]

    for dataset in claim.get("datasets", []):
        if not isinstance(dataset, Mapping):
            raise ValueError(f"{claim_id} datasets must be mappings.")
        dataset_name = str(dataset["name"])
        family = str(dataset["family"])
        if configured_families.get(dataset_name) != family:
            gaps.append(
                _Gap(
                    claim_id,
                    (
                        f"{dataset_name} family gap: claimed {family}, "
                        f"configured {configured_families.get(dataset_name)!r}"
                    ),
                )
            )
        for baseline in baselines:
            if metrics and all(
                _is_excluded(exclusions, dataset_name, baseline, metric)
                for metric in metrics
            ):
                continue
            if (dataset_name, baseline) not in comparison:
                gaps.append(
                    _Gap(
                        claim_id,
                        (
                            f"missing comparator row for dataset={dataset_name}, "
                            f"baseline={baseline}"
                        ),
                    )
                )
            for metric in metrics:
                if _is_excluded(exclusions, dataset_name, baseline, metric):
                    continue
                path = _metric_path(result, dataset_name, baseline, metric)
                if not path.is_file():
                    gaps.append(
                        _Gap(
                            claim_id,
                            (
                                f"missing metric file for dataset={dataset_name}, "
                                f"baseline={baseline}, metric={metric}: {path}"
                            ),
                        )
                    )
    return gaps


def _validate_manifest(manifest: Path) -> tuple[list[str], list[_Gap]]:
    """Validate all benchmark claims in one manifest."""
    data = _load_yaml(manifest)
    claims = data.get("claims", [])
    if not isinstance(claims, Sequence):
        raise ValueError("Benchmark manifest must contain a claims list.")
    ready: list[str] = []
    gaps: list[_Gap] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise ValueError("Each benchmark claim must be a mapping.")
        claim_gaps = _validate_claim(manifest, claim)
        if claim_gaps:
            gaps.extend(claim_gaps)
        else:
            ready.append(str(claim["id"]))
    return ready, gaps


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark publication gate.

    Args:
        argv: Optional CLI arguments; defaults to ``sys.argv``.

    Returns:
        Process exit code, where 0 means every claim is publication-ready.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    ready, gaps = _validate_manifest(args.manifest)
    for claim_id in ready:
        print(f"{claim_id}: ready")
    for gap in gaps:
        print(gap.format(), file=sys.stderr)
    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
