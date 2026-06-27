"""Manuscript claim-ledger boundary tests (GitHub #143, parent PRD #142)."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.publication.manuscript import validate_claim_ledger

LEDGER = Path("docs/manuscript/claim-ledger.yaml")
MANUSCRIPT = Path("docs/manuscript/dune-bayes-paper.md")
BENCHMARK_CLAIMS = Path(
    "experiments/uci_benchmark/results/canonical/benchmark-claims.yaml"
)


def _write_evidence_manifest(root: Path) -> Path:
    """Write a minimal promoted-evidence manifest fixture."""
    manifest = {
        "version": 1,
        "claims": [
            {
                "id": "central-disentanglement",
                "family": "disentanglement",
                "statement": "Variance decomposition separates uncertainty.",
                "requires": "full",
                "evidence": {
                    "path": "experiments/disentanglement/results/canonical",
                    "artifact_class": "simulation",
                    "expected_files": ["run.json"],
                },
            },
            {
                "id": "benchmark-comparator-panel",
                "family": "benchmark_comparator",
                "statement": "Benchmarks characterize real-data behavior.",
                "requires": "full",
                "evidence": {
                    "path": "experiments/uci_benchmark/results/canonical",
                    "artifact_class": "real_data_benchmark",
                    "expected_files": ["run.json"],
                },
            },
        ],
    }
    path = root / "evidence-manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _write_ledger(root: Path) -> Path:
    """Write a valid compact manuscript ledger fixture."""
    ledger = {
        "version": 1,
        "evidence_manifest": "evidence-manifest.yaml",
        "claims": [
            {
                "id": "central-disentanglement",
                "paper_claim": ("DUNE separates epistemic and aleatoric uncertainty."),
                "evidence": [
                    {
                        "evidence_class": "simulation",
                        "promoted_artifact": (
                            "experiments/disentanglement/results/canonical"
                        ),
                    }
                ],
                "intended_outputs": ["figures/central-disentanglement.pdf"],
                "limitation_note": "Synthetic evidence is directional.",
            },
            {
                "id": "benchmark-comparator-panel",
                "paper_claim": "DUNE is characterized on real data.",
                "evidence": [
                    {
                        "evidence_class": "real_data_benchmark",
                        "promoted_artifact": (
                            "experiments/uci_benchmark/results/canonical"
                        ),
                    }
                ],
                "intended_outputs": ["tables/uci-characterization.csv"],
                "limitation_note": "Benchmarks are not leaderboard claims.",
            },
        ],
    }
    path = root / "claim-ledger.yaml"
    path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")
    return path


def test_valid_claim_ledger_resolves_promoted_evidence(tmp_path: Path) -> None:
    """A complete ledger links manuscript claims to promoted manifest evidence."""
    _write_evidence_manifest(tmp_path)
    ledger_path = _write_ledger(tmp_path)

    report = validate_claim_ledger(ledger_path, root=tmp_path)

    assert report.ready is True
    assert report.claim_count == 2
    assert report.failures == ()


def test_claim_ledger_rejects_unknown_claim_id(tmp_path: Path) -> None:
    """Manuscript claim IDs must resolve to the promoted evidence manifest."""
    _write_evidence_manifest(tmp_path)
    ledger_path = _write_ledger(tmp_path)
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"][0]["id"] = "not-promoted"
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

    report = validate_claim_ledger(ledger_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "not-promoted: claim id is not present in evidence-manifest.yaml",
    )


def test_claim_ledger_requires_evidence_classes(tmp_path: Path) -> None:
    """Each ledger evidence row must name its evidence class explicitly."""
    _write_evidence_manifest(tmp_path)
    ledger_path = _write_ledger(tmp_path)
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"][0]["evidence"][0].pop("evidence_class")
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

    report = validate_claim_ledger(ledger_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: evidence.evidence_class is required",
        "central-disentanglement: evidence row does not match promoted manifest "
        "entry (, experiments/disentanglement/results/canonical)",
    )


def test_claim_ledger_rejects_scratch_paths(tmp_path: Path) -> None:
    """Scratch experiment runs cannot become paper evidence through the ledger."""
    _write_evidence_manifest(tmp_path)
    ledger_path = _write_ledger(tmp_path)
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"][0]["evidence"][0]["promoted_artifact"] = (
        "experiments/disentanglement/runs/manual/candidate"
    )
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False), encoding="utf-8")

    report = validate_claim_ledger(ledger_path, root=tmp_path)

    assert report.ready is False
    assert report.failures == (
        "central-disentanglement: scratch artifact paths under runs/ cannot be used",
        "central-disentanglement: evidence row does not match promoted manifest "
        "entry (simulation, experiments/disentanglement/runs/manual/candidate)",
    )


def test_checked_in_claim_ledger_has_paper_contract() -> None:
    """The manuscript ledger includes the required evidence classes and outputs."""
    report = validate_claim_ledger(LEDGER, root=Path("."))
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    evidence_classes = {
        evidence["evidence_class"]
        for claim in ledger["claims"]
        for evidence in claim["evidence"]
    }

    assert report.ready is True
    assert report.failures == ()
    assert {"simulation", "real_data_benchmark"}.issubset(evidence_classes)
    for claim in ledger["claims"]:
        assert claim["limitation_note"].strip()
        assert claim["intended_outputs"]


def test_manuscript_scaffold_names_sections_and_uncertainty_terms() -> None:
    """The source scaffold contains the paper-facing sections and thesis terms."""
    text = MANUSCRIPT.read_text(encoding="utf-8")

    for heading in [
        "## Introduction",
        "## Related Work",
        "## Model And Methods",
        "## Experiments",
        "## Limitations",
        "## Reproducibility",
        "## Citation And Artifact Notes",
    ]:
        assert heading in text
    for term in [
        "shape function",
        "BayesianNAMLSS",
        "epistemic uncertainty",
        "aleatoric uncertainty",
        "variance decomposition",
    ]:
        assert term in text
    for adr in ["ADR-0001", "ADR-0003", "ADR-0006", "ADR-0007", "ADR-0008"]:
        assert adr in text


def test_methods_section_covers_adr_backed_math_contract() -> None:
    """The methods narrative covers the math and exclusions required by #144."""
    methods = _section(
        MANUSCRIPT.read_text(encoding="utf-8"),
        "## Model And Methods",
        "## Experiments",
    ).lower()

    concept_groups = [
        ("bayesiannamlss", "namlss", "shape function", "family parameter"),
        ("negative elbo", "kl/n", "mean-field", "prior"),
        ("local reparameterization", "coherent global", "posterior weight"),
        ("posterior predictive mixture", "epistemic", "aleatoric"),
        ("law of total variance", "variance decomposition"),
        ("effect ribbons", "response-level bands", "effect centering"),
        ("intercept coverage", "epistemic-only", "full predictive"),
        ("softplus(x) + eps", "numerical floor", "family parameterizations"),
        ("waic", "psis-loo", "crps", "pit", "coverage"),
        ("no bayes factors", "literal bayes factors"),
    ]

    missing = [
        token for group in concept_groups for token in group if token not in methods
    ]
    assert missing == []


def test_limitations_section_names_reviewer_scope_boundaries() -> None:
    """The paper limitations preserve the explicit reviewer caveats for #148."""
    limitations = _section(
        MANUSCRIPT.read_text(encoding="utf-8"),
        "## Limitations",
        "## Reproducibility",
    )
    normalized = " ".join(limitations.lower().split())

    required_terms = [
        "mean-field vi",
        "under-cover",
        "validation-only nuts",
        "experiments/hmc_agreement/results/canonical",
        "not a shipped package backend",
        "waic",
        "psis-loo",
        "elbo",
        "literal bayes factors",
        "out of scope",
        "coverage is measured and reported",
        "not asserted nominally correct",
        "no post-hoc band inflation",
        "conformal calibration",
        "recalibration",
    ]

    assert [term for term in required_terms if term not in normalized] == []


def test_benchmark_section_matches_accepted_comparator_scope() -> None:
    """The benchmark prose honors comparator decision #145 for slice #147."""
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    benchmark = _section(
        manuscript,
        "### UCI Benchmark Characterization",
        "## Limitations",
    )
    normalized = " ".join(benchmark.lower().split())
    claim = yaml.safe_load(BENCHMARK_CLAIMS.read_text(encoding="utf-8"))["claims"][0]
    scope = claim["scope_decision"]

    for dataset in claim["datasets"]:
        assert dataset["name"] in normalized
    for family in {dataset["family"] for dataset in claim["datasets"]}:
        family_spellings = {family, family.replace("_", " ")}
        assert any(spelling in normalized for spelling in family_spellings)
    for metric in claim["metrics"]:
        assert metric in normalized

    for scored_comparator in [
        "full BayesianNAMLSS implementation",
        "BayesNAM-style",
        "plain MLP",
        "deep ensemble",
    ]:
        assert scored_comparator.lower() in normalized
    assert "degenerate configuration of the same package" in normalized
    assert "not a canonical external implementation" in normalized

    for excluded in scope["excluded_external_comparators"]:
        assert excluded["manuscript_name"].lower() in normalized
    assert "documented exclusions" in normalized
    assert "experiments/uci_benchmark/results/canonical" in benchmark
    assert "not the namlss smoke tracer" in normalized
    assert "not scratch runs" in normalized
    assert "predictive scoring" in normalized
    assert "uncertainty-structure contribution" in normalized
    assert "not a claim of universal predictive dominance" in normalized

    unsupported_claims = [
        "state-of-the-art",
        "leaderboard",
        "outperforms all",
        "dominates all",
    ]
    assert [phrase for phrase in unsupported_claims if phrase in normalized] == []


def _section(text: str, start: str, end: str) -> str:
    """Return a Markdown section bounded by two headings."""
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]
