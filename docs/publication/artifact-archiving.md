# Artifact Archiving And DOI Deposit

This checklist describes how the `v0.1.0-paper` code state maps to a DOI-backed
artifact deposit. Zenodo is the planned archive provider unless the authors
choose a different repository before tagging.

## Deposit Contents

Archive these items together:

- source code at the exact `v0.1.0-paper` tag;
- `CITATION.cff` with the final author-approved paper/preprint citation;
- promoted canonical results declared by
  `experiments/publication/evidence-manifest.yaml`;
- benchmark claim metadata from
  `experiments/uci_benchmark/results/canonical/benchmark-claims.yaml`;
- generated paper artifacts from
  `experiments/publication/paper-artifacts`;
- bounded reproducibility audit reports:
  `experiments/publication/reproducibility-audit/audit-report.json` and
  `experiments/publication/reproducibility-audit/audit-report.md`.

Do not archive scratch `experiments/*/runs/` output as paper evidence. Scratch
runs must be inspected and promoted into `results/` before they can enter the
deposit.

## Tag-To-DOI Flow

1. Run the bounded audit from a clean checkout with
   `uv sync --locked --extra dev --extra experiments`.
2. Build paper artifacts with `experiments.publication.artifacts` or rely on the
   audit output if it was just regenerated.
3. Confirm `experiments/publication/release-metadata.yaml` names the target tag,
   release notes, evidence manifest, and audit report paths.
4. Replace pending `CITATION.cff` paper fields with the author-approved
   preprint/submission citation and DOI if available.
5. Create the `v0.1.0-paper` tag on the audited commit.
6. Draft the Zenodo deposit from that tag, attaching promoted results and paper
   artifact outputs when the archive integration does not include them
   automatically.
7. Record the reserved DOI in the release notes, `CITATION.cff`, and manuscript
   artifact instructions before making the record public.

The DOI deposit should make the reviewer path clear: code comes from the tag,
claim evidence comes from promoted `results/`, generated manuscript-facing
tables/figures come from `paper-artifacts`, and audit status comes from
`audit-report.json` plus `audit-report.md`.
