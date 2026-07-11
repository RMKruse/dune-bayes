# Bounded Reproducibility Audit

Status: ready

## Dependency Readiness

Install command: `uv sync --locked --extra dev --extra experiments`
Ready: True

## Automated Checks

- ruff format: pass
- ruff lint: pass
- mypy: pass
- pytest core: pass
- disentanglement smoke: pass
- hmc_agreement smoke: pass
- jsu_showcase smoke: pass
- parameter_recovery smoke: pass
- parameter_recovery smoke: pass
- parameter_recovery smoke: pass
- uci_benchmark smoke: pass
- walking_skeleton smoke: pass
- experiment harness tests: pass
- HMC agreement smoke tests: pass

## Evidence And Artifacts

- Evidence manifest: ready (4 claims)
- Benchmark gate: ready
- Paper artifacts: ready
- Paper artifact outputs:
  - central-disentanglement__disentanglement.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-normal__recovery.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-normal__calibration.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-student-t__recovery.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-student-t__calibration.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-gamma__recovery.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-gamma__calibration.pdf
  - per-feature-per-parameter-epistemic-bands__effect_ribbons.pdf
  - vi-vs-nuts-limitation__vi_vs_nuts.pdf
  - per-feature-per-parameter-epistemic-bands__canonical-normal__calibration.csv
  - per-feature-per-parameter-epistemic-bands__canonical-normal__intercept_coverage.csv
  - per-feature-per-parameter-epistemic-bands__canonical-student-t__calibration.csv
  - per-feature-per-parameter-epistemic-bands__canonical-student-t__intercept_coverage.csv
  - per-feature-per-parameter-epistemic-bands__canonical-gamma__calibration.csv
  - per-feature-per-parameter-epistemic-bands__canonical-gamma__intercept_coverage.csv
  - per-feature-per-parameter-epistemic-bands__coverage.csv
  - per-feature-per-parameter-epistemic-bands__canonical__intercept_coverage.csv
  - vi-vs-nuts-limitation__parameter_intervals.csv
  - vi-vs-nuts-limitation__band_width_ratios.csv
  - benchmark-comparator-panel__comparison.csv
  - benchmark-comparator-panel__autompg__nll.csv
  - benchmark-comparator-panel__autompg__crps.csv
  - benchmark-comparator-panel__autompg__calibration.csv
  - benchmark-comparator-panel__autompg__variance_split.csv
  - benchmark-comparator-panel__naval__nll.csv
  - benchmark-comparator-panel__bike__nll.csv
  - evidence-summary.csv
  - provenance.json
  - artifact-metadata.yaml
  - reviewer-evidence-appendix.md

## Bounded Scope

- core package checks: automated
- experiment smokes: automated
- canonical evidence validation: automated
- benchmark publication gate: automated
- paper artifact assembly: automated
- full canonical reruns: manual

## Caveats

- Full canonical experiment reruns are manual
- The bounded audit regenerates smoke outputs and paper artifacts only
- Promoted canonical evidence remains the reviewed source for paper claims
