# Reviewer Evidence Appendix

This appendix is generated from the promoted publication evidence manifest.

## Reviewer Conventions

- epistemic effect ribbons are centered, epistemic-only credible ribbons for per-feature effects. They are not response-level predictive bands.
- response-level predictive bands combine epistemic + aleatoric uncertainty and are interpreted as prediction intervals.
- centered effect recovery is evaluated against centered posterior draws because additive shape functions are identified only up to level; intercept coverage is reported separately.
- Simulation coverage is measured and reported rather than asserted correct, preserving the mean-field VI narrowness limitation in ADR-0001.
- VI-vs-NUTS evidence is validation-only NUTS evidence from experiments/. dune-bayes does not ship an MCMC backend; ADR-0006 keeps JAX/NumPyro behind a future inference seam.
- Family parameterizations follow the package glossary: positivity uses softplus(x) + EPS, and Johnson's SU uses the scipy johnsonsu parameterization.

## Simulation Evidence

### central-disentanglement

The law-of-total-variance decomposition separates epistemic effect uncertainty from aleatoric family uncertainty in the synthetic dense/noisy versus sparse/quiet setting.

- Evidence class: simulation
- Promoted evidence: experiments/disentanglement/results/canonical
- Artifact-builder outputs:
  - figures/central-disentanglement__disentanglement.pdf

### per-feature-per-parameter-epistemic-bands

Per-feature, per-parameter epistemic effect bands are recoverable across Normal, Gamma, Student-t, and Johnson's SU publication simulations.

- Evidence class: simulation
- Promoted evidence: experiments/parameter_recovery/results/canonical-normal
- Artifact-builder outputs:
  - figures/per-feature-per-parameter-epistemic-bands__canonical-normal__recovery.pdf
  - figures/per-feature-per-parameter-epistemic-bands__canonical-normal__calibration.pdf
  - tables/per-feature-per-parameter-epistemic-bands__canonical-normal__calibration.csv
  - tables/per-feature-per-parameter-epistemic-bands__canonical-normal__intercept_coverage.csv

### per-feature-per-parameter-epistemic-bands

Per-feature, per-parameter epistemic effect bands are recoverable across Normal, Gamma, Student-t, and Johnson's SU publication simulations.

- Evidence class: simulation
- Promoted evidence: experiments/parameter_recovery/results/canonical-student-t
- Artifact-builder outputs:
  - figures/per-feature-per-parameter-epistemic-bands__canonical-student-t__recovery.pdf
  - figures/per-feature-per-parameter-epistemic-bands__canonical-student-t__calibration.pdf
  - tables/per-feature-per-parameter-epistemic-bands__canonical-student-t__calibration.csv
  - tables/per-feature-per-parameter-epistemic-bands__canonical-student-t__intercept_coverage.csv

### per-feature-per-parameter-epistemic-bands

Per-feature, per-parameter epistemic effect bands are recoverable across Normal, Gamma, Student-t, and Johnson's SU publication simulations.

- Evidence class: simulation
- Promoted evidence: experiments/parameter_recovery/results/canonical-gamma
- Artifact-builder outputs:
  - figures/per-feature-per-parameter-epistemic-bands__canonical-gamma__recovery.pdf
  - figures/per-feature-per-parameter-epistemic-bands__canonical-gamma__calibration.pdf
  - tables/per-feature-per-parameter-epistemic-bands__canonical-gamma__calibration.csv
  - tables/per-feature-per-parameter-epistemic-bands__canonical-gamma__intercept_coverage.csv

### per-feature-per-parameter-epistemic-bands

Per-feature, per-parameter epistemic effect bands are recoverable across Normal, Gamma, Student-t, and Johnson's SU publication simulations.

- Evidence class: simulation
- Promoted evidence: experiments/jsu_showcase/results/canonical
- Artifact-builder outputs:
  - figures/per-feature-per-parameter-epistemic-bands__effect_ribbons.pdf
  - tables/per-feature-per-parameter-epistemic-bands__coverage.csv
  - tables/per-feature-per-parameter-epistemic-bands__canonical__intercept_coverage.csv

## Validation Evidence

### vi-vs-nuts-limitation

Mean-field VI effect bands are compared against validation-only NUTS so under-coverage and band-width limitations are quantified rather than asserted away.

- Evidence class: validation
- Promoted evidence: experiments/hmc_agreement/results/canonical
- Artifact-builder outputs:
  - figures/vi-vs-nuts-limitation__vi_vs_nuts.pdf
  - tables/vi-vs-nuts-limitation__parameter_intervals.csv
  - tables/vi-vs-nuts-limitation__band_width_ratios.csv

## Real-Data Benchmark Evidence

### benchmark-comparator-panel

UCI benchmark evidence is a characterization panel backed by a full canonical ten-dataset run for dune-bayes, the BayesNAM-style degenerate baseline, plain MLP, and deep ensemble; optional NAMLSS, LA-NAM, and BAMLSS/R comparators are explicitly excluded with documented reasons, and the paper does not claim universal predictive dominance from this panel.

- Evidence class: real_data_benchmark
- Promoted evidence: experiments/uci_benchmark/results/canonical
- Artifact-builder outputs:
  - tables/benchmark-comparator-panel__comparison.csv
  - tables/benchmark-comparator-panel__autompg__nll.csv
  - tables/benchmark-comparator-panel__autompg__crps.csv
  - tables/benchmark-comparator-panel__autompg__calibration.csv
  - tables/benchmark-comparator-panel__autompg__variance_split.csv
  - tables/benchmark-comparator-panel__naval__nll.csv
  - tables/benchmark-comparator-panel__bike__nll.csv
