# Disentanglement centerpiece

This experiment is the paper-facing simulation for GitHub #99. The left half
of the covariate domain contains 80% of the observations and deliberately high
Normal response noise; the right half contains 20% of the observations and low
response noise. The fitted `BayesianNAMLSS` is decomposed through the public
`variance_decomposition` metric, so the figure and run log use the same coherent
posterior draws.

Regenerate the canonical scratch run from the committed config and seed:

```bash
uv run --extra experiments python experiments/disentanglement/run.py \
  experiments/disentanglement/config.yaml
```

Use `--smoke` for the reduced CI workload. Scratch output is written below
`runs/`; the inspected paper evidence is promoted to `results/canonical/`.

## Canonical result

Seed 9901 gives an epistemic sparse/dense regional-mean ratio of 1.316 and an
aleatoric dense/sparse ratio of 3.983. Thus the fitted decomposition follows the
two distinct known-by-construction drivers in their pre-registered directions:
epistemic uncertainty rises with data sparsity, while aleatoric uncertainty
rises with inherent response noise. These are directional simulation results,
not a claim that mean-field VI produces calibrated band widths (ADR-0001).
