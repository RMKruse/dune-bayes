# Canonical UCI benchmark evidence

Promoted full benchmark artifact for GitHub #131 and PRD-0003's
benchmark-comparator claim. This run was generated with:

```bash
.venv/bin/python experiments/uci_benchmark/run.py \
  experiments/uci_benchmark/config.yaml
```

`run.json` records `"smoke": false`. The comparison table contains the full
ten-dataset panel declared by `config.yaml` and four scored comparators:
`dune_bayes`, `BayesNAM-style (our implementation)`, `plain_mlp`, and
`deep_ensemble`. Each row is scored through the common harness metrics path for
NLL, CRPS, and PIT calibration.

`benchmark-claims.yaml` is the paper-facing gate manifest. It also records the
optional comparators that were not claimed in this canonical run:
`nampy_namlss`, `lanam`, and `bamlss_reference`. The NAMLSS process-boundary
smoke artifact remains promoted separately under `results/namlss-smoke/`;
BAMLSS fixture scoring remains covered by committed Auto MPG fixtures and
smoke-scale tests.

Some UCI datasets were available from the original UCI files but not through
`ucimlrepo` import on 2026-06-23. For this run, the local cache was populated
from the official UCI files for naval, protein, yacht, and bike before running
the harness. Naval's Beta response was moved into the open unit interval with
the package `EPS` contraction, matching the checked-in config.

Manual cache sources:

- naval: `https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip`
- protein: `https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv`
- yacht: `https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data`
- bike: `https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip`
