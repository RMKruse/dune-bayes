# UCI benchmark panel

This experiment is the dune-bayes side of the common UCI evaluation harness
(ADR-0008, GitHub #102). The checked-in config pins the ten-dataset panel,
catalog IDs, response families, split seeds, architecture, and evaluation
budget. Auto MPG, concrete, energy, kin8nm, power, protein, wine, and yacht use
the Normal family; bike counts use NegativeBinomial; naval's bounded decay
coefficient uses Beta after moving exact endpoints inward by `EPS`.

Run the one-dataset CI tracer with:

```bash
uv run --extra experiments python experiments/uci_benchmark/run.py \
  experiments/uci_benchmark/config.yaml --smoke
```

Run one full dataset with `--dataset bike`, or omit the selector for the full
panel. Downloads are normalized to numeric CSVs under `data/cache/`; split
indices are created once under `data/splits/` and then reused byte-for-byte.
Both directories are ignored because later baseline experiments consume the
same local artifacts rather than committing upstream datasets.

Each dataset writes `metrics/<dataset>/nll.csv`, `crps.csv`, and
`calibration.csv`. NLL is the held-out posterior-predictive negative
log-likelihood (`logsumexp` over coherent draws), CRPS is the package's fair
sample estimator, and calibration is a ten-bin PIT reliability table
(randomized and seeded for bike). Runs land under `runs/`; after inspecting a
full run, promote the complete run directory to `results/` following the
repository experiment convention.
