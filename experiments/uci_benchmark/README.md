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

For `dune_bayes`, the same posterior draws also write
`metrics/<dataset>/parameter_bands.csv` and
`metrics/<dataset>/variance_split.csv`. These tables are intentionally
Bayesian-only: deterministic comparators can be scored on the same held-out
predictive quantities, but they do not expose per-parameter bands or the
aleatoric/epistemic variance decomposition.

## Common predictive adapter

Every comparison model implements the two-method `BenchmarkAdapter` in
`adapters.py`:

```text
fit(train_data, *, smoke) -> None
predict(features, target, *, draws, predictive_samples, seed)
    -> PredictiveResult(samples, log_density, cdf)
```

`samples` has shape `(M, n)` and feeds `dune_bayes.metrics.crps`;
`log_density` and `cdf` each have shape `(n,)` and feed NLL and PIT-bin
calibration. The runner owns all scoring and passes every adapter the same
training `DataModule`, held-out feature tensors, targets, and persisted split.
This keeps model-specific evaluation code out of comparison tables.

The `dune_bayes`, `plain_mlp`, `deep_ensemble`, and optional `nampy_namlss`
adapters prove the seam end-to-end. The latter two built-in PyTorch baselines
are conventional sanity floors: each uses a homoscedastic Gaussian residual,
and the ensemble mixes independently initialized MLP predictives. They do not
provide dune-bayes's distributional separation of aleatoric family uncertainty
from epistemic uncertainty on shape functions. Per-baseline tables live below
`metrics/<dataset>/<model>/`; `metrics/comparison.csv` is the combined panel.

## Live NAMLSS comparator

`nampy_namlss` runs through `nampy_namlss_runner.py` in a separate TensorFlow
environment. That process boundary is the acceptance-critical part: the root
package remains Python 3.12/PyTorch-only, while the comparator interpreter can
use the old TF/Keras/TFP stack needed by the original NAMLSS paper code.

To enable it, point the config at the supplied `namlss-paper-code/` checkout
and at a TensorFlow-capable Python:

```yaml
baselines:
  namlss:
    enabled: true
    python: ../../.venv-nampy/bin/python
    runner: nampy_namlss_runner.py
    paper_code_dir: ../../namlss-paper-code
```

The runner consumes the common harness's preprocessed train/test arrays and
persisted split, then writes `samples`, `log_density`, and `cdf` back for
`dune_bayes.metrics` scoring. The supplied paper scripts expose Normal NAMLSS;
non-Normal comparator runs fail explicitly until matching original configs are
available. Published paper numbers are not used as comparison targets; once the
original configs are pinned, sanity-check notes belong beside the promoted
`results/` run.
