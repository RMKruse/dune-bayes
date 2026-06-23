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

The `dune_bayes`, `BayesNAM-style (our implementation)`, `plain_mlp`,
`deep_ensemble`, optional `nampy_namlss`, optional `lanam`, and optional
`bamlss_reference` adapters prove the seam end-to-end.
`BayesNAM-style (our implementation)` is a labeled
degenerate dune-bayes config: Bayesian shape functions contribute only to the
Normal location parameter, while a point intercept learns one homoscedastic
scale. The built-in PyTorch sanity floors are `plain_mlp` and `deep_ensemble`:
each uses a homoscedastic Gaussian residual, and the ensemble mixes
independently initialized MLP predictives. Those baselines do not provide
dune-bayes's distributional separation of aleatoric family uncertainty from
epistemic uncertainty on shape functions. Per-baseline tables live below
`metrics/<dataset>/<model>/`; `metrics/comparison.csv` is the combined panel.
When the BayesNAM-style baseline is enabled, the run also writes
`figures/<dataset>/bayesnam_style_band_contrast.pdf`, contrasting its mean-only
epistemic bands with dune-bayes per-parameter bands.

## BAMLSS fixture reference

The BAMLSS route for #107 is HITL by design. The repo commits the seeded
fixture producer at `bamlss/run.R`, but it does not run R or BAMLSS inside the
Python/uv CI environment. A maintainer runs the script locally against the same
cached CSVs and persisted split `.npz` files created by the harness:

```bash
Rscript experiments/uci_benchmark/bamlss/run.R \
  --config experiments/uci_benchmark/config.yaml \
  --dataset autompg \
  --output-dir experiments/uci_benchmark/fixtures/bamlss \
  --seed 10701 \
  --predictive-samples 500
```

Add `--smoke` when generating the tiny CI tracer fixture. The script writes
`fixtures/bamlss/<dataset>/predictions.csv` with one row per held-out
observation: `log_density`, `cdf`, predictive quantiles (`q05`, `q50`, `q95`),
and response predictive draws `sample_0001` ... `sample_N`. It also writes
`provenance.json` beside the CSV with the script version, seed, date, split
path, package versions, and `sessionInfo()`. After those files are committed,
enable the disabled config block:

```yaml
baselines:
  bamlss_reference:
    enabled: true
    fixture_dir: fixtures/bamlss
```

The Python adapter only validates and scores these maintainer-produced
fixtures. It never reimplements BAMLSS and never imports R from the package
runtime.

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

## LA-NAM comparator

The LA-NAM route for #105 is the pinned git dependency route, not fixtures:
`fortuinlab/LA-NAM` is MIT-licensed, so
`experiments/uci_benchmark/requirements-lanam.txt` pins `laplace-skorch` at
commit `d6748ebcb1dd5b5c15ca3120c4dcc19667ead111` (`v0.2.0`). This dependency
file is separate from `pyproject.toml` because upstream LA-NAM requires NumPy
1.x while the package comparison stack uses ArviZ's NumPy-2/DataTree path. Keep
it disabled in the default config until intentionally running the optional
baseline:

```yaml
baselines:
  lanam:
    enabled: true
    python: ../../.venv-lanam/bin/python
    runner: lanam_runner.py
```

`lanam_runner.py` fits the authors' `LaplaceAdditiveNetRegressor` in a separate
process and returns per-observation Gaussian predictive samples, log-density,
and CDF for the common scorer. Its comparison row is labeled
`mean_only_laplace_location`: LA-NAM supplies Bayesian uncertainty over the
location/mean additive predictor, but it does not expose per-distribution-
parameter bands for scale or shape. Those parameter-band and variance-split
tables remain `dune_bayes`-only artifacts.
