# Live NAMpy/NAMLSS smoke comparison

This promoted smoke artifact was generated through
`experiments/uci_benchmark/nampy_namlss_runner.py` with
`baselines.namlss.enabled: true`. It verifies the #104 harness contract: the
external TensorFlow/Keras/TFP NAMLSS process consumes the same persisted Auto
MPG split and preprocessed arrays as `dune_bayes`, then returns per-observation
`samples`, `log_density`, and `cdf` for the common `dune_bayes.metrics` scoring
path.

`metrics/comparison.csv` is the side-by-side smoke table. The
`metrics/autompg/parameter_bands.csv` and `variance_split.csv` files are
`dune_bayes`-only uncertainty artifacts; deterministic NAMLSS is scored on the
shared predictive quantities but does not provide posterior parameter bands or
an aleatoric/epistemic split.

This is not a published-number comparison. The original paper code drop is now
available under `namlss-paper-code/`, but the exact original configs for the
paper tables still need to be pinned before documenting proximity to published
numbers. Until then, published values remain a sanity-check target only, not the
basis for the benchmark comparison.
