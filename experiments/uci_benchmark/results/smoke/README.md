# Smoke result

CI-scale acceptance artifact for GitHub #103 and #106: four held-out Auto MPG
rows, one training epoch, 16 posterior draws, and 32 predictive samples. It
proves the shared adapter, combined table, and labeled BayesNAM-style
degenerate dune-bayes config end-to-end; it is not paper evidence.
`figures/autompg/bayesnam_style_band_contrast.pdf` is the smoke-scale contrast
between mean-only BayesNAM-style bands and dune-bayes per-parameter bands.
The complete panel must be run with the checked-in top-level `config.yaml`
and deliberately promoted separately.
