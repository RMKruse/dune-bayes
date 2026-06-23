# HMC agreement

This validation-only experiment fits the same small, fixed-prior Normal
distributional regression with dune-bayes mean-field VI and NumPyro NUTS. Two
correlated linear feature effects contribute to both location and raw scale; the
model has six scalar parameters in total. The correlation creates posterior
covariance that mean-field VI cannot represent, making band shrinkage measurable.
JAX and NumPyro remain experiment dependencies.

Before inference, the run evaluates the Torch and independently implemented JAX
log joints at the configured truth and requires agreement within `1e-10` absolute
tolerance. The canonical comparison uses pointwise 90% centered epistemic effect
bands. Its pre-registered agreement contract is:

- four NUTS chains, maximum R-hat at most 1.01, minimum bulk ESS at least 400,
  and zero divergences;
- median VI-to-NUTS band-width ratio below 1 for every feature/parameter;
- at least 90% of each VI band contained inside its NUTS band;
- median center difference no greater than 0.25 NUTS band widths.

Run the smoke or canonical study with:

```bash
uv run --extra experiments python experiments/hmc_agreement/run.py \
  experiments/hmc_agreement/config.yaml --smoke

uv run --extra experiments python experiments/hmc_agreement/run.py \
  experiments/hmc_agreement/config.yaml
```

Scratch output lands in `runs/`; reviewed canonical artifacts are promoted to
`results/canonical/`.

## Canonical result

The promoted run satisfies every pre-registered check: maximum R-hat is 1.0037,
minimum bulk ESS is 1330, and no divergences occurred. Every VI band is fully
contained by its NUTS counterpart. Median VI-to-NUTS width ratios range from
0.41 to 0.48, while normalized center differences range from 0.002 to 0.026.
This is the expected mean-field signature: matching centers with materially
narrower epistemic bands.
