# Experiments

`experiments/` is the orchestration tier defined by ADR-0008 and GitHub #97.
Statistical capabilities belong in `src/dune_bayes`, orchestration belongs
here, and throwaway verification belongs in `spikes/`.

Each experiment directory contains:

- `config.yaml`: the seed, family, architecture, posterior draw count, data
  parameters, training parameters, and artifact destination needed to rerun it;
- `run.py`: a CLI accepting the config path and optional `--smoke`;
- `runs/`: ignored scratch output;
- `results/`: deliberately promoted, reviewable canonical output.

Run the walking skeleton with:

```bash
uv run --extra experiments python experiments/walking_skeleton/run.py \
  experiments/walking_skeleton/config.yaml
```

Use `--smoke` for the tiny CI workload. Full runs are manual and opt-in.

The paper-facing simulations currently include:

- `parameter_recovery/`: centered effect recovery and per-parameter coverage;
- `disentanglement/`: dense/noisy versus sparse/quiet variance decomposition.
- `uci_benchmark/`: shared-split UCI panel with NLL, CRPS, and PIT calibration.
- `hmc_agreement/`: fixed-prior VI versus NumPyro NUTS effect-band validation.

## Artifact convention

The harness writes one run beneath
`<artifact-root>/<experiment>/<run-name>/`, with:

```text
arrays/*.npz
figures/*.pdf
metrics/*.json or *.csv
config.yaml
run.json
```

To promote a run, inspect its metrics and figures, then copy the complete run
directory into the experiment's `results/` directory. Promotion is deliberate:
never point a config directly at `results/`, and never commit from `runs/`.
Record any paper-facing interpretation in the experiment-specific README.
