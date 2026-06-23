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

## Publication evidence manifest

`experiments/publication/evidence-manifest.yaml` is the paper-facing ledger for
promoted evidence. Each claim records its claim family, whether it requires full
paper evidence or smoke-only validation, the canonical artifact path(s), expected
artifact class, required files, and minimal provenance from `run.json` such as
experiment name and seed. Entries may also include `file_hashes` for artifacts
whose bytes should be frozen; the validator reports a stale-file failure when a
recorded SHA-256 no longer matches.

Validate the ledger with:

```bash
uv run --extra experiments python -m experiments.publication.evidence \
  experiments/publication/evidence-manifest.yaml --root .
```

The command is intentionally a publication gate over promoted `results/`
artifacts, not a runner for new experiments. Scratch output under `runs/` must be
inspected and promoted first. At the moment the benchmark/comparator claim is
listed but still points at a smoke artifact, so the gate reports `NOT READY`
until a full canonical UCI/comparator panel is promoted.
