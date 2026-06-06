# Verification spikes

> **PyTorch (ADR-0006).** These spikes were originally written against the legacy
> TensorFlow / Keras / TFP stack and have been **ported to PyTorch** along with the
> rest of the package. The two load-bearing ADR-0004 claims — KL aggregation across
> the real NAMLSS graph shape, and a save/load round-trip with variational weights
> intact — are **re-verified green on PyTorch** (torch 2.12 / Python 3.12). The TF
> originals live in git history. On torch the claims get *simpler*, not harder: KL
> is collected by an explicit module-walk (no `add_loss` magic to trust across
> boundaries), and save/load is a single `state_dict` + config path (no
> `.keras`/SavedModel/H5 format matrix, no weight-name-collision failure mode).

Throwaway scripts that prove the two load-bearing claims the dune-bayes design
rests on, **before** building out the real package. If either spike fails, the
corresponding ADR needs revisiting.

These are not package code. `variational_dense.py` here is a deliberately minimal
seed of the real `VariationalDense` (ADR-0004) — just enough to exercise the two
claims. The production module adds the flipout-style variance-reduction estimator
and the hierarchical-scale handle (ADR-0002), which these spikes do not need.

## What each spike proves

| Script | Claim under test | Backing decision |
|---|---|---|
| `spike_kl_propagation.py` | KL stashed inside `forward()` and summed by `collect_kl`'s module-walk aggregates from **nested per-feature sub-modules**, through a sum, through the family distribution head, into a single training loss; an optimizer step adds it to the NLL; the warm-up `beta` gates it; it's weighted as KL/N. | KL/N + warm-up (CONTEXT); ADR-0003 |
| `spike_serialization.py` | A closure-free `get_config`/`from_config` plus `state_dict` lets a model of these modules **save and load** with the variational weights intact (`max|Δw| = 0`) — retiring the old `tfp.layers` save/load fragility. | ADR-0004 (retires ADR-0003's open risk) |

Both scripts deliberately rebuild the **real NAMLSS graph shape** (each feature net
is its own `nn.Module`, summed, then a `torch.distributions` family head), because
that nesting is exactly where KL aggregation and serialization are most likely to
break.

## Running

Any recent PyTorch works; no version pin is needed (the old TF Python-version cap
is gone). Verified on torch 2.12 / Python 3.12.

```bash
# from repo root
uv venv .venv-torch --python 3.12
VIRTUAL_ENV=.venv-torch uv pip install -r spikes/requirements-spike.txt

cd spikes
../.venv-torch/bin/python variational_dense.py        # smoke check of the module
../.venv-torch/bin/python spike_kl_propagation.py     # Spike 1  — exit 0 = all PASS
../.venv-torch/bin/python spike_serialization.py      # Spike 2  — exit 0 = all PASS
# or both at once (activate the env first so `python` resolves):
./run_spikes.sh
```

Each script prints `[PASS]` / `[FAIL]` per check and exits non-zero if any hard
claim fails.

## Reading the results

- **Spike 1 all PASS** → the KL/N + warm-up machinery is safe to build: `collect_kl`
  reaches every variational module, `beta` gates it, it scales as KL/N, and it lands
  in the optimizer step.
- **Spike 1 fails check A** → the module-walk is missing KL from some nested
  sub-module; the real model's KL accounting (and `_iter_variational_layers`) needs
  fixing before trusting the training loss.
- **Spike 2 all PASS** → `state_dict` + config dict is the supported save path;
  ADR-0004's serialization claim holds on PyTorch.
- **Spike 2 fails** → the config is not sufficient to reconstruct the module, or
  weights don't survive the round-trip; ADR-0004's "supported, tested path" wording
  must soften.
