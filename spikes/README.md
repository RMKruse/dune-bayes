# Verification spikes

Throwaway scripts that prove the two load-bearing TFP-behaviour claims the
neural-bamlss design rests on, **before** building out the real package. If either
spike fails on the target TF/TFP version, the corresponding ADR needs revisiting.

These are not package code. `variational_dense.py` here is a deliberately minimal
seed of the real `VariationalDense` (ADR-0004) — just enough to exercise the two
claims. The production layer adds the flipout-style variance-reduction estimator
and the hierarchical-scale handle (ADR-0002), which these spikes do not need.

## What each spike proves

| Script | Claim under test | Backing decision |
|---|---|---|
| `spike_kl_propagation.py` | KL emitted via `add_loss` inside `call()` propagates from **nested per-feature sub-models**, through a sum, through `tfp.layers.DistributionLambda`, into the **outer** `model.losses`; `.fit()` adds it to the NLL; the warm-up `beta` gates it; it's weighted as KL/N. | KL/N + warm-up (CONTEXT); ADR-0003 |
| `spike_serialization.py` | A closure-free `get_config`/`from_config` lets a model of these layers **save and load** with the variational weights intact — retiring the `tfp.layers` save/load fragility. | ADR-0004 (retires ADR-0003's open risk) |

Both scripts deliberately rebuild the **real NAMLSS graph shape** (each feature net
is its own `tf.keras.Model`, summed, then `DistributionLambda(family)`), because
that nesting is exactly where `add_loss` propagation and serialization are most
likely to break.

## Running

The target environment is **TF ≤ 2.15.1 / Keras < 3.0 / TFP ≤ 0.23** (from
`NAMpy/requirements.txt`). TF 2.15 needs **Python 3.9–3.11** — the system default
here is 3.14, which TF does not support, so create a dedicated env first:

```bash
# from repo root
python3.11 -m venv .venv-spike
source .venv-spike/bin/activate
pip install -r spikes/requirements-spike.txt

cd spikes
python variational_dense.py        # smoke check of the layer itself
python spike_kl_propagation.py     # Spike 1  — exit 0 = all PASS
python spike_serialization.py      # Spike 2  — exit 0 = all PASS
# or both at once:
./run_spikes.sh
```

Each script prints `[PASS]` / `[FAIL]` / `[SKIP]` per check and exits non-zero if
any hard claim fails. `[SKIP]` (serialization only) means a particular save format
isn't usable on this TF build — that's information, not failure, as long as at
least one format round-trips.

## Reading the results

- **Spike 1 all PASS** → the KL/N + warm-up machinery is safe to build on the
  existing `compile`/`fit` surface; no custom train loop needed.
- **Spike 1 fails check A** → `add_loss` is being dropped across the sub-model or
  `DistributionLambda` boundary; the real model would need to collect KL manually
  (e.g. surface sub-model `.losses` explicitly) — revisit ADR-0003's "KL auto via
  add_loss" claim.
- **Spike 2 all PASS for ≥1 format** → record which format(s) work as the
  supported save path; ADR-0004's serialization claim holds.
- **Spike 2 all formats SKIP/FAIL** → the closure-free config is not sufficient on
  this TF version; ADR-0004 needs a fallback (e.g. save/load weights-only +
  rebuild-from-formula), and its "supported, tested path" wording must soften.
