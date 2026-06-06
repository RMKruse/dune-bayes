# Issue 0027: Vectorize the T-draw posterior sweeps

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design,
> user stories, and governing ADRs (0001–0006). Motivated by the wall-clock
> profile in `spikes/spike_t_sweep_benchmark.py` (record its numbers in the PR).

## What to build

All three posterior-sweep workhorses run their T draws as Python-level loops —
T full module-tree dispatches per sweep:

- `draw_predictive` — `torch.stack([model.predict_params(X) for _ in range(T)])`
  (`sampling/log_lik_sampler.py`)
- `pointwise_log_lik` — `for t in range(T): model.family(summed_samples[t])...`
  (`sampling/log_lik_sampler.py`)
- `sample_effects` — `torch.stack([net(x) for _ in range(T)])`
  (`sampling/effect_sampler.py`)

The benchmark spike measured the headroom (loop per-draw over a
batched-compute lower bound; torch 2.12, 5-thread CPU, 20 features):

| n (rows) | headroom |
|---:|---:|
| 1 000 | 5.9× |
| 5 000 | 2.0× |
| 10 000 | 1.3× |
| ≥ 25 000 | 1.0× |

So the sweep is **dispatch-bound in the small-batch regime and FLOP-bound at
large n**. That makes vectorization worth doing for exactly the workloads this
package centers on — small-data Bayesian fits (ADR-0001's gold-standard
regime) and the grid-based plotting sweeps (`sample_effects` on a ~512-point
grid, predictive ribbons at `T_predict = 200`), which sit permanently in the
~6× zone regardless of dataset size. It is explicitly **not** a big-n WAIC
speedup: at 10⁵⁺ rows the loop is already at the compute bound on CPU, and the
lever there is BLAS threads / device, not the loop. Replace the loops with a
batched-draw implementation so per-draw cost approaches the lower bound in the
small-batch regime.

Two candidate mechanisms — implementer's choice, both pure torch:

1. **Sample-dimension reparameterization.** `VariationalLayer.forward` accepts a
   leading sample dimension `S`: noise is drawn with shape `(S, ...)`, one
   dispatch emits `(S, batch, units)` with S *independent* weight draws.
   Touches the atom but keeps everything in plain `nn.Module` land.
2. **`torch.func.functional_call` + `torch.func.vmap`** over a stacked `(T, ...)`
   draw of posterior weight samples. Leaves the atom untouched; requires the
   flipout/local-reparameterization path to be vmap-compatible.

Either way the change stays behind the existing public functions and behind the
ADR-0006 inference seam — a vmap-shaped sweep maps one-to-one onto `jax.vmap`
when the NumPyro backend lands, so this *protects* the JAX future rather than
pre-empting it.

### Constraints (non-negotiable)

- **Public signatures and return shapes unchanged**: `sample_effects` →
  `{name: (T, n, param_count)}` float32; `draw_predictive` → `PredictiveDraws`
  with `summed_samples (T, n, param_count)` and the `MixtureSameFamily`
  predictive; `pointwise_log_lik` → `(T, n)` float64.
- **Statistical semantics unchanged**: T *independent* posterior draws — batched
  noise must be fresh per sample, never one draw broadcast T ways. Closed-form
  KL, softplus positivity, and the KL-stash contract (numerical rules 1, 4, 5)
  are untouched; the sweeps run under `eval_mode` + `no_grad`, so KL collection
  in the training path must be unaffected.
- **float64 stays** for the log-lik scoring (`logsumexp`-over-draws dtype rule).
- **Chunk over T.** Fully batching `T = 1000` materializes hidden activations
  at `(T, n, hidden)` — ~128 GB at n=500k, width 64. The implementation chunks
  the sample dimension (internal knob with a documented default); peak memory
  must stay within a small factor of the current loop's.
- **Reproducibility rule respected**: within one model object under a fixed
  seed, not across freshly-built ones.

## Acceptance criteria

- [ ] No Python-level loop over T remains in `draw_predictive`,
      `pointwise_log_lik`, or `sample_effects`; draws are batched (sample-dim
      noise or `torch.func.vmap`) and chunked over T
- [ ] Shape tests: all three return exactly the documented shapes/dtypes
      (including float64 for `pointwise_log_lik`)
- [ ] Draw independence: variance across the T axis is statistically consistent
      with the loop implementation (MC-convergence archetype with a fixed seed
      and a commented tolerance — never a single-draw assert)
- [ ] `pointwise_log_lik` agrees with a loop reference given identical
      `summed_samples` (deterministic given draws; tolerance commented as float
      error, not MC noise)
- [ ] Existing sampling / compare / plotting tests stay green unmodified
- [ ] `spikes/spike_t_sweep_benchmark.py` re-run at small n (≈1 000 rows, the
      regime with measured ~6× headroom) shows per-draw cost within a small
      factor of its tiled lower bound, and large-n numbers are not regressed;
      before/after numbers recorded in the PR body (a local measurement,
      **not** a CI gate — spikes stay out of CI)

## Blocked by

None — can start immediately. Run the benchmark spike first to capture the
"before" numbers on the target machine.
