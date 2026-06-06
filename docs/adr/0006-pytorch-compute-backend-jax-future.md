# 6. PyTorch as the compute backend, with JAX as the numerical future

Date: 2026-06-03

## Status

Accepted. **Amends** ADR-0001, ADR-0003, ADR-0004 (retargets their
TensorFlow/Keras/TFP-specific claims to PyTorch; their statistical decisions are
unchanged — see *Relationship to other ADRs*).

> **Amendment (2026-06-03): reuse → reimplement.** This ADR's *Scope* section
> originally said dune-bayes "reuses NAMpy's TF-free Python directly — the
> formula parser (`nampy/formulas/`) and the `ShapeFunctionRegistry` pattern …
> imported or adapted as-is." That is **superseded**: dune-bayes now treats the
> **entire `NAMpy/` tree as reference-only** and **reimplements the formula parser
> and `ShapeFunctionRegistry` from scratch** in its own `src/dune_bayes/`
> namespace. NAMpy is read to see how things were done before, but **no NAMpy code
> is imported, vendored, or ported.** Rationale: a single clean PyTorch namespace
> with no cross-package coupling, at the modest cost of reimplementing the parser.
> Consequence: issue slices that referenced reuse (notably 0013, which cites
> `nampy/formulas/formulas.py:223`) describe the *target behavior* of the
> reimplemented parser, not a reused module. See `CLAUDE.md` for the process rules.

## Context

dune-bayes is being built by Bayesian-ifying NAMpy, which today runs on
**TensorFlow / Keras 2 / TensorFlow-Probability (TFP)**. Before the Bayesian half
is implemented, we need to decide whether that is the backend we commit to for the
package's life. The decision is forced now because it is far cheaper now than
later.

Findings that drive the decision:

- **The binding dependency is TFP, not TF.** TF-the-tensor-library is replaceable
  mechanically. The real commitment is to TFP idioms: families return `tfd.*`
  distributions, every LSS head is a `tfp.layers.DistributionLambda`, the posterior
  predictive is `tfd.MixtureSameFamily`, the loss is `-y_hat.log_prob(y)`. TFP is
  concentrated in 5 files (`families.py` + the four LSS heads) but is the
  load-bearing part of the design.
- **TFP is the most stagnant part of the stack.** We are pinned to TF 2.15.1 /
  Keras 2.15 / TFP 0.23, which caps Python at ≤3.11; TFP development has slowed
  markedly; and Keras 3 (now multi-backend) does **not** carry TFP forward — so
  there is no in-place upgrade path that keeps the dependency we actually rely on.
- **We already chose not to use TFP's variational layers.** ADR-0004 ships an
  in-house `VariationalDense` precisely because the stock `DenseVariational` /
  `DenseFlipout` did not fit. So our differentiating atom gains almost nothing from
  TFP — what TFP buys us is the distribution zoo, `DistributionLambda`, and
  `MixtureSameFamily`, all of which have direct equivalents elsewhere.
- **The Bayesian half is spikes-only.** No production Bayesian code exists yet.
  This is the cheapest moment we will ever have to change backend.
- **Ecosystem.** PyTorch dominates the research and contributor pool;
  `torch.distributions` covers our families and `MixtureSameFamily` directly;
  **arviz** (already our comparison spine, ADR-0003) is native to the
  PyMC/Pyro/NumPyro world, not the TFP world — i.e. TF is the odd backend out in
  the Python-Bayesian ecosystem we already live in.

## Decision

**Adopt PyTorch as the compute and autodiff backend, replacing TensorFlow / Keras /
TensorFlow-Probability, and designate JAX as the intended substrate for the
numerical/performance future and the deferred MCMC backend.**

### PyTorch — the present backend

- **Tensors / autodiff / nets:** PyTorch (`torch`, `torch.nn`). NAMpy's
  deterministic Keras layers are ported to `nn.Module`s; the deterministic
  baselines are retained (CONTEXT goal).
- **Distributions:** `torch.distributions` for family `log_prob` and for the
  `MixtureSameFamily` posterior predictive (ADR-0003). Replaces `tfd.*` /
  `DistributionLambda` one-for-one.
- **`VariationalDense` survives as a decision (ADR-0004).** It becomes an
  `nn.Module` rather than a Keras layer. Everything load-bearing carries over
  unchanged in spirit: mean-field Normal posterior (`loc` + softplus `scale`), a
  **serializable float / hierarchical-handle prior config** (never a closure),
  closed-form Gaussian–Gaussian KL, and the flipout-style variance-reduction
  estimator flag. **Serialization moves to `state_dict` + a small config dict**,
  which *dissolves* the `.keras`/SavedModel-vs-H5 fragility ADR-0004 spent effort
  on — there is no equivalent weight-name-collision failure mode.
- **KL accounting without `add_loss`.** Keras's `add_loss` auto-propagation is
  replaced by explicitly collecting per-module KL by walking the module tree (the
  TF spike already walks the tree for `set_kl_beta`, so this is not new behavior)
  and adding KL/N — annealed by the warm-up β — in the training step.
- **Training UX preserved.** The familiar `compile(...)`/`fit()` surface
  (CONTEXT, ADR-0003) is re-provided by a thin trainer (a hand-rolled loop or
  PyTorch Lightning), so the user-facing ergonomics survive the backend change.
- **arviz unchanged.** It is backend-agnostic; `to_inference_data()` + WAIC /
  PSIS-LOO / `compare()` (ADR-0003) are unaffected.

### JAX — the numerical future

JAX is designated the intended substrate for the package's numerical/performance
future and for the **deferred MCMC backend** that ADR-0001 explicitly reserved a
seam for:

- **NumPyro / BlackJAX** provide first-class mean-field *and* full VI plus
  HMC/NUTS — turning ADR-0001's "gold-standard small-data backend" from a research
  project into a library integration.
- **`vmap` maps directly onto our `T`-sample posterior-predictive sweeps**
  (`T_predict = 200`, `T_eval = 1000`, ADR-0003), making
  `sample_posterior_predictive` both elegant and fast.
- **arviz already bridges NumPyro**, so the comparison story is continuous across
  backends.

To keep this future open, the PyTorch implementation **must keep the
distribution/inference layer behind a thin seam** — families as small modules,
inference engine swappable — exactly as ADR-0001 already requires for the MCMC
backend. JAX/NumPyro is therefore a planned numerical-future backend that slots in
behind that seam, not a rewrite. PyTorch is the pragmatic *present*; JAX is the
*numerical future* — this is a staged path, not a dead end.

### Scope: NAMpy is not rewritten

This backend switch governs **dune-bayes's own (new) code**, not the upstream
`NAMpy/` package. NAMpy **stays on TensorFlow / Keras / TFP and is not ported** — it
remains the deterministic baseline and the source of reusable machinery. dune-bayes
is its **own PyTorch package** that:

- **Reuses NAMpy's TF-free Python directly** — the formula parser
  (`nampy/formulas/`) and the `ShapeFunctionRegistry` pattern carry no TF
  dependency, so they can be imported or adapted as-is.
- **Reimplements the TF/TFP-coupled pieces in PyTorch** rather than importing them,
  because they cannot cross the backend boundary: families (`tfd.*` →
  `torch.distributions.*`), the `DataModule` (`tf.data` → `torch.utils.data`
  `DataLoader`/`TensorDataset`, with **numpy/sklearn** fit-transform preprocessing
  replacing the Keras `Normalization`/`StringLookup`/`Discretization`/PLE/spline
  layers), and the shape functions (Keras `Layer`/`Model` → `nn.Module`). The
  verified `spikes/VariationalDense` is the seed of the new Bayesian atom.
- **Data stack decision:** the dune-bayes runtime is **pure PyTorch** — no
  TensorFlow import anywhere in dune-bayes. Batching via `torch.utils.data`;
  preprocessing via numpy/sklearn.

Where a deterministic NAMpy baseline is needed for WAIC/LOO comparison (CONTEXT),
it is reached via the TF NAMpy package as an external comparator, not by porting it
into the PyTorch tree.

## Consequences

**Positive**

- Leaves a deprecating substrate (pinned TF 2.15 / Keras 2 / TFP 0.23, Python
  ≤3.11) for the dominant ecosystem, lifting the Python/version cap and widening
  the contributor/user pool.
- `torch.distributions` covers families + `MixtureSameFamily` directly; the
  in-house atom ports trivially; **ADR-0004's serialization risk disappears**
  (`state_dict` instead of the `.keras`/H5 minefield).
- Keeps a clean seam to the JAX/NumPyro numerical future and the ADR-0001 MCMC
  backend — the staged inference plan gets *easier*, not harder.
- arviz, the comparison spine, is native here.

**Negative / accepted trade-offs**

- **Reimplementation cost:** the TF/TFP-coupled machinery dune-bayes wanted to
  reuse from NAMpy (families, `DataModule`, shape functions) cannot cross the
  backend boundary and must be **reimplemented** in PyTorch rather than imported
  (see *Scope*). Only NAMpy's TF-free Python (formula parser, registry) is reused
  directly. NAMpy itself is **not** ported — it stays on TF as the baseline.
- **Lost Keras magic:** `compile`/`fit`/`add_loss` auto-propagation must be
  re-provided via a trainer. Mitigated — the KL module-walk already exists in the
  spike, and Lightning restores a `fit()`-shaped surface.
- **Spikes are now historical.** The two TF verification spikes (KL propagation,
  serialization) no longer pin the production stack and must be **re-verified on
  PyTorch**. Cheaper than the originals: serialization is `state_dict`, and KL
  collection is explicit rather than framework magic — but the re-verification is
  owed before ADR-0004's claims can be called "spike-verified" again.
- **Two backends on the horizon** (torch now, JAX later) = a seam to maintain and
  a divergence risk. Mitigated by keeping inference/distribution behind a narrow
  interface (already mandated by ADR-0001).

## Relationship to other ADRs

- **ADR-0001 (mean-field VI):** statistical decision unchanged. The "tfp
  variational Dense layers" wording is retargeted to the in-house PyTorch
  `VariationalDense`; the reserved MCMC seam is now concretely **NumPyro/BlackJAX
  on JAX**.
- **ADR-0002 (priors as per-feature smoothness):** fully backend-agnostic — no
  change. The serializable-float / hierarchical-scale prior config is reused
  verbatim.
- **ADR-0003 (predictive / plots / comparison):** statistical contract unchanged.
  `tfd.MixtureSameFamily` → `torch.distributions.MixtureSameFamily`;
  `compile/fit`/`add_loss` → thin trainer + explicit KL; the "TFP save/load
  finicky" flagged risk is **retired** by `state_dict`. arviz unchanged.
- **ADR-0004 (in-house `VariationalDense`):** the core decision **survives** — an
  owned variational atom is still the right call, and for the same reasons. Only
  the host framework (Keras layer → `nn.Module`) and the serialization mechanism
  (`.keras`/SavedModel → `state_dict`) change. Its TF-stack verification is now
  historical (see above).
- **ADR-0005 (categoricals / interactions):** fully backend-agnostic — no change.

## Alternatives considered

- **Stay on TensorFlow / TFP.** Zero migration; the spikes are already green. But
  it builds all new code on a deprecating substrate whose load-bearing piece (TFP)
  has the least future, and the migration bill only grows after `BayesianNAMLSS`
  is built. Rejected.
- **Keras 3 (multi-backend) on a JAX/PyTorch backend.** Preserves the Keras
  functional + `compile/fit` UX, but **does not carry TFP forward** — i.e. it does
  not solve the actual load-bearing dependency — and keeps us behind Keras's
  abstraction over a more direct stack. Insufficient; rejected.
- **Jump straight to JAX/NumPyro now.** Best inference substrate and our stated
  future, but a steeper curve and a thinner just-works tabular/plotting/save-load
  ecosystem mean more assembly while the package is still finding its shape —
  higher risk for a small team pre-v1. Deferred to the numerical-future phase
  behind the ADR-0001 seam, not chosen as the v1 backend.
- **PyTorch + Pyro for VI.** Pyro offers richer VI/MCMC, but ADR-0004 already owns
  the mean-field atom and v1 does not need Pyro's machinery; `torch.distributions`
  suffices. Pyro is kept as an option, not a v1 dependency.
</content>
</invoke>
