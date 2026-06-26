# Deferred Nice-To-Haves Register

This register supports PRD #128 and the paper-publication readiness track in
`docs/prd/0003-paper-publication-readiness.md`. It records follow-up ideas that
are useful to keep visible, but are not required for the first paper artifact
unless a reviewer or correctness issue changes their status.

The register is a scope-control document, not a publication-blocking checklist.
It references the existing out-of-scope decisions without reopening them:

- PRD 0003 keeps new inference engines, a JAX runtime port, first-party Laplace,
  post-hoc conformal calibration or band inflation, new family tiers, new
  shape-function architectures, formula syntax extensions, RMSE chasing, full
  expensive CI runs, and package-index publishing outside the first paper scope.
- PRD 0002 keeps hierarchical priors, first-party Laplace, and deferred Tier C
  families such as zero-inflated, hurdle, GEV, and skew-t outside the paper
  hardening scope unless a defined experiment needs them.
- ADR-0001 and ADR-0006 reserve MCMC and JAX/NumPyro work as future backend
  directions rather than first-paper package requirements.

Deferred items may be promoted only when they address a correctness bug,
reproducibility failure, or reviewer-blocking evidence gap. Otherwise, keep them
outside the paper artifact freeze and convert them into future issues after the
paper artifact is frozen.

| Theme | Deferred item | Why it is deferred | Future issue seed |
| --- | --- | --- | --- |
| Ablation | prior-tier ablations | Useful for a broader priors paper, but not required to support the first paper's uncertainty-disentanglement claim. | Create an ablation issue naming the prior tiers, datasets, metrics, expected runtime, and the claim it would strengthen. |
| Ablation | posterior-draw sensitivity | Helpful for robustness reporting, but the first artifact already validates promoted evidence through fixed configs, seeds, and manifests. | Create a sensitivity issue with draw counts, acceptance thresholds, and affected figures or tables. |
| Performance | runtime scaling | Important for adoption, but not a correctness or reviewer-readiness blocker unless current publication runs become infeasible. | Create a benchmark issue with dataset sizes, hardware assumptions, wall-clock metrics, and a target comparison. |
| Baseline | richer baselines | More comparators can deepen the story, but the first paper is blocked only by the declared comparator claims in PRD #128. | Create one issue per baseline with install prerequisites, fixture/scoring route, and an explicit inclusion or exclusion rule. |
| Documentation | tutorial notebooks | Valuable reader material after submission, but the first artifact only requires the reviewer-facing docs and reproducible paper path. | Create a notebook issue with the target workflow, expected outputs, and whether it must run in CI smoke mode. |
| Documentation | documentation site | Better long-term discoverability, but not required for a citable paper artifact or DOI deposit. | Create a docs-site issue with hosting choice, source pages, build command, and release ownership. |
| Family Tier | richer variational families | A meaningful methods extension, but PRD #128 does not reopen the mean-field VI choice for the first paper. | Create a methods issue describing the variational family, numerical stability gates, comparison target, and claims it may enable. |
| Inference | validation or shipped MCMC expansion | Additional NUTS/HMC work is validation-only for the paper and does not ship as a package backend. | Create an inference issue that states whether the work is validation, diagnostics, or a package API proposal. |
| Runtime Backend | JAX-backed performance | ADR-0006 reserves JAX/NumPyro as a future backend direction, not a first-paper runtime dependency. | Create a backend issue with the API seam, required parity tests, performance target, and migration risk. |

When creating a future issue from this register, keep the issue independently
grabbable: name the theme, scope the acceptance criteria to one behavior or
experiment, state whether it changes paper claims, and link back to this
register plus PRD #128. Do not attach these deferred items to the publication
blocker list unless their status changes through reviewer feedback or a
correctness finding.
