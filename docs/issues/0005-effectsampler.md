# Issue 0005: EffectSampler workhorse

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The goal-1 sampling workhorse: `EffectSampler(model, data, T)` draws T posterior weight samples and returns per-feature contribution samples only — the input to centered epistemic effect ribbons and interaction surfaces. Pure function of (model, data, T); default `T = T_predict = 200`, overridable. No log-likelihood or IC concerns (kept separate from `LogLikSampler`, issue 0007).

## Acceptance criteria

- [ ] `EffectSampler(model, data, T)` returns `{feature_name: contribution_samples[T, n, param_count]}`
- [ ] Default `T_predict = 200`, explicitly overridable
- [ ] Posterior-mean of contributions is T-stable (mean converges, CI tightens with T) on a fixed toy posterior
- [ ] Centering posterior-sampled curves yields zero-mean curves over the data
- [ ] Pure function — no mutation of model state

## Blocked by

- Issue 0003 (BayesianNAMLSS skeleton)
