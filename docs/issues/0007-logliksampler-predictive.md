# Issue 0007: LogLikSampler + MixtureSameFamily predictive

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

The goal-3 sampling workhorse, separate from `EffectSampler`: `LogLikSampler(model, data, T)` returns summed-predictor samples and pointwise log-likelihood samples, and assembles the `tfd.MixtureSameFamily` posterior predictive (across-component spread = epistemic, within-component = aleatoric). Backs `sample_posterior_predictive(data, T)`. Pure function; default `T = T_predict` for predictive plots, `T_eval = 1000` when called for IC. Splitting from `EffectSampler` keeps the cheap frequently-run effect path independent from the expensive log-likelihood path.

## Acceptance criteria

- [ ] `LogLikSampler(model, data, T)` returns `summed_samples[T, n, param_count]` and `pointwise_loglik[T, n]`
- [ ] Posterior predictive is a `tfd.MixtureSameFamily` over T weight-sampled family distributions
- [ ] On a known toy posterior the decomposition holds: across-component variance = injected epistemic, within = family aleatoric
- [ ] For a degenerate (single-draw) posterior, pointwise log-lik matches direct family `log_prob`
- [ ] `sample_posterior_predictive(data, T)` is exposed on `BayesianNAMLSS`

## Blocked by

- Issue 0003 (BayesianNAMLSS skeleton)
