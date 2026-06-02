# Issue 0008: Response-level predictive bands

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0005).

## What to build

Response-level plots (`plot_dist` / predicted-vs-actual) showing the full predictive band = epistemic + aleatoric — the proper prediction interval — built from the `LogLikSampler` `MixtureSameFamily` predictive. Default band 90%. Aleatoric is treated as a response-level property here, deliberately not attributed to individual feature curves.

## Acceptance criteria

- [ ] Response-level plot renders the full predictive band (epistemic + aleatoric), default 90%, configurable
- [ ] Band is derived from the `MixtureSameFamily` posterior predictive
- [ ] Distinct from the effect-ribbon view (issue 0006), which is epistemic-only

## Blocked by

- Issue 0007 (LogLikSampler + MixtureSameFamily predictive)
