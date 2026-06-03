# Issue 0023: Numeric preprocessing — train-fit scaling with inverse for plot axes

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006).

## What to build

sklearn-style numeric preprocessing on the DataModule: per-feature scaling statistics are fit on training data only (standardize by default; min-max opt-in per feature), `transform` applies the fitted statistics to new data without refitting, and `inverse_transform` recovers the original feature scale so effect ribbons and plot grids render in the units the modeller actually thinks in (user story 1 — readable per-feature effect curves).

## Acceptance criteria

- [ ] Standardize default: transformed training columns have mean ≈ 0 / sd ≈ 1 (tolerance explicit and commented — float error, not MC noise)
- [ ] Min-max scaling selectable per feature
- [ ] `transform` on held-out data reuses train statistics (no refit) — values match an independent hand-computed/sklearn reference
- [ ] `inverse_transform(transform(x))` recovers x within float tolerance
- [ ] End-to-end: an effect-ribbon plot grid can be expressed on the original feature scale via the DataModule's inverse
- [ ] Boundary tests only (values/shapes at the public interface)

## Blocked by

- Issue 0022 (#49, DataModule walking skeleton)
