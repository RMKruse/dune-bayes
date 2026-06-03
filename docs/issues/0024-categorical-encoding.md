# Issue 0024: Categorical encoding feeding random effects

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006). Data-side counterpart to the BayesianEmbedding random-effect slice (user stories 37–40).

## What to build

Categorical columns on the DataModule are integer-coded into the long-dtype tensors BayesianEmbedding consumes, with a deterministic, serialization-friendly category ↔ index mapping. The number of levels per categorical feature is exposed so the embedding (and its partial-pooling prior) can be sized from the data. Unseen levels at `transform` time follow a defined policy: reject with a clear error by default (a silent new index would fake confidence on a level the posterior never saw).

## Acceptance criteria

- [ ] A categorical column transforms to `torch.long` codes `0..K-1`; the mapping is deterministic and stable across repeated calls
- [ ] Levels-per-feature is exposed and matches what BayesianEmbedding needs for sizing
- [ ] An unseen level at `transform` time raises a clear error (default policy), covered by a test
- [ ] End-to-end: a DataModule with a categorical feature feeds a model containing a BayesianEmbedding term; per-level effect samples are extractable
- [ ] Boundary tests only (codes, shapes, level counts — no internals)

## Blocked by

- Issue 0022 (#49, DataModule walking skeleton)
