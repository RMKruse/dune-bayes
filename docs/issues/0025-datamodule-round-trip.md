# Issue 0025: DataModule state round-trip

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006). Extends the model save/load slice (issue 0015, user story 44) to the data side.

## What to build

Fitted preprocessing state — scaler statistics, category ↔ index maps, `n_obs` — serializes to a closure-free config + state and loads back, so a reloaded DataModule transforms fresh raw data identically to the original. Together with the existing model round-trip, the persisted (model, DataModule) pair reproduces the full pipeline on new raw data without refitting anything.

## Acceptance criteria

- [ ] save → load → `transform` equals the original `transform` with `max|Δ| == 0` (exact equality — same floats, per the round-trip archetype)
- [ ] Category maps and level counts survive the round-trip exactly; an unseen level still errors identically after reload
- [ ] `n_obs` survives the round-trip
- [ ] End-to-end: a reloaded (model, DataModule) pair takes fresh raw data through transform → posterior predictive without refit; transform outputs are bit-identical to pre-save
- [ ] Serialized state is closure-free (config of floats/strings/ints, plus plain tensors/arrays)

## Blocked by

- Issue 0023 (#50, Numeric preprocessing)
- Issue 0024 (#51, Categorical encoding)
