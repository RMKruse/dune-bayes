# Issue 0026: Minibatching via torch.utils.data — KL/N stays full-data N

> Source: See PRD `docs/prd/0001-bayesian-feature-networks.md` for the full design, user stories, and governing ADRs (0001–0006). The package layout (CLAUDE.md) promises `torch.utils.data` in the data component; the current fit loop is full-batch only.

## What to build

Optional batched training: the DataModule provides `torch.utils.data`-backed iteration over (feature-dict, target) batches with seedable shuffling, and the fit surface consumes it when a batch size is requested. Full-batch remains the default — no behavior change when batching is not requested.

The load-bearing numerical point: **the KL divisor stays N = full training-set size, never the batch size.** In the minibatch ELBO the expected NLL is estimated per batch but the KL term appears once over the whole dataset; dividing by batch size would silently over-regularize by N/batch_size. This must be asserted by a test, not just documented.

## Acceptance criteria

- [ ] Batched iteration yields (feature-dict, target) batches with correct shapes; a final partial batch is handled
- [ ] Shuffling is seedable — same seed, same batch order within one run (consistent with the one-model-object reproducibility rule)
- [ ] The KL divisor under minibatching equals full-data N — asserted against the full-batch path / a hand-computed reference on a toy model
- [ ] End-to-end: fitting with a batch size on toy data runs to completion with finite, decreasing loss and per-epoch history
- [ ] Full-batch default is unchanged when no batch size is given (existing tests stay green)

## Blocked by

- Issue 0022 (#49, DataModule walking skeleton)
