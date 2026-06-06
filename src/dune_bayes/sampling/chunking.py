"""T-chunking for the vectorized posterior sweeps (issue 0027 / GitHub #80).

The sweeps batch their T posterior draws along a leading sample dimension
(sample-dimension reparameterization). Fully batching T = 1000 would
materialize hidden activations at (T, n, hidden) — far beyond the per-draw
loop's peak memory at large n — so each dispatch is capped at a row budget:
small-batch sweeps (plotting grids, small-data fits — the dispatch-bound
regime the vectorization targets) run all T draws in one dispatch, while
large-n WAIC sweeps fall back toward the loop's memory profile, where they
are FLOP-bound anyway.
"""

from __future__ import annotations

from collections.abc import Iterator

# Default chunk budget: max sample-slices × rows materialized per dispatch.
# 1M rows × hidden width 64 × 4 bytes ≈ 256 MB peak hidden activation — small-n
# grids batch all T draws in one dispatch, while at n = 500k the chunk drops to
# 2 and memory stays within a small factor of the per-draw loop's.
_CHUNK_BUDGET_ROWS: int = 1_000_000


def _chunk_sizes(T: int, n: int, chunk_size: int | None = None) -> Iterator[int]:
    """Yield per-dispatch sample counts summing to T.

    Args:
        T: Total number of posterior draws.
        n: Rows per draw (batch size of the underlying forward pass).
        chunk_size: Explicit draws-per-dispatch override. None derives it
            from _CHUNK_BUDGET_ROWS so memory scales with n, not with T.

    Yields:
        Chunk sizes in [1, T]; the last chunk carries the ragged remainder.
    """
    if chunk_size is None:
        chunk_size = max(1, _CHUNK_BUDGET_ROWS // max(n, 1))
    chunk_size = max(1, min(int(chunk_size), T))
    for start in range(0, T, chunk_size):
        yield min(chunk_size, T - start)
