"""Benchmark spike — wall-clock profile of fit and the T-draw sweeps at scale.

Unlike the two verification spikes, this one measures rather than proves: it
replaces the getting-started toy scale (hundreds of rows) with realistic
tabular scale and reports where the time actually goes, to decide whether the
posterior sweeps are worth vectorizing *inside torch* before reaching for a
backend change (ADR-0006 keeps JAX as the staged numerical future behind the
inference seam — this spike informs that staging, it does not reopen it).
The follow-up slice is docs/issues/0027-vectorized-posterior-sweep.md.

Stages timed
  1. fit               full-batch ELBO training → sec/epoch
  2. forward           one stochastic predict_params pass — the unit every
                       T-loop repeats
  3. draw_predictive   the T_eval Python-loop sweep + mixture assembly
                       (log_lik_sampler.py — `for _ in range(T)`)
  4. tiled proxy       ONE forward pass on the input tiled S× along the batch
                       dim: per-draw cost if the sweep were perfectly batched.
                       Weights are shared across tiles, so this is a *lower
                       bound* — a real vectorized sweep also draws S
                       independent weight samples per layer.
  5. pointwise_log_lik float64 scoring loop over the T draws (WAIC/LOO input)
  6. sample_effects    per-feature T_predict sweep on a plotting grid

The headline number is the headroom ratio: loop per-draw (3) over the batched
lower bound (4). A ratio >> 1 means the sweep is dispatch-bound, not
FLOP-bound, and batching the draws (sample-dim reparameterization noise or
torch.func.vmap) buys real wall-clock with no backend change.

First measurement (2026-06-06, torch 2.12, 5-thread CPU, 20 features): the
ratio is scale-dependent — ~5.9x at n=1e3, 2.0x at 5e3, 1.0x by n=25e3. So
the loop costs real time in the small-batch regime (grid-based effect/ribbon
sweeps, small-data fits — ADR-0001's home turf) and is already at the compute
bound for big-n WAIC, where threads/device are the lever instead. Issue 0027
is scoped accordingly.

Run with the project venv (package installed editable):

    .venv-torch/bin/python spikes/spike_t_sweep_benchmark.py
    # closer to a real application (several minutes, ~10 GB peak):
    .venv-torch/bin/python spikes/spike_t_sweep_benchmark.py --n 500000 --t-eval 1000
    # quick smoke check that the spike itself runs:
    .venv-torch/bin/python spikes/spike_t_sweep_benchmark.py \
        --n 2000 --features 4 --epochs 2 --t-eval 20 --t-predict 20 --grid 64

Memory note: draw_predictive materializes (T, n, param_count) float32 plus
(n, T) mixture logits plus a (T, n) float64 log-lik — at n=500k / T=1000 that
is ~10 GB resident. That cliff is itself a finding for issue 0027: a
vectorized sweep must chunk over T (hidden activations at (T, n, 64) would be
far worse if it didn't).

CPU only by design: CPU is the package default (CLAUDE.md device rule) and
the question under test is Python-dispatch overhead, which CUDA would only
mask behind kernel-launch latency. CUDA timing is out of scope here.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from neural_bamlss.families import NormalFamily
from neural_bamlss.model import BayesianNAMLSS
from neural_bamlss.sampling.effect_sampler import sample_effects
from neural_bamlss.sampling.log_lik_sampler import draw_predictive, pointwise_log_lik
from neural_bamlss.shapes import BayesianMLP
from neural_bamlss.utils import eval_mode, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=50_000, help="training rows")
    p.add_argument("--features", type=int, default=20, help="numeric features")
    p.add_argument("--epochs", type=int, default=3, help="full-batch fit epochs")
    p.add_argument("--t-eval", type=int, default=1000, help="T for the log-lik sweep")
    p.add_argument("--t-predict", type=int, default=200, help="T for sample_effects")
    p.add_argument("--grid", type=int, default=512, help="effect-plot grid points")
    p.add_argument(
        "--tile",
        type=int,
        default=20,
        help="max batch-tiling factor S for the vectorization lower bound",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_data(n: int, n_features: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Smooth additive ground truth — sin / quadratic / tanh cycled per feature."""
    X: dict[str, torch.Tensor] = {}
    contribs = []
    for i in range(n_features):
        x = torch.rand(n, 1) * 4.0 - 2.0
        X[f"x{i}"] = x
        if i % 3 == 0:
            contribs.append(torch.sin(2.0 * x))
        elif i % 3 == 1:
            contribs.append(0.5 * x**2 - 0.5)
        else:
            contribs.append(torch.tanh(x))
    f = torch.cat(contribs, dim=1).sum(dim=1)
    y = f + 0.3 * torch.randn(n)
    return X, y


def timed(fn, *args, **kwargs) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    print(
        f"[INFO] torch {torch.__version__} | "
        f"{torch.get_num_threads()} CPU threads | "
        f"n={args.n} features={args.features} "
        f"T_eval={args.t_eval} T_predict={args.t_predict}"
    )

    X, y = make_data(args.n, args.features)
    family = NormalFamily()
    formula = {
        name: BayesianMLP(
            in_features=1, param_count=family.param_count, kl_divisor=float(args.n)
        )
        for name in X
    }
    model = BayesianNAMLSS(formula, family=family, n_obs=args.n)

    # ── 1. fit ────────────────────────────────────────────────────────────────
    _, t_fit = timed(
        model.fit, X, y, epochs=args.epochs, warmup_epochs=min(1, args.epochs)
    )
    sec_per_epoch = t_fit / args.epochs
    print(f"[RESULT] fit: {t_fit:.2f}s total, {sec_per_epoch:.3f}s/epoch (full batch)")

    # ── 2. single stochastic forward — the unit the T-loops repeat ───────────
    with eval_mode(model), torch.no_grad():
        model.predict_params(X)  # warm-up (allocator, thread pool)
        t_fwd = min(timed(model.predict_params, X)[1] for _ in range(3))
    print(f"[RESULT] forward: {t_fwd * 1e3:.1f} ms per stochastic pass")

    # ── 3. draw_predictive — the Python-loop sweep ────────────────────────────
    draws, t_draw = timed(draw_predictive, model, X, T=args.t_eval)
    per_draw_loop = t_draw / args.t_eval
    print(
        f"[RESULT] draw_predictive(T={args.t_eval}): {t_draw:.2f}s total, "
        f"{per_draw_loop * 1e3:.1f} ms/draw"
    )

    # ── 4. tiled lower bound — same compute, one dispatch ─────────────────────
    # Cap S*n so the proxy itself stays cheap in memory.
    S = max(1, min(args.tile, 2_000_000 // args.n))
    X_tiled = {k: v.repeat(S, 1) for k, v in X.items()}
    with eval_mode(model), torch.no_grad():
        model.predict_params(X_tiled)  # warm-up
        _, t_tiled = timed(model.predict_params, X_tiled)
    per_draw_tiled = t_tiled / S
    print(
        f"[RESULT] tiled proxy (S={S}): {t_tiled:.2f}s for one pass on S*n rows, "
        f"{per_draw_tiled * 1e3:.1f} ms/draw lower bound"
    )

    # ── 5. pointwise_log_lik — float64 scoring of the drawn samples ──────────
    _, t_score = timed(pointwise_log_lik, model, draws.summed_samples, y)
    print(
        f"[RESULT] pointwise_log_lik(T={args.t_eval}): {t_score:.2f}s "
        f"({t_score / args.t_eval * 1e3:.1f} ms/draw, float64)"
    )

    # ── 6. sample_effects on a plotting grid ──────────────────────────────────
    grid = {name: torch.linspace(-2.0, 2.0, args.grid).unsqueeze(1) for name in X}
    _, t_eff = timed(sample_effects, model, grid, T=args.t_predict)
    print(
        f"[RESULT] sample_effects(T={args.t_predict}, grid={args.grid}): {t_eff:.2f}s"
    )

    # ── summary ───────────────────────────────────────────────────────────────
    headroom = per_draw_loop / per_draw_tiled if per_draw_tiled > 0 else float("inf")
    sweep_total = t_draw + t_score
    print()
    print(f"[SUMMARY] one fit epoch        : {sec_per_epoch:8.2f}s")
    print(
        f"[SUMMARY] full WAIC/LOO sweep  : {sweep_total:8.2f}s "
        f"(draw {t_draw:.2f}s + score {t_score:.2f}s)"
    )
    print(f"[SUMMARY] loop per-draw        : {per_draw_loop * 1e3:8.1f} ms")
    print(f"[SUMMARY] batched lower bound  : {per_draw_tiled * 1e3:8.1f} ms")
    print(
        f"[SUMMARY] vectorization headroom: {headroom:6.1f}x "
        f"(loop per-draw / batched lower bound)"
    )
    if headroom > 2.0:
        print(
            "[SUMMARY] sweep is dispatch-bound -> vectorizing the T-loop "
            "(issue 0027) buys real wall-clock before any backend change."
        )
    else:
        print(
            "[SUMMARY] sweep is already near the compute bound at this scale -> "
            "vectorization buys little here; re-measure at larger n/T."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
