"""
Benchmark + strong-scaling study for the Mandelbrot workload.

    python benchmark.py

What it does:
  1. Profiles the naive kernel (cProfile) and prints the hotspot.
  2. Sanity-times naive vs. vectorized vs. parallel on a moderate view so the
     three implementations can be compared on the same problem.
  3. Runs a strong-scaling study with the parallel kernel at 1, 2, 4, 8, ...
     workers, printing a real speedup / efficiency table.
  4. Saves figures/speedup.png and figures/efficiency.png.

Only real measured timings are printed — nothing is hard-coded.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hpc.mandelbrot import (
    DEFAULT_VIEW,
    MandelbrotView,
    NUMBA_AVAILABLE,
    mandelbrot_naive,
    mandelbrot_numpy,
    mandelbrot_parallel,
)
from hpc.profiling import profile_naive
from hpc.scaling import ScalingPoint, format_table, strong_scaling

FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def worker_ladder(max_cores: int) -> list[int]:
    """1, 2, 4, 8, ... up to the core count, with the exact core count appended."""
    ladder = []
    w = 1
    while w <= max_cores:
        ladder.append(w)
        w *= 2
    if ladder[-1] != max_cores:
        ladder.append(max_cores)
    return ladder


def time_call(fn, *args) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - start, result


def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)
    cores = os.cpu_count() or 1
    print(f"Machine: {cores} logical CPUs\n")

    # --- 1. Profiling --------------------------------------------------------
    print("=" * 70)
    print("PROFILING the naive kernel (small view, so it finishes quickly)")
    print("=" * 70)
    prof_view = MandelbrotView(width=200, height=200, max_iter=200)
    table, hotspot = profile_naive(prof_view, top=6)
    print(table)
    print(f">>> Hotspot (most self-time): {hotspot}\n")

    # --- 2. Three implementations head to head -------------------------------
    print("=" * 70)
    print("IMPLEMENTATION COMPARISON (same view for all three)")
    print("=" * 70)
    cmp_view = MandelbrotView(width=400, height=400, max_iter=400)
    print(f"View: {cmp_view.width}x{cmp_view.height}, max_iter={cmp_view.max_iter}\n")

    t_naive, _ = time_call(mandelbrot_naive, cmp_view)
    print(f"  naive  (pure Python)   {t_naive:8.3f}s   1.00x")
    t_numpy, _ = time_call(mandelbrot_numpy, cmp_view)
    print(f"  numpy  (vectorized)    {t_numpy:8.3f}s   {t_naive / t_numpy:6.1f}x vs naive")
    t_par, _ = time_call(mandelbrot_parallel, cmp_view, cores)
    print(f"  parallel ({cores} cores)   {t_par:8.3f}s   {t_naive / t_par:6.1f}x vs naive")
    if NUMBA_AVAILABLE:
        from hpc.mandelbrot import mandelbrot_numba

        mandelbrot_numba(MandelbrotView(width=8, height=8, max_iter=8))  # warm JIT
        t_numba, _ = time_call(mandelbrot_numba, cmp_view)
        print(f"  numba  (jit, threads)  {t_numba:8.3f}s   {t_naive / t_numba:6.1f}x vs naive")
    print()

    # --- 3. Strong-scaling study ---------------------------------------------
    print("=" * 70)
    print("STRONG-SCALING STUDY (fixed problem size, more workers)")
    print("=" * 70)
    view = DEFAULT_VIEW
    ladder = worker_ladder(cores)
    print(
        f"View: {view.width}x{view.height}, max_iter={view.max_iter}; "
        f"workers = {ladder}; best of 3 each\n"
    )
    points = strong_scaling(view, ladder, repeats=3)
    print(format_table(points))
    print()

    _save_figures(points, cores)


def _save_figures(points: list[ScalingPoint], cores: int) -> None:
    ws = [p.workers for p in points]
    speedups = [p.speedup for p in points]
    effs = [p.efficiency * 100 for p in points]

    # Speedup vs cores, against the ideal y = x line.
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    ax.plot(ws, speedups, "o-", color="#9333ea", label="measured", linewidth=2)
    ax.plot(ws, ws, "--", color="#999", label="ideal (linear)")
    ax.set_xlabel("worker processes")
    ax.set_ylabel("speedup  T(1) / T(p)")
    ax.set_title("Strong scaling — Mandelbrot render")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(FIGURES, "speedup.png")
    fig.savefig(out1)
    print(f"Saved {out1}")

    # Parallel efficiency.
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    ax.plot(ws, effs, "o-", color="#9333ea", linewidth=2)
    ax.axhline(100, linestyle="--", color="#999", label="ideal (100%)")
    ax.set_xlabel("worker processes")
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Parallel efficiency — Mandelbrot render")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(FIGURES, "efficiency.png")
    fig.savefig(out2)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
