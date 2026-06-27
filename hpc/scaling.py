"""
Strong-scaling harness.

A *strong-scaling* study fixes the problem size and increases the number of
workers, measuring how wall-clock time falls. From the timings we derive:

    speedup(p)    = T(1) / T(p)           ideal is p
    efficiency(p) = speedup(p) / p        ideal is 1.0 (100%)

The gap between actual and ideal speedup is exactly what Amdahl's law predicts:
the un-parallelisable fraction (here, process spawn, pickling result bands, and
the final ``vstack``) caps how far the curve can go.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .mandelbrot import MandelbrotView, mandelbrot_parallel


@dataclass
class ScalingPoint:
    workers: int
    seconds: float
    speedup: float
    efficiency: float


def time_once(view: MandelbrotView, workers: int, repeats: int = 3) -> float:
    """Return the best-of-``repeats`` wall time for the parallel kernel.

    Best-of (min) is used rather than mean because it is the most stable
    estimate of the achievable time: it discards runs perturbed by OS
    scheduling, GC, or background load, which can only ever make a run *slower*.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = mandelbrot_parallel(view, workers=workers)
        elapsed = time.perf_counter() - start
        # Touch the result so a clever runtime cannot optimise the work away.
        _ = int(result[0, 0])
        best = min(best, elapsed)
    return best


def warmup(view: MandelbrotView, worker_counts: list[int]) -> None:
    """Spin up each pool size once and throw the result away.

    Process spawn, module import in the children, and the first allocation are
    one-time costs that have nothing to do with steady-state scaling. Warming up
    before timing keeps the reported numbers honest (it measures the work, not
    Python's cold start).
    """
    for w in worker_counts:
        mandelbrot_parallel(view, workers=w)


def strong_scaling(
    view: MandelbrotView,
    worker_counts: list[int],
    repeats: int = 3,
    warm: bool = True,
) -> list[ScalingPoint]:
    """Run the kernel at each worker count and compute speedup / efficiency.

    The single-worker time is the serial baseline for the speedup ratios.
    Set ``warm=False`` to skip the warmup pass (used in fast unit tests).
    """
    if warm:
        warmup(view, worker_counts)

    timings: dict[int, float] = {}
    for w in worker_counts:
        timings[w] = time_once(view, w, repeats=repeats)

    baseline = timings[worker_counts[0]]
    points: list[ScalingPoint] = []
    for w in worker_counts:
        t = timings[w]
        speedup = baseline / t
        points.append(
            ScalingPoint(
                workers=w,
                seconds=t,
                speedup=speedup,
                efficiency=speedup / w,
            )
        )
    return points


def format_table(points: list[ScalingPoint]) -> str:
    """Render a scaling study as a fixed-width text table."""
    lines = [
        f"{'workers':>8} {'time (s)':>10} {'speedup':>9} {'efficiency':>11}",
        f"{'-'*8} {'-'*10} {'-'*9} {'-'*11}",
    ]
    for p in points:
        lines.append(
            f"{p.workers:>8d} {p.seconds:>10.3f} {p.speedup:>8.2f}x {p.efficiency*100:>10.1f}%"
        )
    return "\n".join(lines)
