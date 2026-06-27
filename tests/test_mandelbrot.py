"""Correctness and scaling-harness tests for the Mandelbrot kernels."""

from __future__ import annotations

import numpy as np
import pytest

from hpc.mandelbrot import (
    MandelbrotView,
    NUMBA_AVAILABLE,
    mandelbrot_naive,
    mandelbrot_numpy,
    mandelbrot_parallel,
)
from hpc.profiling import profile_naive
from hpc.scaling import strong_scaling, time_once, warmup
from hpc.mandelbrot import mandelbrot_parallel as _mp  # noqa: F401 (clarity)


# A small view that is fast enough for the naive reference to run in the suite,
# yet rich enough to contain both fast-escaping and never-escaping pixels.
SMALL = MandelbrotView(width=64, height=48, max_iter=120)
# Deliberately awkward: prime-ish height and worker counts that do not divide it.
AWKWARD = MandelbrotView(width=53, height=41, max_iter=90)


def test_numpy_matches_naive_exactly():
    """The vectorized kernel must reproduce the reference array exactly.

    Iteration counts are integers, so this is an exact equality, not a tolerance.
    """
    ref = mandelbrot_naive(SMALL)
    got = mandelbrot_numpy(SMALL)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("workers", [1, 2, 3, 4, 5])
def test_parallel_matches_naive(workers):
    """The multiprocessing kernel must match the reference for any worker count,
    including counts that do not evenly divide the number of rows."""
    ref = mandelbrot_naive(AWKWARD)
    got = mandelbrot_parallel(AWKWARD, workers=workers)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_parallel_reassembles_rows_in_order():
    """Row bands must be stitched back in the original top-to-bottom order."""
    ref = mandelbrot_numpy(SMALL)
    got = mandelbrot_parallel(SMALL, workers=4)
    # If bands were reordered, individual rows would still be valid Mandelbrot
    # rows but in the wrong place — full-array equality catches that.
    assert np.array_equal(got, ref)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_matches_naive():
    from hpc.mandelbrot import mandelbrot_numba

    ref = mandelbrot_naive(SMALL)
    got = mandelbrot_numba(SMALL)
    assert np.array_equal(got, ref)


def test_profiler_identifies_hotspot():
    """cProfile on the naive kernel must point at the naive escape-loop function."""
    view = MandelbrotView(width=40, height=40, max_iter=80)
    table, hotspot = profile_naive(view, top=5)
    assert "mandelbrot_naive" in hotspot
    assert "mandelbrot_naive" in table


def test_scaling_harness_returns_consistent_points():
    """The strong-scaling harness must return one point per worker count, with a
    speedup of exactly 1.0 at the baseline and positive timings throughout."""
    view = MandelbrotView(width=200, height=200, max_iter=200)
    workers = [1, 2]
    points = strong_scaling(view, workers, repeats=2, warm=False)

    assert [p.workers for p in points] == workers
    assert all(p.seconds > 0 for p in points)
    # Baseline speedup is 1.0 by construction.
    assert points[0].speedup == pytest.approx(1.0)
    # Efficiency = speedup / workers, always positive.
    assert all(p.efficiency > 0 for p in points)


def test_speedup_exceeds_one_on_large_instance():
    """On a large-enough instance, 2 workers should beat 1 worker.

    Timing is inherently noisy, so we (a) warm up the spawn pools first to
    exclude one-time process-creation cost, and (b) use best-of-several timing,
    which is robust against transient load because noise can only make a run
    *slower*. We then require only a real, possibly sub-linear, speedup ( > 1 ).
    """
    view = MandelbrotView(width=800, height=800, max_iter=500)
    warmup(view, [1, 2])  # exclude cold spawn-pool start from the measurement
    t1 = time_once(view, workers=1, repeats=3)
    t2 = time_once(view, workers=2, repeats=3)
    speedup = t1 / t2
    assert speedup > 1.0, f"expected speedup > 1, got {speedup:.2f} (t1={t1:.3f}, t2={t2:.3f})"
