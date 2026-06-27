"""
hpc — a Mandelbrot-set escape-time renderer used as a high-performance-computing
case study.

The same workload (count escape iterations for every pixel of a complex-plane
grid) is implemented three ways so they can be compared head to head:

    naive        pure-Python double loop over pixels      (reference)
    vectorized   one NumPy pass over the whole grid        (SIMD-style)
    parallel     multiprocessing across CPU cores          (row-block chunks)

Every implementation returns the *same* integer iteration-count array (verified
in the test suite), so any speedup is a genuine win, not a change of answer.

Optionally, if `numba` is importable, a JIT-compiled variant is exposed too.
"""

from .mandelbrot import (
    DEFAULT_VIEW,
    MandelbrotView,
    mandelbrot_naive,
    mandelbrot_numba,
    mandelbrot_numpy,
    mandelbrot_parallel,
    NUMBA_AVAILABLE,
)

__all__ = [
    "DEFAULT_VIEW",
    "MandelbrotView",
    "mandelbrot_naive",
    "mandelbrot_numpy",
    "mandelbrot_parallel",
    "mandelbrot_numba",
    "NUMBA_AVAILABLE",
]
