"""
Mandelbrot escape-time kernels.

The Mandelbrot set is the set of complex numbers ``c`` for which the iteration

    z_{n+1} = z_n^2 + c,   z_0 = 0

stays bounded. In practice we iterate up to ``max_iter`` times and record, for
each pixel ``c``, how many steps it took for ``|z| > 2`` (the escape radius). A
pixel that never escapes is given ``max_iter``. That per-pixel iteration count is
the image.

This is an *embarrassingly parallel* workload: every pixel is independent, there
is no communication between them, and the per-pixel cost varies wildly (points
inside the set run the full ``max_iter`` loop). That makes it a clean subject for
a strong-scaling study.

All public kernels here take the same arguments and return the same
``int32`` array of shape ``(height, width)``, so results are directly
comparable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Optional numba acceleration.
#
# We only *detect* numba at import time (cheap) and defer the heavy
# numba/llvmlite import until the JIT kernel is actually called. This matters
# for the multiprocessing path: every spawned worker re-imports this module, and
# eagerly importing numba in each child would add seconds of startup that swamp
# the parallel speedup. Detection keeps NUMBA_AVAILABLE accurate for callers.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

NUMBA_AVAILABLE = _importlib_util.find_spec("numba") is not None
_numba_kernel = None  # compiled lazily on first use


@dataclass(frozen=True)
class MandelbrotView:
    """A rectangular window onto the complex plane, sampled on a pixel grid."""

    xmin: float = -2.5
    xmax: float = 1.0
    ymin: float = -1.25
    ymax: float = 1.25
    width: int = 800
    height: int = 600
    max_iter: int = 256

    def real_axis(self) -> np.ndarray:
        return np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)

    def imag_axis(self) -> np.ndarray:
        return np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)


# A view that is a bit zoomed-in on the seahorse-valley boundary, where many
# pixels run the full iteration count — i.e. genuinely expensive work.
DEFAULT_VIEW = MandelbrotView(
    xmin=-0.74877,
    xmax=-0.74872,
    ymin=0.06505,
    ymax=0.06510,
    width=800,
    height=800,
    max_iter=1000,
)


# ---------------------------------------------------------------------------
# 1. Naive pure-Python reference
# ---------------------------------------------------------------------------
def mandelbrot_naive(view: MandelbrotView) -> np.ndarray:
    """Reference implementation: an explicit Python loop over every pixel.

    Correct and readable, but slow — this is the baseline every other kernel is
    measured against and verified against.
    """
    xs = view.real_axis()
    ys = view.imag_axis()
    out = np.empty((view.height, view.width), dtype=np.int32)
    max_iter = view.max_iter

    for j in range(view.height):
        cy = ys[j]
        row = out[j]
        for i in range(view.width):
            cx = xs[i]
            zr = 0.0
            zi = 0.0
            count = 0
            # Iterate z = z^2 + c until escape (|z|^2 > 4) or max_iter.
            while count < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + cy
                zr = zr2 - zi2 + cx
                count += 1
            row[i] = count
    return out


# ---------------------------------------------------------------------------
# 2. Vectorized NumPy
# ---------------------------------------------------------------------------
def mandelbrot_numpy(view: MandelbrotView) -> np.ndarray:
    """Vectorized kernel: iterate the *entire* grid at once with NumPy.

    A boolean mask tracks which pixels are still iterating; once a pixel escapes
    it is frozen. One pass of arithmetic touches all pixels, so the Python
    interpreter overhead is paid once per iteration instead of once per pixel.
    """
    return _mandelbrot_numpy_block(
        view.real_axis(), view.imag_axis(), view.max_iter
    )


def _mandelbrot_numpy_block(
    xs: np.ndarray, ys: np.ndarray, max_iter: int
) -> np.ndarray:
    """Vectorized escape-time over the grid defined by axes ``xs`` x ``ys``.

    Pulled out as a free function so the multiprocessing workers can call it on
    a horizontal slab of rows without depending on a ``MandelbrotView``.
    """
    c = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    zr = np.zeros(c.shape, dtype=np.float64)
    zi = np.zeros(c.shape, dtype=np.float64)
    out = np.zeros(c.shape, dtype=np.int32)
    cr = c.real
    ci = c.imag

    # `alive` marks pixels that have not yet escaped. The naive reference
    # increments `count` once per loop pass that begins without having escaped,
    # so we mirror that exactly: test escape, then add 1 to survivors, then step.
    for _ in range(max_iter):
        mag2 = zr * zr + zi * zi
        alive = mag2 <= 4.0
        if not alive.any():
            break
        out[alive] += 1
        # z = z^2 + c, computed in-place on the real/imag parts.
        new_zr = zr * zr - zi * zi + cr
        new_zi = 2.0 * zr * zi + ci
        # Only advance live pixels; escaped ones are frozen (value is ignored
        # once `alive` is False, but freezing avoids overflow warnings).
        zr = np.where(alive, new_zr, zr)
        zi = np.where(alive, new_zi, zi)
    return out


# ---------------------------------------------------------------------------
# 3. Multiprocessing across cores
# ---------------------------------------------------------------------------
def _worker_rows(args):
    """Compute a horizontal band of rows. Module-level so it is picklable."""
    xs, ys_block, max_iter = args
    return _mandelbrot_numpy_block(xs, ys_block, max_iter)


def mandelbrot_parallel(view: MandelbrotView, workers: int | None = None) -> np.ndarray:
    """Parallel kernel: split the image into row-bands, one task per band, and
    run the vectorized kernel on each band in a separate process.

    Rows are independent, so there is no inter-process communication beyond
    shipping each band's result back — a textbook embarrassingly parallel
    decomposition. ``workers`` defaults to ``os.cpu_count()``.
    """
    import multiprocessing as mp

    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, int(workers))

    xs = view.real_axis()
    ys = view.imag_axis()

    if workers == 1:
        # Avoid process-pool overhead when there is nothing to parallelise.
        return _mandelbrot_numpy_block(xs, ys, view.max_iter)

    # Chop the rows into `workers` contiguous bands (last band absorbs remainder).
    bands = np.array_split(np.arange(view.height), workers)
    tasks = [(xs, ys[band], view.max_iter) for band in bands if len(band) > 0]

    ctx = mp.get_context("spawn")  # spawn is the only portable start method on Windows
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_worker_rows, tasks)

    return np.vstack(results)


# ---------------------------------------------------------------------------
# 4. Optional numba JIT (compiled on first call so import stays cheap)
# ---------------------------------------------------------------------------
def _get_numba_kernel():  # pragma: no cover - only when numba present
    """Compile and cache the numba kernel on first use."""
    global _numba_kernel
    if _numba_kernel is None:
        from numba import njit, prange

        @njit(cache=True, fastmath=True, parallel=True)
        def kernel(xs, ys, max_iter):
            h = ys.shape[0]
            w = xs.shape[0]
            out = np.empty((h, w), dtype=np.int32)
            for j in prange(h):
                cy = ys[j]
                for i in range(w):
                    cx = xs[i]
                    zr = 0.0
                    zi = 0.0
                    count = 0
                    while count < max_iter:
                        zr2 = zr * zr
                        zi2 = zi * zi
                        if zr2 + zi2 > 4.0:
                            break
                        zi = 2.0 * zr * zi + cy
                        zr = zr2 - zi2 + cx
                        count += 1
                    out[j, i] = count
            return out

        _numba_kernel = kernel
    return _numba_kernel


def mandelbrot_numba(view: MandelbrotView) -> np.ndarray:  # pragma: no cover
    """JIT-compiled, thread-parallel kernel (requires numba to be installed)."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not installed")
    kernel = _get_numba_kernel()
    return kernel(view.real_axis(), view.imag_axis(), view.max_iter)
