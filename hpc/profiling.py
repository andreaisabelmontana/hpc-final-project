"""
cProfile wrapper that pins down the hotspot of the naive implementation.

The whole HPC discipline starts here: profile before you optimise. Running this
on the pure-Python kernel shows that essentially all the time is spent inside the
per-pixel escape loop (``mandelbrot_naive``), which is precisely the function the
vectorized and parallel kernels replace.
"""

from __future__ import annotations

import cProfile
import io
import pstats

from .mandelbrot import MandelbrotView, mandelbrot_naive


def profile_naive(view: MandelbrotView, top: int = 8) -> tuple[str, str]:
    """Profile a naive render and return (full_table, hottest_function_name).

    The hottest function is taken as the one with the largest cumulative time
    (excluding the profiler's own machinery and the script entry point).
    """
    profiler = cProfile.Profile()
    profiler.enable()
    mandelbrot_naive(view)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime")
    stats.print_stats(top)
    table = stream.getvalue()

    # Identify the hottest function by total (self) time.
    stats_dict = stats.stats  # type: ignore[attr-defined]
    hottest = max(
        stats_dict.items(),
        key=lambda kv: kv[1][2],  # index 2 == tottime
    )
    (filename, _lineno, funcname) = hottest[0]
    return table, funcname
