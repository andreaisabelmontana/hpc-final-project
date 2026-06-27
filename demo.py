"""
Render the Mandelbrot view and save it as an image.

    python demo.py

Writes ``figures/mandelbrot.png``. Uses the parallel kernel so the full-detail
image renders quickly; the result is identical to the naive reference.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

from hpc.mandelbrot import DEFAULT_VIEW, mandelbrot_parallel

FIGURES = os.path.join(os.path.dirname(__file__), "figures")


def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)
    view = DEFAULT_VIEW

    print(
        f"Rendering {view.width}x{view.height} pixels, max_iter={view.max_iter} "
        f"on up to {os.cpu_count()} cores ..."
    )
    start = time.perf_counter()
    image = mandelbrot_parallel(view)  # default workers = cpu_count()
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.3f}s. Iteration counts: "
          f"min={image.min()}, max={image.max()}, mean={image.mean():.1f}")

    # A smooth log-scaled colour map makes the boundary filaments visible.
    shaded = np.log1p(image.astype(np.float64))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.imshow(
        shaded,
        cmap="magma",
        extent=(view.xmin, view.xmax, view.ymin, view.ymax),
        origin="lower",
        interpolation="bilinear",
    )
    ax.set_title("Mandelbrot set — seahorse valley", fontsize=11)
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    fig.tight_layout()

    out = os.path.join(FIGURES, "mandelbrot.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
