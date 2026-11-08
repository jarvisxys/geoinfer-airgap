"""Stitching-quality measures for study S1: seam lines, border band, seam score.

A seam line at position ``k`` is the boundary between pixel ``k - 1`` and pixel ``k``.
For a grid ``(T, O)`` the lines are the union of the tile edges and the centre-crop
edges, so every strategy is measured on the same lines: naive pasting switches source
at tile edges, crop stitching at crop edges, and blending spreads the change over the
overlap between them.

Seam score: mean absolute probability difference across the seam lines, divided by the
mean absolute difference between all other neighbouring pixel pairs in the same
direction. 1.0 means the borders look like the rest of the map (no visible seams);
a constant map scores 1.0 by definition.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from geoinfer.tiling.tiler import tile_grid

IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]


def seam_lines(height: int, width: int, tile: int, overlap: int) -> tuple[IntArray, IntArray]:
    """Row and column seam-line positions (interior only) for a tiling grid."""
    specs = tile_grid(height, width, tile, overlap)
    c0, c1 = specs[0].crop

    def lines(starts: set[int], n: int) -> IntArray:
        cand = set()
        for s in starts:
            cand |= {s, s + tile, s + c0, s + c1}
        return np.array(sorted(v for v in cand if 0 < v < n), dtype=np.intp)

    return lines({s.y0 for s in specs}, height), lines({s.x0 for s in specs}, width)


def _band_1d(lines: IntArray, n: int, width: int) -> BoolArray:
    out = np.zeros(n, dtype=bool)
    for k in lines:
        out[max(k - width, 0) : min(k + width, n)] = True
    return out


def border_band(
    shape: tuple[int, int], rows: IntArray, cols: IntArray, width: int = 32
) -> BoolArray:
    """Pixels within ``width`` of a seam line (``k - width <= i < k + width``).

    Parameters
    ----------
    shape
        ``(height, width)`` of the scene.
    rows, cols
        Seam-line positions from :func:`seam_lines`.
    width
        Band half-width in pixels (32 in the plan's border-band IoU).
    """
    r = _band_1d(np.asarray(rows, dtype=np.intp), shape[0], width)
    c = _band_1d(np.asarray(cols, dtype=np.intp), shape[1], width)
    return r[:, None] | c[None, :]


def label_edges(labels: npt.NDArray[np.integer], radius: int) -> BoolArray:
    """Pixels within ``radius`` of a change of label (e.g. a D4 strip join)."""
    h, w = labels.shape
    out = np.zeros((h, w), dtype=bool)
    if radius <= 0:
        return out
    # boundary lines between columns c-1 and c, and between rows r-1 and r
    vline = np.zeros((h, w + 1), dtype=bool)
    vline[:, 1:w] = labels[:, 1:] != labels[:, :-1]
    hline = np.zeros((h + 1, w), dtype=bool)
    hline[1:h, :] = labels[1:, :] != labels[:-1, :]
    for d in range(-radius + 1, radius + 1):
        # pixel i lies within radius of line k when k - radius <= i < k + radius
        lo, hi = max(d, 0), min(w + d, w + 1)
        out[:, lo - d : hi - d] |= vline[:, lo:hi]
        lo, hi = max(d, 0), min(h + d, h + 1)
        out[lo - d : hi - d, :] |= hline[lo:hi, :]
    return out


def seam_score(
    prob: npt.NDArray[np.floating],
    rows: IntArray,
    cols: IntArray,
    valid: BoolArray | None = None,
    exclude: BoolArray | None = None,
) -> float:
    """Across-seam difference over off-seam difference (see module docstring).

    Pairs touching an invalid or excluded pixel are skipped. NaN if no seam pair is
    left; ``inf`` if the map is flat everywhere except on the seams.
    """
    p = np.asarray(prob, dtype=np.float64)
    ok = np.isfinite(p)
    if valid is not None:
        ok &= valid
    if exclude is not None:
        ok &= ~exclude
    h, w = p.shape
    on_sum = off_sum = 0.0
    on_n = off_n = 0
    for axis, lines, n in ((0, rows, h), (1, cols, w)):
        if n < 2:
            continue
        d = np.abs(np.diff(p, axis=axis))
        pair_ok = (ok[1:, :] & ok[:-1, :]) if axis == 0 else (ok[:, 1:] & ok[:, :-1])
        is_line = np.zeros(n - 1, dtype=bool)
        idx = np.asarray(lines, dtype=np.intp) - 1
        is_line[idx[(idx >= 0) & (idx < n - 1)]] = True
        on = pair_ok & (is_line[:, None] if axis == 0 else is_line[None, :])
        off = pair_ok & ~on
        on_sum += float(np.where(on, d, 0.0).sum())
        off_sum += float(np.where(off, d, 0.0).sum())
        on_n += int(on.sum())
        off_n += int(off.sum())
    if on_n == 0:
        return math.nan
    on_mean = on_sum / on_n
    off_mean = off_sum / off_n if off_n else 0.0
    if off_mean == 0.0:
        return 1.0 if on_mean == 0.0 else math.inf
    return on_mean / off_mean
