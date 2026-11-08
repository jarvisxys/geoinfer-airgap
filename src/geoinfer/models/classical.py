"""Classical "no deep learning" baseline: Otsu thresholding (plan step 6).

Otsu's method picks the threshold that maximises the between-class variance
``w0 w1 (m0 - m1)^2`` of a histogram. As in ``skimage.filters.threshold_otsu`` the
threshold is the centre of the last class-0 bin and pixels ``> t`` are foreground
(bright SAR returns, e.g. building double bounce).

``otsu_per_label`` computes one threshold per label (e.g. per SAR acquisition strip) in
a single pass, because SN6 SAR brightness differs between strips.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def otsu_from_hist(counts: npt.NDArray[Any], edges: npt.NDArray[Any]) -> float:
    """Otsu threshold of a histogram; NaN when no split separates two classes."""
    c = np.asarray(counts, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    if e.size != c.size + 1:
        raise ValueError(f"need len(edges) == len(counts) + 1, got {e.size} and {c.size}")
    centres = 0.5 * (e[:-1] + e[1:])
    w0 = np.cumsum(c)[:-1]
    s0 = np.cumsum(c * centres)[:-1]
    total, total_s = c.sum(), float((c * centres).sum())
    w1 = total - w0
    ok = (w0 > 0) & (w1 > 0)
    if not ok.any():
        return math.nan
    m0 = np.divide(s0, w0, out=np.zeros_like(s0), where=ok)
    m1 = np.divide(total_s - s0, w1, out=np.zeros_like(s0), where=ok)
    between = np.where(ok, w0 * w1 * (m0 - m1) ** 2, -1.0)
    return float(centres[int(np.argmax(between))])


def _bin_index(x: npt.NDArray[Any], bins: int, lo: float, hi: float) -> tuple[Any, Any]:
    """Bin of each value on ``np.histogram``'s edges (last bin closed); in-range mask."""
    if not hi > lo:
        raise ValueError(f"empty value range [{lo}, {hi}]")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.where(x == hi, bins - 1, idx)
    inside = np.isfinite(x) & (x >= lo) & (x <= hi)
    return idx, inside


def otsu_threshold(
    x: npt.NDArray[Any], valid: BoolArray | None, bins: int, lo: float, hi: float
) -> float:
    """Otsu threshold of the valid values of ``x`` within ``[lo, hi]``."""
    v = np.asarray(x, dtype=np.float64)
    if valid is not None:
        v = v[valid]
    v = v.ravel()
    idx, inside = _bin_index(v, bins, lo, hi)
    counts = np.bincount(idx[inside], minlength=bins)
    return otsu_from_hist(counts, np.linspace(lo, hi, bins + 1))


def otsu_per_label(
    x: npt.NDArray[Any],
    labels: npt.NDArray[np.integer[Any]],
    valid: BoolArray | None,
    n_labels: int,
    bins: int,
    lo: float,
    hi: float,
) -> npt.NDArray[np.float64]:
    """One Otsu threshold per label ``0 .. n_labels - 1``; NaN where undefined."""
    v = np.asarray(x, dtype=np.float64).ravel()
    lab = np.asarray(labels).ravel().astype(np.int64)
    idx, inside = _bin_index(v, bins, lo, hi)
    keep = inside & (lab >= 0) & (lab < n_labels)
    if valid is not None:
        keep &= np.asarray(valid).ravel()
    flat = np.bincount(lab[keep] * bins + idx[keep], minlength=n_labels * bins)
    hist: npt.NDArray[np.int64] = flat.reshape(n_labels, bins)
    edges = np.linspace(lo, hi, bins + 1)
    return np.array([otsu_from_hist(hist[k], edges) for k in range(n_labels)])
