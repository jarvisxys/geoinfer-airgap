"""Optical / generic band normalisation (comparison S2).

Baseline: per-tile min-max, which gives the same roof a different brightness in
neighbouring tiles. Improved: a global percentile stretch with train-split statistics,
then a z-score. Arrays are ``(bands, h, w)``; ``valid`` is ``(h, w)``; invalid pixels
are set to 0 in every output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def _per_band(v: npt.ArrayLike, n: int, name: str) -> npt.NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} has {a.size} values for {n} bands")
    return a[:, None, None]


def minmax_per_tile(x: npt.NDArray[Any], valid: BoolArray) -> FloatArray:
    """Scale each band to [0, 1] with its own min and max over valid pixels (baseline).

    Parameters
    ----------
    x
        ``(bands, h, w)`` array, any numeric dtype.
    valid
        ``(h, w)`` mask of usable pixels.

    Returns
    -------
    numpy.ndarray
        float32 in [0, 1]; constant bands and invalid pixels are 0.
    """
    out = np.zeros(x.shape, dtype=np.float32)
    if not valid.any():
        return out
    for b in range(x.shape[0]):
        vals = x[b][valid].astype(np.float64)
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            out[b][valid] = ((vals - lo) / (hi - lo)).astype(np.float32)
    return out


def percentile_stretch(
    x: npt.NDArray[Any], valid: BoolArray, lo: npt.ArrayLike, hi: npt.ArrayLike
) -> FloatArray:
    """Map ``[lo, hi]`` per band linearly to [0, 1] and clip.

    Parameters
    ----------
    x
        ``(bands, h, w)`` array.
    valid
        ``(h, w)`` mask of usable pixels.
    lo, hi
        Per-band black and white points (e.g. train-split 2nd and 98th percentiles).

    Returns
    -------
    numpy.ndarray
        float32 in [0, 1]; invalid pixels are 0.
    """
    n = x.shape[0]
    lo_, hi_ = _per_band(lo, n, "lo"), _per_band(hi, n, "hi")
    if np.any(hi_ <= lo_):
        raise ValueError("every band needs hi > lo")
    out = np.clip((x.astype(np.float64) - lo_) / (hi_ - lo_), 0.0, 1.0)
    out[:, ~valid] = 0.0
    return out.astype(np.float32)


def zscore(
    x: npt.NDArray[Any], valid: BoolArray, mean: npt.ArrayLike, std: npt.ArrayLike
) -> FloatArray:
    """Standardise each band with given (train-split) mean and std.

    Returns
    -------
    numpy.ndarray
        float32 ``(x - mean) / std``; invalid pixels are 0 (the train mean).
    """
    n = x.shape[0]
    mean_, std_ = _per_band(mean, n, "mean"), _per_band(std, n, "std")
    if np.any(std_ <= 0):
        raise ValueError("every band needs std > 0")
    out = (x.astype(np.float64) - mean_) / std_
    out[:, ~valid] = 0.0
    return out.astype(np.float32)


def select_arm_stats(
    strip: str | None,
    global_stats: Mapping[str, Any],
    strip_stats: Mapping[str, Mapping[str, Any] | None],
) -> tuple[Mapping[str, Any], str]:
    """Statistics for a strip-scoped arm: the strip's own, else the global arm's.

    Returns
    -------
    tuple
        ``(stats, source)`` with ``source`` ``"strip"`` or ``"fallback"``.
    """
    own = strip_stats.get(strip) if strip is not None else None
    return (own, "strip") if own is not None else (global_stats, "fallback")


def apply_sar_arm(
    db: npt.NDArray[Any],
    valid: BoolArray,
    stats: Mapping[str, Any],
    domain: Literal["db", "linear"],
    clip: bool,
) -> FloatArray:
    """Normalise SAR for one S2 arm: domain -> optional percentile clip -> z-score.

    Parameters
    ----------
    db
        ``(bands, h, w)`` dB after the arm's speckle filter (``sar_chain``).
    valid
        ``(h, w)`` mask of usable pixels; invalid pixels come out as 0.
    stats
        ``p_lo``, ``p_hi``, ``mean``, ``std`` per band in the arm's domain (global or
        per strip).
    domain
        ``db`` or ``linear`` (``10 ** (dB / 10)``).
    clip
        Clip to ``[p_lo, p_hi]`` before standardising.
    """
    if domain not in ("db", "linear"):
        raise ValueError(f"domain must be 'db' or 'linear', got {domain!r}")
    x = db.astype(np.float64)
    if domain == "linear":
        x = np.power(10.0, x / 10.0)
    if clip:
        n = x.shape[0]
        lo, hi = _per_band(stats["p_lo"], n, "p_lo"), _per_band(stats["p_hi"], n, "p_hi")
        x = np.clip(x, lo, hi)
    return zscore(x, valid, stats["mean"], stats["std"])
