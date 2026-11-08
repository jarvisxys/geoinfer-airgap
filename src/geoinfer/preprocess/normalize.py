"""Optical / generic band normalisation (comparison S2).

Baseline: per-tile min-max, which gives the same roof a different brightness in
neighbouring tiles. Improved: a global percentile stretch with train-split statistics,
then a z-score. Arrays are ``(bands, h, w)``; ``valid`` is ``(h, w)``; invalid pixels
are set to 0 in every output.
"""

from __future__ import annotations

from typing import Any

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
