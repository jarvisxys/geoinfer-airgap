"""SAR radiometry: dB <-> linear, Lee speckle filter, ENL and edge preservation (S2).

SpaceNet 6 SAR-Intensity is stored in dB (10 * log10 of 2x2-multilooked intensity;
0 = nodata, 1e-5 = clipped floor). Speckle is multiplicative in linear intensity, which
is the model the Lee filter assumes, so the improved chain is
dB -> linear -> Lee -> dB, followed by clipping and z-scoring (see ``stats.py``).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_BORDER = cv2.BORDER_REFLECT_101


def db_to_linear(db: npt.NDArray[Any], valid: BoolArray) -> FloatArray:
    """Convert dB to linear intensity, ``10 ** (db / 10)``; invalid pixels become 0."""
    out = np.power(10.0, db.astype(np.float64) / 10.0)
    out[..., ~valid] = 0.0
    return out.astype(np.float32)


def linear_to_db(lin: npt.NDArray[Any], valid: BoolArray) -> FloatArray:
    """Convert linear intensity to dB, ``10 * log10(lin)``; invalid pixels become 0."""
    safe = np.maximum(lin.astype(np.float64), np.finfo(np.float64).tiny)
    out = 10.0 * np.log10(safe)
    out[..., ~valid] = 0.0
    return out.astype(np.float32)


def _box_sum(a: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.float64]:
    out = np.asarray(
        cv2.boxFilter(a, ddepth=-1, ksize=(size, size), normalize=False, borderType=_BORDER),
        dtype=np.float64,
    )
    return out


def lee_filter(lin: npt.NDArray[Any], valid: BoolArray, size: int, enl: float) -> FloatArray:
    r"""Lee speckle filter on linear intensity, vectorised with box filters.

    .. math::
        \hat I = m + K (I - m), \quad
        K = \frac{v - m^2 \sigma_n^2}{v (1 + \sigma_n^2)}, \quad \sigma_n^2 = 1/\mathrm{ENL}

    with local mean ``m`` and variance ``v`` over a ``size x size`` window. ``K`` is
    clipped to [0, 1] and set to 0 where ``v = 0``, so a constant image is returned
    unchanged. Local statistics use valid pixels only, so nodata never bleeds in.

    Parameters
    ----------
    lin
        Linear intensity, ``(bands, h, w)`` or ``(h, w)``.
    valid
        ``(h, w)`` mask of usable pixels.
    size
        Odd window size >= 3 (5 and 7 in the S2 study).
    enl
        Equivalent number of looks of the speckle (> 0).

    Returns
    -------
    numpy.ndarray
        float32, same shape as ``lin``; invalid pixels are 0.
    """
    if size < 3 or size % 2 == 0:
        raise ValueError(f"size must be odd and >= 3, got {size}")
    if enl <= 0:
        raise ValueError(f"enl must be > 0, got {enl}")
    x = np.asarray(lin, dtype=np.float64)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[np.newaxis]
    m = valid.astype(np.float64)
    n = _box_sum(m, size)
    has = n > 0
    n_safe = np.where(has, n, 1.0)
    sn2 = 1.0 / enl

    out = np.zeros(x.shape, dtype=np.float32)
    for b in range(x.shape[0]):
        xb = np.where(valid, x[b], 0.0)
        mean = _box_sum(xb, size) / n_safe
        var = np.maximum(_box_sum(xb * xb, size) / n_safe - mean * mean, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = (var - mean * mean * sn2) / (var * (1.0 + sn2))
        k = np.clip(np.where(var > 0, k, 0.0), 0.0, 1.0)
        res = mean + k * (xb - mean)
        out[b] = np.where(valid & has, res, 0.0)
    return out[0] if squeeze else out


def enl(x: npt.NDArray[Any], mask: BoolArray | None = None) -> float:
    """Equivalent number of looks, ``mean**2 / var``, of linear intensity.

    Meaningful on a homogeneous patch, where all variation is speckle.

    Parameters
    ----------
    x
        Linear intensity patch.
    mask
        Optional mask of pixels to use.
    """
    vals = (x[mask] if mask is not None else x.reshape(-1)).astype(np.float64)
    var = float(vals.var())
    return float(vals.mean() ** 2 / var) if var > 0 else float("inf")


def estimate_enl(lin: npt.NDArray[Any], valid: BoolArray, window: int, quantile: float) -> float:
    """Estimate ENL from the most homogeneous windows of a linear-intensity band.

    The band is cut into non-overlapping ``window x window`` blocks (fully valid ones
    only). Rows are split into two interleaved sets (alternating pairs of rows): the
    coefficient of variation on the first set selects the ``quantile`` most homogeneous
    blocks, and ENL is measured on the second set. Using different pixels for selection
    and measurement avoids the upward bias of picking low-variance blocks by their own
    variance. Returns the median ENL of the selected blocks.

    Parameters
    ----------
    lin
        ``(h, w)`` linear intensity.
    valid
        ``(h, w)`` mask of usable pixels.
    window
        Block size in pixels (multiple of 4).
    quantile
        Share of blocks (lowest coefficient of variation) to keep.
    """
    if window < 4 or window % 4:
        raise ValueError(f"window must be a multiple of 4, got {window}")
    h, w = (lin.shape[0] // window) * window, (lin.shape[1] // window) * window
    nh, nw = h // window, w // window

    def blocks(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return a[:h, :w].reshape(nh, window, nw, window).transpose(0, 2, 1, 3)

    x = blocks(lin.astype(np.float64)).reshape(nh * nw, window, window)
    full = blocks(valid).reshape(nh * nw, -1).all(axis=1)
    if not full.any():
        raise ValueError(f"no fully valid {window} x {window} window")
    x = x[full]
    rows = (np.arange(window) // 2) % 2 == 0
    sel, est = x[:, rows, :].reshape(len(x), -1), x[:, ~rows, :].reshape(len(x), -1)
    cv = sel.std(axis=1) / sel.mean(axis=1)
    keep = np.argsort(cv)[: max(1, int(np.ceil(quantile * len(x))))]
    e = est[keep]
    return float(np.median(e.mean(axis=1) ** 2 / e.var(axis=1)))


def edge_preservation_index(
    original: npt.NDArray[Any], filtered: npt.NDArray[Any], valid: BoolArray
) -> float:
    """Edge-preservation index: sum of |neighbour differences| after / before filtering.

    Horizontal and vertical neighbour pairs count only when both pixels are valid. 1.0
    means edges are fully kept, lower values mean smoothing; a constant image gives 1.0.
    Compute it in dB for SAR so bright and dark edges weigh alike.
    """
    o, f = original.astype(np.float64), filtered.astype(np.float64)
    ph, pv = valid[:, 1:] & valid[:, :-1], valid[1:, :] & valid[:-1, :]
    num = np.abs(np.diff(f, axis=1))[ph].sum() + np.abs(np.diff(f, axis=0))[pv].sum()
    den = np.abs(np.diff(o, axis=1))[ph].sum() + np.abs(np.diff(o, axis=0))[pv].sum()
    return float(num / den) if den > 0 else 1.0


def sar_chain(
    db: npt.NDArray[Any], valid: BoolArray, lee: int | None, enl_value: float
) -> FloatArray:
    """Apply one S2 SAR variant to stored dB data: optionally dB -> linear -> Lee -> dB.

    Parameters
    ----------
    db
        ``(bands, h, w)`` SN6 SAR-Intensity (dB).
    valid
        ``(h, w)`` mask of usable pixels.
    lee
        Lee window size, or None for no filtering.
    enl_value
        ENL used by the Lee filter.

    Returns
    -------
    numpy.ndarray
        float32 dB; invalid pixels are 0. Clipping and z-scoring happen afterwards.
    """
    if lee is None:
        return np.where(valid, db, 0.0).astype(np.float32)
    lin = db_to_linear(db, valid)
    return linear_to_db(lee_filter(lin, valid, lee, enl_value), valid)
