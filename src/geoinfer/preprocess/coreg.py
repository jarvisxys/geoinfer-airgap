"""SAR / optical co-registration check by masked correlation of edge maps (stage 2).

SpaceNet 6 SAR and optical tiles share one grid, but that should be verified, not
assumed. SAR and optical intensities are not comparable (a bright roof in one can be
dark in the other), so both are lightly smoothed and reduced to Sobel edge magnitudes,
which are blind to contrast sign and scale. The shift between the edge maps is the peak
of their masked normalised cross-correlation (Padfield 2012), searched within
``max_shift`` px and refined to sub-pixel by a parabola through the peak.

Masking matters: SN6 tiles are 20-50 % nodata with the same footprint in both sensors.
Plain (phase) correlation then locks onto the shared mask outline and reports zero
shift whatever the true offset is; masked NCC only compares pixels valid in both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class CoregResult(NamedTuple):
    """Shift of the SAR edge map relative to the optical one.

    Attributes
    ----------
    dx, dy
        Shift in pixels (positive: SAR content lies to the right / below).
    response
        Peak masked NCC of the edge maps in [-1, 1]; low values mean an unreliable match.
    window
        Side of the square central window used.
    """

    dx: float
    dy: float
    response: float
    window: int


def edge_magnitude(img: npt.NDArray[Any], valid: BoolArray, ksize: int) -> FloatArray:
    """Sobel gradient magnitude, zeroed on and next to invalid pixels.

    Parameters
    ----------
    img
        ``(h, w)`` single band.
    valid
        ``(h, w)`` mask of usable pixels.
    ksize
        Sobel kernel size (1, 3, 5 or 7).
    """
    x = np.where(valid, img, 0.0).astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=ksize)
    keep = cv2.erode(valid.astype(np.uint8), np.ones((ksize, ksize), np.uint8)) > 0
    return np.where(keep, np.hypot(gx, gy), 0.0).astype(np.float32)


def _corr(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], shape: tuple[int, int]) -> Any:
    """``c[d] = sum_x a(x) b(x + d)`` for all lags, by zero-padded real FFTs."""
    return np.fft.irfft2(np.conj(np.fft.rfft2(a, shape)) * np.fft.rfft2(b, shape), shape)


def _parabola(lo: float, mid: float, hi: float) -> float:
    den = lo - 2.0 * mid + hi
    if not np.isfinite(den) or den >= 0.0:  # not a strict maximum
        return 0.0
    return float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5))


def masked_ncc_shift(
    ref: npt.NDArray[Any],
    mov: npt.NDArray[Any],
    ref_valid: BoolArray,
    mov_valid: BoolArray,
    max_shift: int,
    overlap_ratio: float,
) -> tuple[float, float, float]:
    """Shift of ``mov`` relative to ``ref`` by masked normalised cross-correlation.

    For every integer lag within ``max_shift`` the NCC is computed over the pixels
    valid in both images (Padfield 2012, via FFTs); the best lag is refined by a
    parabola through the peak and its neighbours along each axis.

    Parameters
    ----------
    ref, mov
        Same-shape single-band images.
    ref_valid, mov_valid
        Masks of usable pixels.
    max_shift
        Search radius in pixels (lags in ``[-max_shift, max_shift]``).
    overlap_ratio
        Lags whose overlap is below this share of the smaller mask are skipped.

    Returns
    -------
    tuple
        ``(dx, dy, peak)`` with ``mov(x, y) ~ ref(x - dx, y - dy)`` and ``peak`` the
        NCC at the best integer lag, in [-1, 1].

    Raises
    ------
    ValueError
        On shape mismatch, bad parameters, or no lag with enough overlap.
    """
    if not (ref.shape == mov.shape == ref_valid.shape == mov_valid.shape):
        raise ValueError(f"shape mismatch: {ref.shape} vs {mov.shape}")
    if max_shift < 1:
        raise ValueError(f"max_shift must be >= 1, got {max_shift}")
    if not 0.0 < overlap_ratio <= 1.0:
        raise ValueError(f"overlap_ratio must be in (0, 1], got {overlap_ratio}")
    mf, mm = ref_valid.astype(np.float64), mov_valid.astype(np.float64)
    if mf.sum() == 0 or mm.sum() == 0:
        raise ValueError("no valid overlap: an image mask is empty")
    # centre on the valid means to limit cancellation in the variance terms
    f = np.where(ref_valid, ref - ref[ref_valid].mean(), 0.0).astype(np.float64)
    m = np.where(mov_valid, mov - mov[mov_valid].mean(), 0.0).astype(np.float64)

    h, w = ref.shape
    shape = (cv2.getOptimalDFTSize(h + max_shift), cv2.getOptimalDFTSize(w + max_shift))
    n = np.rint(_corr(mf, mm, shape))
    sf, sm = _corr(f, mm, shape), _corr(mf, m, shape)
    sff, smm, sfm = _corr(f * f, mm, shape), _corr(mf, m * m, shape), _corr(f, m, shape)

    lags = np.arange(-max_shift, max_shift + 1)
    iy, ix = np.ix_(lags % shape[0], lags % shape[1])
    n, sf, sm, sff, smm, sfm = (x[iy, ix] for x in (n, sf, sm, sff, smm, sfm))
    with np.errstate(divide="ignore", invalid="ignore"):
        var_f = sff - sf * sf / n
        var_m = smm - sm * sm / n
        ncc = (sfm - sf * sm / n) / np.sqrt(var_f * var_m)
    tiny = 1e-12 * max(float(np.abs(f).max()), float(np.abs(m).max()), 1e-300) ** 2
    ok = (n >= overlap_ratio * min(mf.sum(), mm.sum())) & (var_f > tiny) & (var_m > tiny)
    ncc = np.where(ok, ncc, -np.inf)
    if not np.isfinite(ncc).any():
        raise ValueError("no lag has enough valid overlap")

    i, j = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    last = 2 * max_shift
    di = _parabola(ncc[i - 1, j], ncc[i, j], ncc[i + 1, j]) if 0 < i < last else 0.0
    dj = _parabola(ncc[i, j - 1], ncc[i, j], ncc[i, j + 1]) if 0 < j < last else 0.0
    peak = float(np.clip(ncc[i, j], -1.0, 1.0))
    return float(lags[j] + dj), float(lags[i] + di), peak


def _mask_margin(ksize: int, blur_sigma: float) -> int:
    return ksize // 2 + int(np.ceil(3.0 * blur_sigma))


def coregistration_shift(
    sar_db: npt.NDArray[Any],
    optical: npt.NDArray[Any],
    valid: BoolArray,
    window: int,
    ksize: int,
    optical_bands: Sequence[int],
    blur_sigma: float,
    max_shift: int,
    overlap_ratio: float,
) -> CoregResult:
    """Estimate the SAR-to-optical shift of one co-gridded tile.

    Parameters
    ----------
    sar_db
        ``(bands, h, w)`` SAR in dB; the band mean is used.
    optical
        ``(bands, h, w)`` optical; the mean of ``optical_bands`` is used.
    valid
        ``(h, w)`` pixels valid in both sensors.
    window
        Side of the central square window; tiles smaller than this are used whole.
    ksize
        Sobel kernel size.
    optical_bands
        1-based optical bands averaged into a luminance image.
    blur_sigma
        Gaussian smoothing (px) before the edges; suppresses SAR speckle edges.
        0 disables it.
    max_shift, overlap_ratio
        See :func:`masked_ncc_shift`.
    """
    h, w = valid.shape
    s = min(window, h, w)
    r0, c0 = (h - s) // 2, (w - s) // 2
    sl = (slice(r0, r0 + s), slice(c0, c0 + s))
    v = valid[sl]
    sar = sar_db[:, sl[0], sl[1]].astype(np.float64).mean(axis=0)
    opt = optical[[b - 1 for b in optical_bands], sl[0], sl[1]].astype(np.float64).mean(axis=0)
    if blur_sigma > 0 and v.any():
        # fill nodata with the valid mean so the blur does not smear a step into the tile
        sar = np.asarray(
            cv2.GaussianBlur(np.where(v, sar, sar[v].mean()), (0, 0), blur_sigma), np.float64
        )
        opt = np.asarray(
            cv2.GaussianBlur(np.where(v, opt, opt[v].mean()), (0, 0), blur_sigma), np.float64
        )
    k = 2 * _mask_margin(ksize, blur_sigma) + 1
    keep = cv2.erode(v.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    dx, dy, peak = masked_ncc_shift(
        edge_magnitude(opt, v, ksize),
        edge_magnitude(sar, v, ksize),
        keep,
        keep,
        max_shift,
        overlap_ratio,
    )
    return CoregResult(dx, dy, peak, s)
