"""SAR / optical co-registration check by phase correlation of edge maps (stage 2).

SpaceNet 6 SAR and optical tiles share one grid, but that should be verified, not
assumed. SAR and optical intensities are not comparable (a bright roof in one can be
dark in the other), so both are reduced to Sobel edge magnitudes, which are blind to
contrast sign and scale, and the sub-pixel shift between them is found by phase
correlation under a Hann window, refined with the upsampled-DFT method of
Guizar-Sicairos et al. (2008) via ``skimage.registration.phase_cross_correlation``.
On synthetic edge maps that method is about 2x more accurate than the centroid
refinement of ``cv2.phaseCorrelate`` (median error ~0.06 px vs ~0.1-0.16 px).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt
from skimage.registration import phase_cross_correlation

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class CoregResult(NamedTuple):
    """Shift of the SAR edge map relative to the optical one.

    Attributes
    ----------
    dx, dy
        Shift in pixels (positive: SAR content lies to the right / below).
    response
        Phase-correlation peak height in [0, 1]; low values mean an unreliable match.
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


def phase_correlation_shift(
    ref: npt.NDArray[Any], mov: npt.NDArray[Any], upsample: int = 50
) -> tuple[float, float, float]:
    """Sub-pixel shift of ``mov`` relative to ``ref`` (Hann-windowed phase correlation).

    Parameters
    ----------
    ref, mov
        Same-shape single-band images.
    upsample
        Sub-pixel precision is ``1 / upsample`` px.

    Returns
    -------
    tuple
        ``(dx, dy, response)`` with ``mov(x, y) ~ ref(x - dx, y - dy)``. ``response``
        is the peak of the phase-correlation surface in [0, 1]; near 1 for a clean
        match, low when the images share little structure.
    """
    if ref.shape != mov.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {mov.shape}")
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_64F)
    a = (ref.astype(np.float64) - float(ref.mean())) * win
    b = (mov.astype(np.float64) - float(mov.mean())) * win
    shift, _, _ = phase_cross_correlation(  # type: ignore[no-untyped-call]
        a, b, upsample_factor=upsample, normalization="phase"
    )
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    cross /= np.maximum(np.abs(cross), np.finfo(np.float64).tiny)
    response = float(np.real(np.fft.ifft2(cross)).max())
    # skimage returns the shift that registers mov onto ref, i.e. minus ours
    return float(-shift[1]), float(-shift[0]), response


def coregistration_shift(
    sar_db: npt.NDArray[Any],
    optical: npt.NDArray[Any],
    valid: BoolArray,
    window: int,
    ksize: int,
    optical_bands: Sequence[int],
    upsample: int = 50,
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
    upsample
        Sub-pixel precision is ``1 / upsample`` px.
    """
    h, w = valid.shape
    s = min(window, h, w)
    r0, c0 = (h - s) // 2, (w - s) // 2
    sl = (slice(r0, r0 + s), slice(c0, c0 + s))
    v = valid[sl]
    sar = sar_db[:, sl[0], sl[1]].astype(np.float64).mean(axis=0)
    opt = optical[[b - 1 for b in optical_bands], sl[0], sl[1]].astype(np.float64).mean(axis=0)
    dx, dy, resp = phase_correlation_shift(
        edge_magnitude(opt, v, ksize), edge_magnitude(sar, v, ksize), upsample
    )
    return CoregResult(dx, dy, resp, s)
