"""Synthetic clouds for the S3 cloud-robustness curve.

Cloud shapes come from fractal noise made by spectral synthesis (white noise shaped to
a power spectrum ~ 1/f^beta, the same family as Perlin / fBm noise, but vectorised and
seeded). The noise is thresholded at a quantile, so the cloud fraction hits the target
exactly, with a soft edge, and alpha-blended towards a bright cloud value per band.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def fractal_noise(shape: tuple[int, int], beta: float, seed: int) -> FloatArray:
    """Seeded fractal noise in [0, 1] with power spectrum ~ ``1 / f**beta``.

    Larger ``beta`` gives smoother, larger blobs.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    spec = np.fft.rfft2(rng.normal(size=shape))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = np.inf  # drop the DC term
    n = np.fft.irfft2(spec * f ** (-beta / 2.0), s=shape)
    n = (n - n.min()) / (n.max() - n.min())
    return n.astype(np.float32)


def cloud_mask(
    shape: tuple[int, int], coverage: float, beta: float, softness: float, seed: int
) -> FloatArray:
    """Cloud opacity ``alpha`` in [0, 1] whose ``alpha >= 0.5`` area equals ``coverage``.

    Parameters
    ----------
    shape
        ``(h, w)``.
    coverage
        Target cloud fraction in [0, 1].
    beta
        Spectral slope of the noise.
    softness
        Edge width in noise units; alpha ramps from 0 to 1 across it.
    seed
        Random seed.
    """
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be in [0, 1], got {coverage}")
    if coverage == 0.0:
        return np.zeros(shape, dtype=np.float32)
    if coverage == 1.0:
        return np.ones(shape, dtype=np.float32)
    n = fractal_noise(shape, beta, seed)
    t = float(np.quantile(n, 1.0 - coverage))
    return np.asarray(np.clip(0.5 + (n - t) / softness, 0.0, 1.0), dtype=np.float32)


def cloud_fraction(alpha: npt.NDArray[Any]) -> float:
    """Share of pixels counted as cloud (``alpha >= 0.5``)."""
    return float(np.mean(alpha >= 0.5))


def apply_clouds(
    x: npt.NDArray[Any],
    valid: BoolArray,
    alpha: npt.NDArray[Any],
    brightness: Sequence[float],
) -> npt.NDArray[Any]:
    """Blend clouds into optical bands: ``(1 - alpha) * x + alpha * brightness``.

    Parameters
    ----------
    x
        ``(bands, h, w)`` optical.
    valid
        ``(h, w)`` mask; invalid pixels are left unchanged.
    alpha
        ``(h, w)`` opacity from :func:`cloud_mask`.
    brightness
        Cloud value per band (e.g. the train-split p98 white point).

    Returns
    -------
    numpy.ndarray
        Same shape and dtype as ``x`` (integers are rounded and clipped to range).
    """
    b = np.asarray(brightness, dtype=np.float64).reshape(-1, 1, 1)
    if b.shape[0] != x.shape[0]:
        raise ValueError(f"brightness has {b.shape[0]} values for {x.shape[0]} bands")
    a = alpha.astype(np.float64)[None]
    out = (1.0 - a) * x.astype(np.float64) + a * b
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    out = out.astype(x.dtype)
    out[:, ~valid] = x[:, ~valid]
    return out


class TileClouds:
    """Deterministic synthetic clouds per tile, for the S3 robustness curve.

    Each tile gets its own cloud field (seed = ``seed`` combined with a CRC of the tile
    ID), so every model is tested on identical clouds and a re-run reproduces them.

    Parameters
    ----------
    coverage
        Target cloud fraction in [0, 1); 0 leaves tiles unchanged.
    beta, softness
        Fractal-noise slope and cloud-edge width (see :func:`cloud_mask`).
    brightness
        Cloud value per band (train-split p98 white point by default in the config).
    seed
        Base seed.
    """

    def __init__(
        self,
        coverage: float,
        beta: float,
        softness: float,
        brightness: Sequence[float],
        seed: int,
    ) -> None:
        if not 0.0 <= coverage < 1.0:
            raise ValueError(f"coverage must be in [0, 1), got {coverage}")
        self.coverage, self.beta, self.softness = coverage, beta, softness
        self.brightness, self.seed = list(brightness), seed

    def __call__(self, x: npt.NDArray[Any], valid: BoolArray, tile_id: str) -> npt.NDArray[Any]:
        """Cloudy copy of ``(bands, h, w)`` optical ``x``."""
        if self.coverage == 0.0:
            return x
        tile_seed = (self.seed * 1_000_003 + zlib.crc32(tile_id.encode())) % 2**32
        alpha = cloud_mask(x.shape[1:], self.coverage, self.beta, self.softness, tile_seed)
        return apply_clouds(x, valid, alpha, self.brightness)
