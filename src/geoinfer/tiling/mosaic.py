"""Mosaic tiles that live on slightly different pixel grids (used to build D4).

SN6 tiles of one collection strip share a grid, but each strip has its own sub-pixel
origin. Tiles are pasted onto one common grid at the nearest whole-pixel offset (nearest
neighbour: values are unchanged, positions move by at most half a pixel per axis), and
the first valid pixel wins, so callers control priority by paste order.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from rasterio.transform import Affine, from_origin

BoolArray = npt.NDArray[np.bool_]


def snap_grid(bounds: tuple[float, float, float, float], res: float) -> tuple[Affine, int, int]:
    """North-up grid on the ``res`` lattice that covers ``(minx, miny, maxx, maxy)``.

    Returns
    -------
    tuple
        ``(transform, width, height)``.
    """
    minx, miny, maxx, maxy = bounds
    if not (maxx > minx and maxy > miny):
        raise ValueError(f"empty bounds {bounds}")
    x0, y1 = math.floor(minx / res) * res, math.ceil(maxy / res) * res
    width = math.ceil((maxx - x0) / res - 1e-9)
    height = math.ceil((y1 - miny) / res - 1e-9)
    return from_origin(x0, y1, res, res), width, height


def paste_offset(src: Affine, dst: Affine) -> tuple[int, int, float, float]:
    """Nearest whole-pixel offset of a source grid inside a destination grid.

    Returns
    -------
    tuple
        ``(row, col, residual_rows, residual_cols)``: the source's top-left pixel goes to
        ``(row, col)``; the exact offset is ``(row + residual_rows, col + residual_cols)``,
        with residuals in ``[-0.5, 0.5]`` px.
    """
    if src.b or src.d or dst.b or dst.d:
        raise ValueError("rotation terms are not supported")
    if not (math.isclose(src.a, dst.a) and math.isclose(src.e, dst.e)):
        raise ValueError(f"resolution differs: {(src.a, src.e)} vs {(dst.a, dst.e)}")
    fc = (src.c - dst.c) / dst.a
    fr = (src.f - dst.f) / dst.e
    col, row = math.floor(fc + 0.5), math.floor(fr + 0.5)
    return row, col, fr - row, fc - col


def paste_first_valid(
    dst: npt.NDArray[Any],
    filled: BoolArray,
    src: npt.NDArray[Any],
    src_valid: BoolArray,
    row: int,
    col: int,
) -> BoolArray:
    """Copy valid source pixels into still-empty destination pixels, in place.

    Parameters
    ----------
    dst
        ``(bands, H, W)`` destination.
    filled
        ``(H, W)`` pixels already taken; updated.
    src
        ``(bands, h, w)`` source with the same band count.
    src_valid
        ``(h, w)`` usable source pixels.
    row, col
        Destination position of the source's top-left pixel; may lie outside.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` mask of the pixels this call filled, so callers can paste matching
        layers (e.g. the optical twin, a source-ID raster) at the same pixels.
    """
    if src.ndim != 3 or src.shape[0] != dst.shape[0]:
        raise ValueError(f"source shape {src.shape} does not match {dst.shape[0]} bands")
    if src_valid.shape != src.shape[1:]:
        raise ValueError(f"valid shape {src_valid.shape} != {src.shape[1:]}")
    big_h, big_w = filled.shape
    h, w = src_valid.shape
    r0, c0 = max(row, 0), max(col, 0)
    r1, c1 = min(row + h, big_h), min(col + w, big_w)
    new = np.zeros((big_h, big_w), dtype=bool)
    if r1 <= r0 or c1 <= c0:
        return new
    sv = src_valid[r0 - row : r1 - row, c0 - col : c1 - col]
    take = sv & ~filled[r0:r1, c0:c1]
    new[r0:r1, c0:c1] = take
    region = dst[:, r0:r1, c0:c1]
    region[:, take] = src[:, r0 - row : r1 - row, c0 - col : c1 - col][:, take]
    filled[r0:r1, c0:c1] |= take
    return new
