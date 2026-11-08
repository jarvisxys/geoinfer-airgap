"""Binary clean-up of building probabilities (pipeline step 8).

Chain: threshold -> opening (drops speckle and thin false positives) -> closing (fills
roof holes) -> removal of components below ``min_area_m2``. OpenCV's default border
makes the scene edge neutral: erosion treats outside pixels as foreground and dilation
as background, so objects touching the edge are not eaten away.

Large rasters are cleaned block by block with a halo of
``(opening - 1) + (closing - 1) + min_px`` pixels, which makes the result identical to
cleaning the whole array: the morphology of a core pixel depends on pixels at most
``(opening - 1) + (closing - 1)`` away, and a component touching the core either lies
within ``min_px`` of it (then it is fully inside the halo and its area is exact) or has
at least ``min_px`` exact pixels on its way out (then it is kept, as it should be).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from rasterio.transform import Affine
from rasterio.windows import Window

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import RasterReader, RasterWriter
from geoinfer.postprocess.config import PostprocessConfig

BoolArray = npt.NDArray[np.bool_]
MASK_NODATA = 255


def binarize(prob: npt.NDArray[Any], threshold: float, valid: BoolArray | None = None) -> BoolArray:
    """``prob >= threshold``; NaN and invalid pixels are background."""
    with np.errstate(invalid="ignore"):
        out = np.asarray(prob >= threshold, dtype=bool)
    if valid is not None:
        out &= valid
    return out


def _morph(mask: BoolArray, size: int, op: int) -> BoolArray:
    if size < 1 or size % 2 == 0:
        raise ValueError(f"kernel size must be odd and >= 1, got {size}")
    if size == 1:
        return mask.copy()
    kernel = np.ones((size, size), np.uint8)
    return np.asarray(cv2.morphologyEx(mask.astype(np.uint8), op, kernel), dtype=bool)


def opening(mask: BoolArray, size: int) -> BoolArray:
    """Morphological opening with a ``size x size`` square."""
    return _morph(mask, size, cv2.MORPH_OPEN)


def closing(mask: BoolArray, size: int) -> BoolArray:
    """Morphological closing with a ``size x size`` square."""
    return _morph(mask, size, cv2.MORPH_CLOSE)


def min_area_px(transform: Affine, min_area_m2: float) -> int:
    """Smallest component size in pixels to keep, for the raster's pixel size."""
    px_area = float(abs(transform.a * transform.e))
    return math.ceil(min_area_m2 / px_area - 1e-9)


def label_components(mask: BoolArray, connectivity: int) -> tuple[npt.NDArray[np.int32], int]:
    """Label connected components; returns ``(labels, n)`` with 0 = background."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return np.asarray(labels, dtype=np.int32), int(n) - 1


def remove_small(mask: BoolArray, min_px: int, connectivity: int) -> BoolArray:
    """Drop connected components with fewer than ``min_px`` pixels."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if min_px <= 1:
        return mask.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def clean_mask(
    prob: npt.NDArray[Any],
    valid: BoolArray | None,
    cfg: PostprocessConfig,
    transform: Affine,
) -> BoolArray:
    """Threshold, open, close and remove small components (whole array)."""
    m = binarize(prob, cfg.threshold, valid)
    m = closing(opening(m, cfg.opening), cfg.closing)
    return remove_small(m, min_area_px(transform, cfg.min_area_m2), cfg.connectivity)


def halo_px(cfg: PostprocessConfig, transform: Affine) -> int:
    """Halo that makes block-wise cleaning exact (see module docstring)."""
    return (cfg.opening - 1) + (cfg.closing - 1) + min_area_px(transform, cfg.min_area_m2)


def _blocks(h: int, w: int, block: int, halo: int) -> list[tuple[Window, Window]]:
    """``(read window with halo, core window)`` pairs, both in scene coordinates."""
    out = []
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            r1, c1 = min(r0 + block, h), min(c0 + block, w)
            er0, ec0 = max(r0 - halo, 0), max(c0 - halo, 0)
            er1, ec1 = min(r1 + halo, h), min(c1 + halo, w)
            out.append((Window(ec0, er0, ec1 - ec0, er1 - er0), Window(c0, r0, c1 - c0, r1 - r0)))
    return out


def _inner(ext: Window, core: Window) -> tuple[slice, slice]:
    """Slices of ``core`` inside an array read for ``ext``."""
    r = int(core.row_off - ext.row_off)
    c = int(core.col_off - ext.col_off)
    return slice(r, r + int(core.height)), slice(c, c + int(core.width))


def clean_mask_blockwise(
    prob: npt.NDArray[Any],
    valid: BoolArray | None,
    cfg: PostprocessConfig,
    transform: Affine,
) -> BoolArray:
    """Same result as :func:`clean_mask`, computed in ``cfg.block`` blocks with a halo."""
    h, w = prob.shape
    out = np.zeros((h, w), dtype=bool)
    for ext, core in _blocks(h, w, cfg.block, halo_px(cfg, transform)):
        er, ec = ext.toslices()
        v = None if valid is None else valid[er, ec]
        m = clean_mask(prob[er, ec], v, cfg, transform)
        out[core.toslices()] = m[_inner(ext, core)]
    return out


def clean_raster(
    src: str | Path,
    dst: str | Path,
    cfg: PostprocessConfig,
    io_cfg: IOConfig,
    band: int = 1,
) -> None:
    """Clean a probability raster block-wise into a uint8 mask COG.

    The output is 1 = building, 0 = background, 255 = no prediction (input nodata),
    on the input's grid (CRS and transform kept).
    """
    with RasterReader(src) as r:
        info = r.info
        halo = halo_px(cfg, info.transform)
        with RasterWriter(
            dst,
            info,
            io_cfg,
            count=1,
            dtype="uint8",
            nodata=MASK_NODATA,
            overview_resampling="NEAREST",
        ) as wr:
            for ext, core in _blocks(info.height, info.width, cfg.block, halo):
                win = r.read(ext, [band])
                m = clean_mask(win.data[0], win.valid, cfg, info.transform)
                inner = _inner(ext, core)
                out = np.where(win.valid[inner], m[inner], MASK_NODATA).astype(np.uint8)
                wr.write(out[None], core)
