"""Raster inspection and windowed read / write that preserve CRS and transform.

Nothing here reads a whole raster: readers return one window at a time (with a size
cap) and writers accept one window at a time, so scenes larger than RAM stream through.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, Self

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.windows import Window

from geoinfer.io.cog import to_cog
from geoinfer.io.config import IOConfig
from geoinfer.io.paths import assert_writable

__all__ = [
    "DEFAULT_MAX_WINDOW_PX",
    "NotGeoreferencedError",
    "RasterInfo",
    "RasterReader",
    "RasterWriter",
    "WindowRead",
    "assert_writable",
    "inspect_raster",
    "read_window",
]

# Safety cap for callers that do not pass one; pipeline code passes IOConfig.max_window_px.
DEFAULT_MAX_WINDOW_PX = 4096 * 4096

BoolArray = npt.NDArray[np.bool_]


class NotGeoreferencedError(ValueError):
    """The raster has no CRS or no geotransform."""


@dataclass(frozen=True)
class RasterInfo:
    """Stage-1 ingest report for one raster.

    Attributes
    ----------
    path
        Source file.
    crs, transform
        Georeferencing; every output derived from this raster reuses them.
    width, height, count
        Size in pixels and number of bands.
    dtypes
        Per-band numpy dtype names.
    nodata
        Nodata value, or None.
    block_shape
        Internal block ``(rows, cols)`` of band 1.
    tiled
        True for a tiled layout (False for one-row-per-block strips).
    overviews
        Number of overview levels on band 1.
    compression
        Codec name (e.g. ``DEFLATE``) or None.
    layout
        GDAL ``IMAGE_STRUCTURE`` layout, ``COG`` for cloud-optimised files.
    """

    path: Path
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    nodata: float | None
    block_shape: tuple[int, int]
    tiled: bool
    overviews: int
    compression: str | None
    layout: str | None

    @property
    def is_cog(self) -> bool:
        """True if the file is a tiled GeoTIFF with the COG layout."""
        return self.layout == "COG" and self.tiled


def _info(src: DatasetReader, path: Path) -> RasterInfo:
    rows, cols = src.block_shapes[0]
    return RasterInfo(
        path=path,
        crs=src.crs,
        transform=src.transform,
        width=src.width,
        height=src.height,
        count=src.count,
        dtypes=tuple(src.dtypes),
        nodata=src.nodata,
        block_shape=(int(rows), int(cols)),
        tiled=bool(src.profile.get("tiled", False)),
        overviews=len(src.overviews(1)),
        compression=src.compression.value if src.compression else None,
        layout=src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"),
    )


def _check_georef(info: RasterInfo) -> None:
    if info.crs is None:
        raise NotGeoreferencedError(f"{info.path}: no CRS")
    if info.transform == Affine.identity():
        raise NotGeoreferencedError(f"{info.path}: no geotransform (identity transform)")


def inspect_raster(path: str | Path, *, require_georef: bool = True) -> RasterInfo:
    """Open a raster's header (no pixels are read) and report its structure.

    Parameters
    ----------
    path
        Raster file.
    require_georef
        Raise if the raster has no CRS or no geotransform.

    Returns
    -------
    RasterInfo
        CRS, transform, size, dtypes, nodata, block layout, overviews, compression.

    Raises
    ------
    NotGeoreferencedError
        If ``require_georef`` and georeferencing is missing.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        info = _info(src, path)
    if require_georef:
        _check_georef(info)
    return info


def _int_window(window: Window) -> Window:
    vals = (window.col_off, window.row_off, window.width, window.height)
    if any(float(v) != int(v) for v in vals):
        raise ValueError(f"window must have integer offsets and sizes, got {window}")
    col, row, w, h = (int(v) for v in vals)
    if w <= 0 or h <= 0:
        raise ValueError(f"window must be non-empty, got {window}")
    return Window(col, row, w, h)


def _inside(win: Window, width: int, height: int) -> bool:
    return bool(
        win.col_off >= 0
        and win.row_off >= 0
        and win.col_off + win.width <= width
        and win.row_off + win.height <= height
    )


class WindowRead(NamedTuple):
    """One window of a raster.

    Attributes
    ----------
    data
        ``(bands, h, w)`` array in the raster's dtype.
    valid
        ``(h, w)`` mask, False where every band is nodata or the pixel lies outside the
        raster (boundless reads).
    transform
        Affine transform of the window's top-left pixel.
    """

    data: npt.NDArray[Any]
    valid: BoolArray
    transform: Affine


class RasterReader:
    """Open a raster once and read many windows from it.

    Parameters
    ----------
    path
        Raster file.
    max_window_px
        Largest ``width * height`` a single read may return.

    Examples
    --------
    >>> with RasterReader("scene.tif") as rd:  # doctest: +SKIP
    ...     tile = rd.read(Window(0, 0, 512, 512))
    """

    def __init__(self, path: str | Path, max_window_px: int = DEFAULT_MAX_WINDOW_PX) -> None:
        self.path = Path(path)
        self.max_window_px = max_window_px
        self._src: DatasetReader | None = None
        self._info: RasterInfo | None = None

    def __enter__(self) -> Self:
        self._src = rasterio.open(self.path)
        self._info = _info(self._src, self.path)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._src is not None:
            self._src.close()
            self._src = None

    @property
    def info(self) -> RasterInfo:
        """Structure of the open raster."""
        if self._info is None:
            raise RuntimeError("RasterReader must be used as a context manager")
        return self._info

    def read(
        self,
        window: Window,
        bands: Sequence[int] | None = None,
        *,
        boundless: bool = False,
    ) -> WindowRead:
        """Read one window.

        Parameters
        ----------
        window
            Pixel window with integer offsets and sizes.
        bands
            1-based band indices; all bands by default.
        boundless
            Allow the window to extend past the raster; outside pixels are filled with
            nodata (or 0 when the raster has none) and marked invalid. Reflect padding
            is the tiler's job, not this reader's.

        Returns
        -------
        WindowRead
            Data, validity mask and window transform.

        Raises
        ------
        ValueError
            If the window is fractional, too large, or (when not boundless) outside
            the raster.
        """
        if self._src is None:
            raise RuntimeError("RasterReader must be used as a context manager")
        src = self._src
        win = _int_window(window)
        w, h = int(win.width), int(win.height)
        if w * h > self.max_window_px:
            raise ValueError(f"window {w}x{h} exceeds max_window_px={self.max_window_px}")
        inside = _inside(win, src.width, src.height)
        if not boundless and not inside:
            raise ValueError(f"window {win} lies outside the {src.width}x{src.height} raster")

        idx = list(bands) if bands is not None else list(range(1, src.count + 1))
        nodata = src.nodata
        if inside:
            data = src.read(idx, window=win)
        else:
            fill = nodata if nodata is not None else 0
            data = src.read(idx, window=win, boundless=True, fill_value=fill)

        if nodata is None:
            valid = np.ones((h, w), dtype=bool)
        elif np.isnan(nodata):
            valid = ~np.all(np.isnan(data), axis=0)
        else:
            valid = ~np.all(data == nodata, axis=0)
        if not inside:
            in_bounds = np.zeros((h, w), dtype=bool)
            col, row = int(win.col_off), int(win.row_off)
            r0, r1 = max(0, -row), min(h, src.height - row)
            c0, c1 = max(0, -col), min(w, src.width - col)
            if r1 > r0 and c1 > c0:
                in_bounds[r0:r1, c0:c1] = True
            valid &= in_bounds
        return WindowRead(data, valid, src.window_transform(win))


def read_window(
    path: str | Path,
    window: Window,
    bands: Sequence[int] | None = None,
    *,
    boundless: bool = False,
    max_window_px: int = DEFAULT_MAX_WINDOW_PX,
) -> WindowRead:
    """Read a single window from ``path`` (opens and closes the file).

    See Also
    --------
    RasterReader.read : same parameters; use the reader for many windows.
    """
    with RasterReader(path, max_window_px) as rd:
        return rd.read(window, bands, boundless=boundless)


def _gtiff_predictor(dtype: str, predictor: str) -> int:
    if predictor.upper() != "YES":
        return 1
    return 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2


class RasterWriter:
    """Write a raster window by window on the grid of a reference raster.

    The output gets the reference's CRS, transform and size. Data goes to ``<dst>.wip``;
    on a clean exit it is converted to a COG (or renamed) into ``dst``, and on an
    exception every temporary file is removed, so ``dst`` is either complete or absent.

    Parameters
    ----------
    dst
        Output path; parent directories are created.
    like
        Reference raster whose grid (CRS, transform, width, height) is copied.
    cfg
        I/O config (block size, compression, read-only roots).
    count, dtype
        Band count and dtype; default to the reference's.
    nodata
        Output nodata value (not inherited: a mask's 0 is data, not nodata).
    cog
        Convert to a COG on close; otherwise keep the tiled GeoTIFF.
    overview_resampling
        Override the COG overview resampling (``NEAREST`` for masks).
    """

    def __init__(
        self,
        dst: str | Path,
        like: RasterInfo,
        cfg: IOConfig,
        *,
        count: int | None = None,
        dtype: str | None = None,
        nodata: float | None = None,
        cog: bool = True,
        overview_resampling: str | None = None,
    ) -> None:
        self.dst = Path(dst)
        self.like = like
        self.cfg = cfg
        self.count = count if count is not None else like.count
        self.dtype = np.dtype(dtype if dtype is not None else like.dtypes[0]).name
        self.nodata = nodata
        self.cog = cog
        self.overview_resampling = overview_resampling
        self._wip = self.dst.with_name(self.dst.name + ".wip")
        self._ds: DatasetWriter | None = None

    def __enter__(self) -> Self:
        assert_writable(self.dst, self.cfg.read_only_roots)
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        bs = self.cfg.cog.blocksize
        self._ds = rasterio.open(
            self._wip,
            "w",
            driver="GTiff",
            width=self.like.width,
            height=self.like.height,
            count=self.count,
            dtype=self.dtype,
            crs=self.like.crs,
            transform=self.like.transform,
            nodata=self.nodata,
            tiled=True,
            blockxsize=bs,
            blockysize=bs,
            compress=self.cfg.cog.compress,
            predictor=_gtiff_predictor(self.dtype, self.cfg.cog.predictor),
            BIGTIFF="IF_SAFER",
        )
        return self

    def write(self, array: npt.NDArray[Any], window: Window) -> None:
        """Write ``array`` ``(bands, h, w)`` (or ``(h, w)`` for one band) at ``window``.

        Raises
        ------
        ValueError
            On dtype or shape mismatch, or a window outside the raster.
        """
        if self._ds is None:
            raise RuntimeError("RasterWriter must be used as a context manager")
        win = _int_window(window)
        if not _inside(win, self.like.width, self.like.height):
            raise ValueError(f"window {win} lies outside the output raster")
        arr = array[np.newaxis] if array.ndim == 2 else array
        expected = (self.count, int(win.height), int(win.width))
        if arr.shape != expected:
            raise ValueError(f"array shape {arr.shape} does not match {expected}")
        if arr.dtype != np.dtype(self.dtype):
            raise ValueError(f"array dtype {arr.dtype} does not match output dtype {self.dtype}")
        self._ds.write(arr, window=win)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None
        try:
            if exc_type is not None:
                return
            if self.cog:
                to_cog(
                    self._wip,
                    self.dst,
                    self.cfg,
                    overview_resampling=self.overview_resampling,
                )
            else:
                os.replace(self._wip, self.dst)
        finally:
            self._wip.unlink(missing_ok=True)
