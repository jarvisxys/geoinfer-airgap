"""Overlap tiling of large rasters with reflect padding (pipeline step 4).

The grid starts half an overlap outside the scene and steps by ``T - O``, so the
centre crops ``[O/2, T - O/2)`` of the tiles partition the scene exactly; at ``O = 0``
the tiles themselves partition it. Tiles crossing the scene edge are filled by
reflecting the scene (numpy ``reflect``: the edge pixel is not repeated) and the
padded pixels are marked invalid. Each tile is one windowed read of at most
``T x T`` pixels, so the scene is never loaded whole.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from rasterio.windows import Window

from geoinfer.io.raster import RasterReader
from geoinfer.tiling.config import PAD_MODES, check_grid

IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class TileSpec:
    """One tile of a scene grid.

    Attributes
    ----------
    row, col
        Grid indices.
    y0, x0
        Scene coordinates of the tile's top-left pixel; negative or past the scene
        edge for border tiles.
    tile
        Tile side ``T``.
    crop
        ``(start, stop)`` of the centre crop in tile coordinates, same on both axes.
    """

    row: int
    col: int
    y0: int
    x0: int
    tile: int
    crop: tuple[int, int]


class TileRead(NamedTuple):
    """Pixels of one tile.

    Attributes
    ----------
    data
        ``(bands, T, T)`` in the raster's dtype.
    valid
        ``(T, T)``; False on nodata and on padded pixels.
    spec
        Where the tile sits in the scene.
    """

    data: npt.NDArray[Any]
    valid: BoolArray
    spec: TileSpec


def tile_grid(height: int, width: int, tile: int, overlap: int) -> list[TileSpec]:
    """Tile specs covering a ``height x width`` scene, row-major.

    Parameters
    ----------
    height, width
        Scene size in pixels.
    tile, overlap
        ``T`` and ``O``; ``O`` must be even and ``0 <= O < T``.
    """
    check_grid(tile, overlap)
    if height <= 0 or width <= 0:
        raise ValueError(f"scene must be non-empty, got {height} x {width}")
    stride, half = tile - overlap, overlap // 2
    rows, cols = math.ceil(height / stride), math.ceil(width / stride)
    crop = (half, half + stride)
    return [
        TileSpec(r, c, r * stride - half, c * stride - half, tile, crop)
        for r in range(rows)
        for c in range(cols)
    ]


def reflect_index(idx: npt.NDArray[np.integer[Any]], n: int) -> IntArray:
    """Map possibly out-of-range indices into ``[0, n)`` by mirror reflection.

    Matches ``np.pad(..., mode="reflect")`` for any pad width; ``n == 1`` maps to 0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return np.zeros(idx.shape, dtype=np.intp)
    period = 2 * (n - 1)
    m = np.mod(idx, period).astype(np.intp)
    return np.asarray(np.where(m < n, m, period - m), dtype=np.intp)


def read_tile(
    reader: RasterReader,
    spec: TileSpec,
    bands: Sequence[int] | None = None,
    pad_mode: str = "reflect",
) -> TileRead:
    """Read one tile, padding the parts outside the scene.

    Parameters
    ----------
    reader
        Open scene reader.
    spec
        Tile to read.
    bands
        1-based bands; all by default.
    pad_mode
        ``reflect`` mirrors the scene; ``constant`` fills zeros.
    """
    if pad_mode not in PAD_MODES:
        raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}")
    h, w = reader.info.height, reader.info.width
    yy = np.arange(spec.y0, spec.y0 + spec.tile)
    xx = np.arange(spec.x0, spec.x0 + spec.tile)
    in_y, in_x = (yy >= 0) & (yy < h), (xx >= 0) & (xx < w)
    ry, rx = reflect_index(yy, h), reflect_index(xx, w)
    # one read of the bounding window of every scene pixel the tile needs (<= T x T)
    r0, r1, c0, c1 = int(ry.min()), int(ry.max()) + 1, int(rx.min()), int(rx.max()) + 1
    win = reader.read(Window(c0, r0, c1 - c0, r1 - r0), bands)
    data = win.data[:, (ry - r0)[:, None], (rx - c0)[None, :]]
    inside = in_y[:, None] & in_x[None, :]
    valid = win.valid[(ry - r0)[:, None], (rx - c0)[None, :]] & inside
    if pad_mode == "constant":
        data = np.where(inside, data, np.zeros((), dtype=data.dtype))
    return TileRead(data, valid, spec)


def iter_tiles(
    reader: RasterReader,
    tile: int,
    overlap: int,
    bands: Sequence[int] | None = None,
    pad_mode: str = "reflect",
    skip_nodata: bool = True,
) -> Iterator[TileRead]:
    """Yield the tiles of a scene in row-major order.

    Parameters
    ----------
    reader
        Open scene reader.
    tile, overlap
        ``T`` and ``O``.
    bands
        1-based bands; all by default.
    pad_mode
        See :func:`read_tile`.
    skip_nodata
        Drop tiles without any valid pixel.
    """
    for spec in tile_grid(reader.info.height, reader.info.width, tile, overlap):
        t = read_tile(reader, spec, bands, pad_mode)
        if skip_nodata and not t.valid.any():
            continue
        yield t
