"""Building masks to polygons (pipeline step 8, second half).

``rasterio.features.shapes`` traces pixel outlines (holes kept) in the raster's CRS;
Douglas-Peucker (``shapely.simplify`` with topology preserved) removes the pixel
staircase. Large masks are traced block by block: polygons that touch an inner block
edge are merged with ``unary_union`` and split back into parts. With 4-connectivity
this reproduces whole-array tracing exactly, because pixels of one component that sit
in neighbouring blocks share an edge, and polygons that only touch at a corner stay
separate parts. (8-connectivity would need corner contacts merged too, so the
block-wise path refuses it.)

GeoJSON is exported in ``geojson_crs`` (EPSG:4326 per RFC 7946) or, on request, in the
raster's own CRS.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from geoinfer.io.paths import assert_writable
from geoinfer.io.raster import RasterReader
from geoinfer.postprocess.config import PostprocessConfig

BoolArray = npt.NDArray[np.bool_]


def polygonize(mask: BoolArray, transform: Affine, connectivity: int) -> list[Polygon]:
    """Trace the True regions of ``mask`` as polygons in the raster's CRS.

    Parameters
    ----------
    mask
        ``(h, w)`` building mask.
    transform
        Affine transform of the mask's top-left pixel.
    connectivity
        4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if not mask.any():
        return []
    m = mask.astype(np.uint8)
    out = [
        shape(geom)
        for geom, _ in shapes(m, mask=mask, transform=transform, connectivity=connectivity)
    ]
    return _sorted([p for p in out if isinstance(p, Polygon)])


def _sorted(polys: Iterable[Polygon]) -> list[Polygon]:
    """North to south, then west to east (deterministic ids)."""
    return sorted(polys, key=lambda p: (-p.bounds[3], p.bounds[0], -p.area))


def _parts(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [] if geom.is_empty else [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _merge_blocks(
    blocks: Iterable[tuple[BoolArray, Window]], transform: Affine, height: int, width: int
) -> list[Polygon]:
    done: list[Polygon] = []
    edge: list[Polygon] = []
    eps = 1e-6 * max(abs(transform.a), abs(transform.e))
    for m, win in blocks:
        wt = window_transform(win, transform)
        polys = polygonize(m, wt, connectivity=4)
        if not polys:
            continue
        # world coordinates of this block's inner edges (scene edges are not seams)
        xs, ys = [], []
        if win.col_off > 0:
            xs.append(wt.c)
        if win.col_off + win.width < width:
            xs.append(wt.c + wt.a * win.width)
        if win.row_off > 0:
            ys.append(wt.f)
        if win.row_off + win.height < height:
            ys.append(wt.f + wt.e * win.height)
        for p in polys:
            minx, miny, maxx, maxy = p.bounds
            on_seam = any(abs(minx - x) < eps or abs(maxx - x) < eps for x in xs) or any(
                abs(miny - y) < eps or abs(maxy - y) < eps for y in ys
            )
            (edge if on_seam else done).append(p)
    if edge:
        done.extend(_parts(unary_union(edge)))
    return _sorted(done)


def _windows(height: int, width: int, block: int) -> list[Window]:
    return [
        Window(c0, r0, min(block, width - c0), min(block, height - r0))
        for r0 in range(0, height, block)
        for c0 in range(0, width, block)
    ]


def vectorize_mask_blockwise(
    mask: BoolArray, transform: Affine, block: int, connectivity: int = 4
) -> list[Polygon]:
    """Same polygons as ``polygonize(mask, transform, 4)``, traced in blocks."""
    if connectivity != 4:
        raise ValueError("block-wise vectorizing supports connectivity 4 only")
    h, w = mask.shape
    blocks = ((mask[win.toslices()], win) for win in _windows(h, w, block))
    return _merge_blocks(blocks, transform, h, w)


def simplify(polys: Sequence[Polygon], tolerance: float) -> list[Polygon]:
    """Douglas-Peucker with topology preserved; empty results are dropped."""
    out = []
    for p in polys:
        if p.is_empty:
            continue
        s = p.simplify(tolerance, preserve_topology=True) if tolerance > 0 else p
        out.extend(_parts(s))
    return out


def to_geodataframe(
    polys: Sequence[Polygon],
    crs: str | CRS,
    areas: Sequence[float] | None = None,
) -> gpd.GeoDataFrame:
    """Polygons with ``id`` (1..n) and ``area_m2`` in the given CRS.

    ``areas`` lets callers report pixel-exact areas from before simplification.
    """
    area = list(areas) if areas is not None else [p.area for p in polys]
    return gpd.GeoDataFrame(
        {"id": np.arange(1, len(polys) + 1), "area_m2": area},
        geometry=list(polys),
        crs=crs,
    )


def vectorize_raster(path: str | Path, cfg: PostprocessConfig, value: int = 1) -> gpd.GeoDataFrame:
    """Trace a mask raster (``value`` = building) block-wise into a GeoDataFrame.

    Nodata and other values are background. Areas are pixel-exact (before
    simplification); geometries are simplified with ``cfg.simplify_tolerance_m``.
    """
    if cfg.connectivity != 4:
        raise ValueError("block-wise vectorizing supports connectivity 4 only")
    with RasterReader(path) as r:
        info = r.info

        def blocks() -> Iterable[tuple[BoolArray, Window]]:
            for win in _windows(info.height, info.width, cfg.block):
                rd = r.read(win, [1])
                yield (rd.data[0] == value) & rd.valid, win

        polys = _merge_blocks(blocks(), info.transform, info.height, info.width)
    areas = [p.area for p in polys]
    simple = [p.simplify(cfg.simplify_tolerance_m, preserve_topology=True) for p in polys]
    if info.crs is None:
        raise ValueError(f"{path} has no CRS")
    return to_geodataframe(simple, info.crs, areas)


def export_geojson(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
    crs: str | None,
    read_only_roots: Sequence[Path] = (),
) -> Path:
    """Write ``gdf`` as GeoJSON, reprojected to ``crs`` (``None`` keeps its own CRS).

    The file is written to ``<path>.part`` and renamed, so it is complete or absent.
    """
    out = Path(path)
    assert_writable(out, tuple(read_only_roots))
    out.parent.mkdir(parents=True, exist_ok=True)
    data = gdf.to_crs(crs) if crs is not None else gdf
    part = out.with_name(out.name + ".part")
    try:
        data.to_file(part, driver="GeoJSON")
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)
    return out
