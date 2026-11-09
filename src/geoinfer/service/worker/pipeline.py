"""One segmentation job: AOI window -> tiles -> model -> stitch -> clean -> vectorize.

The job's AOI (EPSG:4326) is reprojected to the scene CRS and its bounding box, rounded
outwards to whole pixels and clipped to the scene, becomes the output grid. The tile grid
covers that window; tiles are read from the full scene, so context outside the AOI is
real imagery and only the scene edge is reflect-padded (as in S1). Input normalisation
matches training (``Normalizer``, global statistics). SAR and fused inputs from
look-direction-1 scenes are flipped north-south before the model and the prediction is
flipped back (decision D5). Probabilities are Hann-blended (S1), cleaned with the
module (d) chain at the model's tuned threshold, and traced into polygons.

Outputs in ``out_dir``: ``mask.tif`` (uint8 COG: 1 building, 0 background, 255 outside
the AOI or without data; same CRS as the scene, transform of the AOI window) and
``buildings.geojson`` (EPSG:4326, ``area_m2`` pixel-exact, ``score`` = mean probability).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import RasterReader, RasterWriter
from geoinfer.postprocess.config import PostprocessConfig
from geoinfer.postprocess.morphology import clean_mask_blockwise, label_components
from geoinfer.postprocess.vectorize import (
    export_geojson,
    simplify,
    to_geodataframe,
    vectorize_mask_blockwise,
)
from geoinfer.tiling.config import TilingConfig
from geoinfer.tiling.stitcher import Stitcher
from geoinfer.tiling.tiler import TileSpec, read_tile, tile_grid
from geoinfer.train.dataset import Normalizer

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
Predict = Callable[[FloatArray], FloatArray]

MASK_NODATA = 255
OUT_CRS = "EPSG:4326"


@dataclass(frozen=True)
class SceneInputs:
    """Rasters of one registered scene.

    Attributes
    ----------
    sensor
        ``optical``, ``sar`` or ``fused``.
    optical, sar
        COG paths; ``optical`` for optical and fused, ``sar`` for SAR and fused (on the
        optical grid).
    orientation
        SAR look direction (0 or 1); required for SAR and fused.
    """

    sensor: str
    optical: Path | None
    sar: Path | None
    orientation: int | None


@dataclass(frozen=True)
class SegmentResult:
    """What a job produced."""

    buildings: gpd.GeoDataFrame
    n_tiles: int
    timings: dict[str, float]
    mask_path: Path
    geojson_path: Path


def needs_flip(sensor: str, orientation: int | None) -> bool:
    """True when the model input must be flipped north-south (decision D5)."""
    if sensor == "optical":
        return False
    if orientation not in (0, 1):
        raise ValueError(f"{sensor} scenes need an orientation of 0 or 1, got {orientation}")
    return orientation == 1


def to_crs(geom: BaseGeometry, src: Any, dst: Any, segments: int = 64) -> Any:
    """Reproject a geometry; edges are densified first so they stay straight in ``src``."""
    tr = Transformer.from_crs(src, dst, always_xy=True)
    x0, y0, x1, y1 = geom.bounds
    step = max(x1 - x0, y1 - y0) / segments
    dense = geom.segmentize(step) if step > 0 else geom

    def fn(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.column_stack(tr.transform(xy[:, 0], xy[:, 1]))

    return shapely.transform(dense, fn)


def aoi_window(aoi: BaseGeometry, transform: Affine, width: int, height: int) -> Window:
    """Whole-pixel window of the AOI bounding box (scene CRS), clipped to the scene."""
    x0, y0, x1, y1 = aoi.bounds
    inv = ~transform
    cols, rows = zip(*(inv @ (x, y) for x in (x0, x1) for y in (y0, y1)), strict=True)
    c0, c1 = max(0, int(np.floor(min(cols)))), min(width, int(np.ceil(max(cols))))
    r0, r1 = max(0, int(np.floor(min(rows)))), min(height, int(np.ceil(max(rows))))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("the AOI lies outside the scene")
    return Window(c0, r0, c1 - c0, r1 - r0)


def prepare_tile(
    spec: TileSpec,
    scene: SceneInputs,
    readers: dict[str, RasterReader],
    norm: Normalizer,
    pad_mode: str,
    flip: bool,
) -> tuple[FloatArray, BoolArray] | None:
    """Model input and stitch mask of one tile (scene coordinates), None if empty."""
    parts: list[npt.NDArray[Any]] = []
    stitch_valid: BoolArray | None = None
    norm_valid: BoolArray | None = None
    opt = sar = sar_valid = None
    if scene.sensor in ("optical", "fused"):
        t = read_tile(readers["optical"], spec, None, pad_mode)
        opt = t.data
        # reflected padding is real imagery: normalise it, but never stitch it
        ov = np.any(opt != 0, axis=0)
        norm_valid, stitch_valid = ov, t.valid & ov
    if scene.sensor in ("sar", "fused"):
        s = read_tile(readers["sar"], spec, None, pad_mode)
        sar = s.data
        sar_valid = np.all(sar != 0, axis=0)
        sv = s.valid & sar_valid
        norm_valid = sar_valid if norm_valid is None else norm_valid & sar_valid
        stitch_valid = sv if stitch_valid is None else stitch_valid & sv
    assert stitch_valid is not None
    assert norm_valid is not None
    if not stitch_valid.any():
        return None
    if opt is not None:
        parts.append(norm.optical(opt, norm_valid))
    if sar is not None:
        assert sar_valid is not None
        parts.append(norm.sar_from_chain(norm.sar_chain(sar, sar_valid), norm_valid, None))
    x = np.concatenate(parts, axis=0)
    if flip:
        x = x[:, ::-1]
    return np.ascontiguousarray(x, dtype=np.float32), stitch_valid


def _scores(polys: list[Polygon], clean: BoolArray, prob: FloatArray, tr: Affine) -> list[float]:
    """Mean probability of each polygon's connected component (4-connectivity)."""
    labels, n = label_components(clean, 4)
    if n == 0:
        return []
    p = np.nan_to_num(prob, nan=0.0)
    sums = np.bincount(labels.ravel(), weights=p.ravel(), minlength=n + 1)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    inv = ~tr
    out = []
    for poly in polys:
        c, r = inv @ poly.representative_point().coords[0]
        lab = labels[min(int(r), clean.shape[0] - 1), min(int(c), clean.shape[1] - 1)]
        out.append(float(sums[lab] / counts[lab]) if lab > 0 else float("nan"))
    return out


def segment_aoi(
    scene: SceneInputs,
    aoi_wgs84: BaseGeometry,
    predictor: Predict,
    norm: Normalizer,
    threshold: float,
    tcfg: TilingConfig,
    pcfg: PostprocessConfig,
    io_cfg: IOConfig,
    out_dir: Path,
    *,
    batch_size: int,
    max_aoi_px: int,
) -> SegmentResult:
    """Run one building-segmentation job and write its outputs (see module docstring).

    Raises
    ------
    ValueError
        If the AOI misses the scene or exceeds ``max_aoi_px``, or a SAR/fused scene has
        no orientation.
    """
    t_start = time.perf_counter()
    flip = needs_flip(scene.sensor, scene.orientation)
    ref = scene.optical if scene.sensor in ("optical", "fused") else scene.sar
    assert ref is not None
    timings = dict.fromkeys(("prepare_s", "infer_s", "stitch_s"), 0.0)
    with ExitStack() as es:
        readers: dict[str, RasterReader] = {}
        if scene.optical is not None and scene.sensor != "sar":
            readers["optical"] = es.enter_context(RasterReader(scene.optical))
        if scene.sar is not None and scene.sensor != "optical":
            readers["sar"] = es.enter_context(RasterReader(scene.sar))
        info = readers["optical" if "optical" in readers else "sar"].info
        if info.crs is None:
            raise ValueError(f"{ref} has no CRS")
        aoi = to_crs(aoi_wgs84, OUT_CRS, info.crs)
        win = aoi_window(aoi, info.transform, info.width, info.height)
        h, w = int(win.height), int(win.width)
        if h * w > max_aoi_px:
            raise ValueError(f"AOI window {w} x {h} px exceeds the limit of {max_aoi_px} px")
        tr = window_transform(win, info.transform)
        in_aoi = geometry_mask([aoi], (h, w), tr, invert=True)
        st = es.enter_context(
            Stitcher(
                h,
                w,
                tcfg.tile,
                tcfg.overlap,
                tcfg.strategy,
                memmap_min_px=tcfg.memmap_min_px,
                scratch_dir=tcfg.scratch_dir,
            )
        )
        batch: list[tuple[TileSpec, FloatArray, BoolArray]] = []
        n_tiles = 0

        def flush() -> None:
            if not batch:
                return
            t0 = time.perf_counter()
            probs = predictor(np.stack([b[1] for b in batch]))
            t1 = time.perf_counter()
            for (spec, _, valid), p in zip(batch, probs, strict=True):
                st.add(spec, p[::-1] if flip else p, valid)
            timings["infer_s"] += t1 - t0
            timings["stitch_s"] += time.perf_counter() - t1
            batch.clear()

        for spec in tile_grid(h, w, tcfg.tile, tcfg.overlap):
            y0, x0 = max(0, spec.y0), max(0, spec.x0)
            if not in_aoi[y0 : spec.y0 + spec.tile, x0 : spec.x0 + spec.tile].any():
                continue
            t0 = time.perf_counter()
            g = replace(spec, y0=spec.y0 + int(win.row_off), x0=spec.x0 + int(win.col_off))
            prep = prepare_tile(g, scene, readers, norm, tcfg.pad_mode, flip)
            timings["prepare_s"] += time.perf_counter() - t0
            if prep is None:
                continue
            batch.append((spec, *prep))
            n_tiles += 1
            if len(batch) >= batch_size:
                flush()
        flush()
        t0 = time.perf_counter()
        prob = st.read(Window(0, 0, w, h))[0]
        covered = np.isfinite(prob) & in_aoi
        clean = clean_mask_blockwise(prob, covered, replace(pcfg, threshold=threshold), tr)
        raw = vectorize_mask_blockwise(clean, tr, pcfg.block, 4)
        scores = _scores(raw, clean, prob, tr)
        polys = simplify(raw, pcfg.simplify_tolerance_m)
        if len(polys) != len(raw):  # simplify split or dropped a part: keep the raw ones
            polys = raw
        gdf = to_geodataframe(polys, info.crs, [p.area for p in raw])
        gdf["score"] = scores
        timings["postprocess_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = np.where(covered, clean.astype(np.uint8), np.uint8(MASK_NODATA)).astype(np.uint8)
        like = replace(info, transform=tr, width=w, height=h, count=1, dtypes=("uint8",))
        mask_path = out_dir / "mask.tif"
        with RasterWriter(
            mask_path,
            like,
            io_cfg,
            count=1,
            dtype="uint8",
            nodata=MASK_NODATA,
            overview_resampling="NEAREST",
        ) as wr:
            wr.write(mask, Window(0, 0, w, h))
        out = gdf.to_crs(OUT_CRS)
        geojson_path = export_geojson(
            out, out_dir / "buildings.geojson", None, io_cfg.read_only_roots
        )
        timings["write_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_start
    return SegmentResult(
        out, n_tiles, {k: round(v, 3) for k, v in timings.items()}, mask_path, geojson_path
    )
