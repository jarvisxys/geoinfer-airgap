"""Imagery helpers for the web UI: map tiles, coordinate conversion, image checks, prep.

Everything reads rasters through windows or decimated ``out_shape`` reads, never whole
scenes, and nothing touches the network.

- :func:`pyramid` / :func:`tile_png`: the scene itself is the base map, served as 256 px
  PNG tiles of a pixel-space pyramid (level ``max_zoom`` = full resolution, each level
  below halves it). Optical scenes show bands 1-3 (R, G, B), SAR scenes band 1, each
  stretched to the scene's 2-98 % range; nodata and outside-scene pixels are transparent.
- :func:`px_to_lonlat` / :func:`lonlat_to_px`: scene pixel <-> EPSG:4326.
- :func:`inspect_image`: checks a GeoTIFF against what the models need and explains
  each problem in plain words, saying whether :func:`prepare_image` can fix it.
- :func:`prepare_image`: reprojects to the image's UTM zone at 0.5 m and writes a
  Cloud-Optimised GeoTIFF, block by block.
- :func:`overlay_rings`: job polygons (EPSG:4326) in scene pixel coordinates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyproj
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely.geometry import shape

from geoinfer.io.config import IOConfig
from geoinfer.io.quicklook import read_decimated
from geoinfer.io.raster import RasterWriter, inspect_raster

TILE = 256
TARGET_GSD_M = 0.5
Stretch = list[tuple[float, float]]


def pyramid(width: int, height: int, tile: int = TILE) -> dict[str, int]:
    """Levels of the pixel pyramid: ``max_zoom`` is full resolution."""
    longest = max(width, height)
    max_zoom = max(0, math.ceil(math.log2(longest / tile))) if longest > tile else 0
    return {"tile": tile, "max_zoom": int(max_zoom), "width": width, "height": height}


def _layer_bands(layer: str) -> list[int]:
    return [1] if layer == "sar" else [1, 2, 3]


def stretch_stats(path: str | Path, layer: str) -> Stretch:
    """2-98 % range per displayed band, from a decimated read (longest side 1024 px)."""
    data, valid = read_decimated(path, _layer_bands(layer), max_size=1024)
    out = []
    for b in data:
        v = b[valid & (b != 0)]
        if v.size == 0:
            out.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(v, [2, 98])
        out.append((float(lo), float(hi) if hi > lo else float(lo) + 1.0))
    return out


def _empty_png(tile: int) -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((tile, tile, 4), np.uint8))
    assert ok
    return bytes(buf)


def tile_png(
    path: str | Path,
    z: int,
    x: int,
    y: int,
    layer: str,
    stretch: Stretch,
    tile: int = TILE,
) -> bytes:
    """One RGBA PNG tile of the scene pyramid (see module docstring)."""
    with rasterio.open(path) as src:
        p = pyramid(src.width, src.height, tile)
        if not 0 <= z <= p["max_zoom"]:
            raise ValueError(f"zoom must be 0..{p['max_zoom']}")
        scale = 2 ** (p["max_zoom"] - z)
        c0, r0 = x * tile * scale, y * tile * scale
        c1, r1 = min(src.width, c0 + tile * scale), min(src.height, r0 + tile * scale)
        if x < 0 or y < 0 or c1 <= c0 or r1 <= r0:
            return _empty_png(tile)
        oh, ow = math.ceil((r1 - r0) / scale), math.ceil((c1 - c0) / scale)
        bands = _layer_bands(layer)
        data = src.read(
            bands,
            window=Window(c0, r0, c1 - c0, r1 - r0),
            out_shape=(len(bands), oh, ow),
            resampling=Resampling.average if scale > 1 else Resampling.nearest,
        ).astype(np.float32)
        nodata = src.nodata
    valid = np.any(data != 0, axis=0)
    if nodata is not None and nodata != 0:
        valid &= np.any(data != nodata, axis=0)
    chans = []
    for b, (lo, hi) in zip(data, stretch, strict=False):
        chans.append(np.clip((b - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8))
    if len(chans) == 1:
        chans = chans * 3
    rgba = np.zeros((tile, tile, 4), np.uint8)
    rgba[:oh, :ow, 0], rgba[:oh, :ow, 1], rgba[:oh, :ow, 2] = chans[2], chans[1], chans[0]  # BGR
    rgba[:oh, :ow, 3] = np.where(valid, 255, 0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", rgba)
    assert ok
    return bytes(buf)


def px_to_lonlat(
    ring: Sequence[Sequence[float]], transform: Affine, crs: Any
) -> list[tuple[float, float]]:
    """Scene pixel (col, row) -> (lon, lat)."""
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xy = [transform @ (float(c), float(r)) for c, r in ring]
    lon, lat = tr.transform([p[0] for p in xy], [p[1] for p in xy])
    return list(zip(map(float, lon), map(float, lat), strict=True))


def lonlat_to_px(
    ring: Sequence[Sequence[float]], transform: Affine, crs: Any
) -> list[tuple[float, float]]:
    """(lon, lat) -> scene pixel (col, row)."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform([float(p[0]) for p in ring], [float(p[1]) for p in ring])
    inv = ~transform
    return [tuple(map(float, inv @ (a, b))) for a, b in zip(x, y, strict=True)]  # type: ignore[misc]


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS 84 / UTM zone containing (lon, lat)."""
    zone = min(60, int((lon + 180.0) // 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def _check(cid: str, level: str, message: str, fixable: bool = False) -> dict[str, Any]:
    return {"id": cid, "level": level, "message": message, "fixable": fixable}


def inspect_image(path: str | Path, train_optical: dict[str, Any] | None) -> dict[str, Any]:
    """Check a GeoTIFF against what the models need (see module docstring).

    Levels: ``pass``, ``info`` (advice), ``warn`` (works, expect lower accuracy or a slow
    job) and ``fail`` (a job would fail or be meaningless). ``ok`` is true without fails.
    """
    checks: list[dict[str, Any]] = []
    try:
        src = rasterio.open(path)
    except Exception as e:
        return {"ok": False, "checks": [_check("format", "fail", f"not a readable GeoTIFF: {e}")]}
    with src:
        w, h, count = src.width, src.height, src.count
        dtype = src.dtypes[0]
        crs = src.crs
        res = (abs(src.transform.a), abs(src.transform.e))
        tiled = src.profile.get("tiled", False)
        overviews = len(src.overviews(1)) if count else 0
        bounds = src.bounds
    out: dict[str, Any] = {
        "width": w,
        "height": h,
        "bands": count,
        "dtype": dtype,
        "crs": crs.to_string() if crs else None,
    }
    # georeferencing
    lonlat_bounds = None
    if crs is None:
        checks.append(
            _check(
                "crs",
                "fail",
                "No georeferencing: the image cannot be placed on "
                "the map. Export it again as a GeoTIFF with a CRS.",
            )
        )
        gsd = None
    elif not crs.is_projected:
        checks.append(
            _check(
                "crs",
                "fail",
                f"Coordinates are in degrees ({crs.to_string()}). "
                "'Prepare' reprojects it to metres (UTM).",
                True,
            )
        )
        lat = (bounds.bottom + bounds.top) / 2
        gsd = res[0] * 111_320 * math.cos(math.radians(lat))
    else:
        checks.append(_check("crs", "pass", f"Projected CRS {crs.to_string()}."))
        gsd = res[0]
    if crs is not None:
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = tr.transform([bounds.left, bounds.right], [bounds.bottom, bounds.top])
        lonlat_bounds = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    out["gsd_m"] = round(gsd, 3) if gsd is not None else None
    out["bounds_lonlat"] = lonlat_bounds
    # pixel size
    if gsd is not None:
        if abs(gsd - TARGET_GSD_M) / TARGET_GSD_M <= 0.1:
            checks.append(_check("gsd", "pass", f"Pixel size {gsd:.2f} m (models use 0.5 m)."))
        elif gsd > 5:
            checks.append(
                _check(
                    "gsd",
                    "fail",
                    f"Pixel size {gsd:.1f} m: a house is a few pixels "
                    "at most. The models need about 0.5 m imagery; "
                    "resampling cannot add the missing detail.",
                )
            )
        else:
            checks.append(
                _check(
                    "gsd",
                    "warn",
                    f"Pixel size {gsd:.2f} m; the models were trained "
                    "at 0.5 m. 'Prepare' resamples it.",
                    True,
                )
            )
    # bands and data type
    if count == 4:
        checks.append(_check("bands", "pass", "4 bands."))
    else:
        checks.append(
            _check(
                "bands",
                "fail",
                f"The models need 4 bands: R, G, B, NIR (optical) "
                f"or HH, HV, VH, VV (SAR). This file has {count}.",
            )
        )
    if dtype == "uint16":
        suggested = "optical" if count == 4 else None
        checks.append(_check("dtype", "pass", "16-bit integers (optical)."))
    elif dtype in ("float32", "float64"):
        suggested = "sar" if count == 4 else None
        checks.append(_check("dtype", "info", "Floating point: treated as SAR in dB."))
    else:
        suggested = "optical" if count == 4 else None
        checks.append(
            _check(
                "dtype",
                "warn",
                f"{dtype} data; the optical model was trained on "
                "16-bit WorldView-2. Expect lower accuracy.",
            )
        )
    # brightness against the training data (optical)
    if suggested == "optical" and train_optical and count == 4:
        data, valid = read_decimated(path, [1, 2, 3], max_size=512)
        v = data[:, valid]
        if v.size:
            med = float(np.median(v))
            lo, hi = float(np.mean(train_optical["p_lo"])), float(np.mean(train_optical["p_hi"]))
            if lo <= med <= hi:
                checks.append(_check("values", "pass", "Brightness in the training range."))
            else:
                checks.append(
                    _check(
                        "values",
                        "warn",
                        f"Median brightness {med:.0f} is outside "
                        f"the training range ({lo:.0f}-{hi:.0f}); "
                        "results may be poor.",
                    )
                )
    # layout
    if tiled and overviews:
        checks.append(_check("cog", "pass", "Tiled with overviews (fast to read)."))
    else:
        checks.append(
            _check(
                "cog",
                "info",
                "Not a Cloud-Optimised GeoTIFF: maps and jobs read "
                "it more slowly. 'Prepare' converts it.",
                True,
            )
        )
    out["checks"] = checks
    out["ok"] = not any(c["level"] == "fail" for c in checks)
    out["fixable"] = any(c["fixable"] and c["level"] in ("fail", "warn", "info") for c in checks)
    out["suggested_sensor"] = suggested
    return out


def prepare_image(
    src_path: str | Path, dst_path: str | Path, gsd: float, io_cfg: IOConfig, block: int = 1024
) -> Path:
    """Reproject to the image's UTM zone at ``gsd`` metres and write a COG (block-wise)."""
    info = inspect_raster(src_path)
    if info.crs is None:
        raise ValueError("the image has no CRS")
    with rasterio.open(src_path) as src:
        tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        b = src.bounds
        lon, lat = tr.transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)
        dst_crs = pyproj.CRS.from_epsg(utm_epsg(lon, lat))
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *b, resolution=gsd
        )
        nodata = src.nodata if src.nodata is not None else 0
        like = replace(
            info,
            crs=rasterio.crs.CRS.from_epsg(dst_crs.to_epsg()),
            transform=transform,
            width=int(width),
            height=int(height),
        )
        with (
            WarpedVRT(
                src,
                crs=dst_crs.to_wkt(),
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                nodata=nodata,
            ) as vrt,
            RasterWriter(dst_path, like, io_cfg, nodata=nodata) as out,
        ):
            for y0 in range(0, int(height), block):
                hh = min(block, int(height) - y0)
                win = Window(0, y0, int(width), hh)
                out.write(vrt.read(window=win).astype(info.dtypes[0]), win)
    return Path(dst_path)


_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def safe_upload_name(name: str) -> str:
    """A safe inbox file name: no folders, GeoTIFF suffix, plain characters."""
    base = re.split(r"[\\/]", name)[-1]
    low = base.lower()
    ext = next((e for e in (".tiff", ".tif") if low.endswith(e)), None)
    if ext is None:
        raise ValueError("only GeoTIFF files (.tif, .tiff) can be uploaded")
    stem = _BAD.sub("_", base[: -len(ext)]).strip("._")
    stem = re.sub(r"_+", "_", stem)
    if not stem:
        raise ValueError("the file name is empty")
    return f"{stem}.tif"


def overlay_rings(
    features: Sequence[dict[str, Any]], transform: Affine, crs: Any
) -> list[dict[str, Any]]:
    """Building polygons (GeoJSON features, EPSG:4326) as scene pixel rings."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = ~transform
    out = []
    for f in features:
        geom = shape(f["geometry"])
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        props = f.get("properties") or {}
        for p in polys:
            rings = []
            for ring in [p.exterior, *p.interiors]:
                coords = np.asarray(ring.coords, dtype=np.float64)
                xs, ys = tr.transform(coords[:, 0], coords[:, 1])
                pts = [inv @ (a, b) for a, b in zip(xs, ys, strict=True)]
                rings.append([[round(c, 2), round(r, 2)] for c, r in pts])
            out.append(
                {
                    "id": props.get("id"),
                    "area_m2": props.get("area_m2"),
                    "score": props.get("score"),
                    "rings": rings,
                }
            )
    return out
