"""Build the D3-val scene: the PS-RGBNIR TrainVal mosaic (D3) cut to the official Val
extent, with the official All-Val building labels (decision D1: scene-level optical
evaluation on the official Val extent, which has no D2 tiles).

Writes, in the same layout as D4 so the S1 harness takes either scene, to ``scenes.d3val``:
``d3val_optical.tif`` (COG), ``d3val_strips.tif`` (1 = inside the extent with optical
data, 0 = no data), ``d3val_labels.tif`` (Building_ID instances, 0 = background),
``d3val_buildings.geojson`` (EPSG:32631) and ``d3val.json``.

Usage: uv run python scripts/build_d3val.py
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import yaml
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

from geoinfer.io.config import load_io_config
from geoinfer.io.paths import assert_writable
from geoinfer.io.raster import RasterReader, RasterWriter, inspect_raster

TILE_CRS = "EPSG:32631"


def main() -> None:
    t0 = time.perf_counter()
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    raw = Path(dcfg["raw_root"])
    d3 = Path(dcfg["scenes"]["d3"])
    out = Path(dcfg["scenes"]["d3val"])
    io_cfg = load_io_config()
    assert_writable(out, io_cfg.read_only_roots)
    out.mkdir(parents=True, exist_ok=True)

    extent = gpd.read_file(raw / dcfg["splits"]["val_scene_extent"]).to_crs(TILE_CRS)
    poly = extent.union_all()
    info = inspect_raster(d3)
    win = from_bounds(*poly.bounds, transform=info.transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, info.width, info.height))
    w, h = int(win.width), int(win.height)
    tr = window_transform(win, info.transform)
    like = replace(info, transform=tr, width=w, height=h, path=out / "d3val_optical.tif")
    print(f"D3 window {win} -> {w} x {h} px")

    inside = rasterize([(poly, 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8") > 0
    valid = np.zeros((h, w), dtype=bool)
    step = io_cfg.cog.blocksize
    with (
        RasterReader(d3) as r,
        RasterWriter(
            out / "d3val_optical.tif", like, io_cfg, count=4, dtype="uint16", nodata=0
        ) as wr,
    ):
        for r0 in range(0, h, step):
            n = min(step, h - r0)
            src = Window(int(win.col_off), int(win.row_off) + r0, w, n)
            data = r.read(src).data
            v = np.any(data != 0, axis=0) & inside[r0 : r0 + n]
            data[:, ~v] = 0
            valid[r0 : r0 + n] = v
            wr.write(data, Window(0, r0, w, n))

    bld = gpd.read_file(raw / dcfg["labels"]["val_buildings"]).to_crs(TILE_CRS)
    bld = bld[bld.geometry.notna() & ~bld.geometry.is_empty & bld.intersects(poly)]
    bld = bld[["Building_ID", "geometry"]].reset_index(drop=True)
    labels = rasterize(
        zip(bld.geometry, bld.Building_ID, strict=True),
        out_shape=(h, w),
        transform=tr,
        fill=0,
        dtype="uint32",
    )
    labels[~valid] = 0
    for name, arr, nod, res in (
        ("d3val_strips.tif", valid.astype(np.uint8), 0, "NEAREST"),
        ("d3val_labels.tif", labels, None, "NEAREST"),
    ):
        with RasterWriter(
            out / name,
            like,
            io_cfg,
            count=1,
            dtype=arr.dtype.name,
            nodata=nod,
            overview_resampling=res,
        ) as wr:
            for r0 in range(0, h, step):
                n = min(step, h - r0)
                wr.write(arr[None, r0 : r0 + n], Window(0, r0, w, n))
    bld.to_file(out / "d3val_buildings.geojson", driver="GeoJSON")
    manifest = {
        "source": f"{d3.name} cut to {dcfg['splits']['val_scene_extent']}",
        "labels": dcfg["labels"]["val_buildings"],
        "grid": {"crs": TILE_CRS, "transform": list(tr)[:6], "width": w, "height": h},
        "valid_fraction": round(float(valid.mean()), 4),
        "buildings": len(bld),
        "building_pixel_fraction_of_valid": round(float((labels > 0).sum() / valid.sum()), 4),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "d3val.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
