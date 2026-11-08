"""Vectorize a building mask COG (1 = building) into GeoJSON polygons.

Block-wise tracing (``postprocess.vectorize.vectorize_raster``), Douglas-Peucker
simplification, export in ``geojson_crs`` from configs/postprocess.yaml (EPSG:4326,
RFC 7946) or, with ``--native-crs``, in the mask's own CRS (handy for QGIS checks).

Usage: uv run python scripts/vectorize_mask.py MASK.tif OUT.geojson [--native-crs]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from geoinfer.io.config import load_io_config
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.postprocess.vectorize import export_geojson, vectorize_raster


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mask", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--native-crs", action="store_true", help="keep the mask's CRS")
    a = p.parse_args()
    cfg, io_cfg = load_postprocess_config(), load_io_config()
    t0 = time.perf_counter()
    gdf = vectorize_raster(a.mask, cfg)
    crs = None if a.native_crs else cfg.geojson_crs
    export_geojson(gdf, a.out, crs, io_cfg.read_only_roots)
    print(
        f"{len(gdf)} polygons, {gdf.area_m2.sum():.0f} m2, "
        f"CRS {crs or gdf.crs.to_string()}, {time.perf_counter() - t0:.1f} s -> {a.out}"
    )


if __name__ == "__main__":
    main()
