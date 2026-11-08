"""Build D4: the held-out SAR + optical + labels scene from the val-split tiles.

D2 strips are narrow (about 900 m x 5 km) and each has its own sub-pixel grid, so D4
mosaics every val tile onto one 0.5 m grid (docs/decisions.md, D3):

- tiles are pasted at their nearest whole-pixel offset (nearest neighbour: values are
  unchanged, positions move by at most half a pixel per axis);
- priority: look direction 1 strips first, then direction 0, each in time order; the
  first valid SAR pixel wins and the optical twin is pasted at exactly the same pixels,
  so SAR / optical co-registration is kept;
- labels: building fragments from all val tiles are merged by Building_ID and
  rasterised on the D4 grid as an instance raster (0 = background);
- a strip-ID raster records which acquisition owns each pixel (0 = no data), so S1 can
  mask acquisition seams out of its seam metrics.

Outputs (COGs + buildings.geojson + d4.json) go to ``scenes.d4`` in data.yaml.

Usage: uv run python scripts/build_d4.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import yaml
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import unary_union

from geoinfer.io.config import load_io_config
from geoinfer.io.paths import assert_writable
from geoinfer.io.raster import RasterInfo, RasterReader, RasterWriter, inspect_raster
from geoinfer.tiling.config import load_tiling_config
from geoinfer.tiling.mosaic import paste_first_valid, paste_offset, snap_grid

TILE_CRS = "EPSG:32631"
RES = 0.5


def _by_id(folder: Path, rx: re.Pattern[str], suffix: str) -> dict[str, Path]:
    return {m.group(1): f for f in folder.glob(f"*{suffix}") if (m := rx.search(f.name))}


def _write(
    path: Path, like: RasterInfo, data: Any, nodata: float | None, resampling: str | None
) -> None:
    io_cfg = load_io_config()
    count = 1 if data.ndim == 2 else data.shape[0]
    step = io_cfg.cog.blocksize
    with RasterWriter(
        path,
        like,
        io_cfg,
        count=count,
        dtype=data.dtype.name,
        nodata=nodata,
        overview_resampling=resampling,
    ) as wr:
        for r0 in range(0, like.height, step):
            win = Window(0, r0, like.width, min(step, like.height - r0))
            block = data[..., r0 : r0 + win.height, :]
            wr.write(np.asarray(block if block.ndim == 3 else block[None]), win)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    io_cfg = load_io_config()
    out_dir = Path(cfg["scenes"]["d4"]).expanduser()
    assert_writable(out_dir, io_cfg.read_only_roots)
    interim = Path(cfg["interim_root"]).expanduser()
    tif_rx = re.compile(cfg["tile_id_regex"])
    geo_rx = re.compile(cfg["tile_id_regex"].replace(r"\.tif$", r"\.geojson$"))
    sar_idx = _by_id(interim / cfg["modalities"]["sar"], tif_rx, ".tif")
    opt_idx = _by_id(interim / cfg["modalities"]["optical"], tif_rx, ".tif")
    geo_idx = _by_id(interim / "geojson_buildings", geo_rx, ".geojson")
    orient_file = interim / cfg["orientations"]
    orient = dict(ln.split() for ln in orient_file.read_text().splitlines() if ln.strip())
    val = Path(cfg["splits"]["out_dir"], "val.txt").read_text().split()
    t0 = time.perf_counter()

    infos = {t: inspect_raster(sar_idx[t]) for t in val}
    for t in val:
        oi = inspect_raster(opt_idx[t])
        if not oi.transform.almost_equals(infos[t].transform, precision=1e-3):
            raise ValueError(f"{t}: optical twin is on another grid")
    bounds = np.array([box(*_bounds(infos[t])).bounds for t in val])
    transform, width, height = snap_grid(
        (bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()), RES
    )
    strips = sorted({t.split("_tile_")[0] for t in val}, key=lambda s: (orient[s] != "1", s))
    strip_no = {s: i + 1 for i, s in enumerate(strips)}  # 0 = no data
    order = sorted(val, key=lambda t: (strip_no[t.split("_tile_")[0]], t))
    print(f"{len(val)} val tiles from {len(strips)} strips -> {width} x {height} px grid")

    scratch = load_tiling_config().scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="d4-", dir=scratch))
    try:
        sar = np.memmap(tmp / "sar", np.float32, "w+", shape=(4, height, width))
        opt = np.memmap(tmp / "opt", np.uint16, "w+", shape=(4, height, width))
        owner = np.zeros((1, height, width), np.uint16)
        filled = [np.zeros((height, width), bool) for _ in range(3)]
        residual = 0.0
        used = 0
        for i, t in enumerate(order, 1):
            r, c, rr, rc = paste_offset(infos[t].transform, transform)
            residual = max(residual, abs(rr), abs(rc))
            with RasterReader(sar_idx[t]) as rs, RasterReader(opt_idx[t]) as ro:
                full = Window(0, 0, infos[t].width, infos[t].height)
                s = rs.read(full).data
                o = ro.read(full).data
            valid = np.all(s != 0, axis=0)
            new = paste_first_valid(sar, filled[0], s, valid, r, c)
            new_o = paste_first_valid(opt, filled[1], o, valid, r, c)
            sid = np.full((1, *valid.shape), strip_no[t.split("_tile_")[0]], np.uint16)
            paste_first_valid(owner, filled[2], sid, valid, r, c)
            if not np.array_equal(new, new_o):
                raise AssertionError(f"{t}: SAR and optical pasted at different pixels")
            used += bool(new.any())
            if i % 100 == 0:
                print(f"  {i}/{len(order)} tiles  {time.perf_counter() - t0:.0f} s")
        covered = filled[0]
        print(f"mosaic done: {covered.mean():.1%} covered, max residual {residual:.3f} px")

        parts: dict[int, list[Any]] = defaultdict(list)
        n_frag = 0
        for t in val:
            if t not in geo_idx:
                continue
            f = gpd.read_file(geo_idx[t])
            if not len(f):
                continue
            for bid, geom in zip(f.Building_ID, f.to_crs(TILE_CRS).geometry, strict=True):
                if geom is not None and not geom.is_empty:
                    parts[int(bid)].append(geom)
                    n_frag += 1
        ids = sorted(parts)
        buildings = gpd.GeoDataFrame(
            {"Building_ID": ids, "n_fragments": [len(parts[i]) for i in ids]},
            geometry=[unary_union(parts[i]) for i in ids],
            crs=TILE_CRS,
        )
        buildings["area_m2"] = buildings.geometry.area
        labels = rasterize(
            zip(buildings.geometry, buildings.Building_ID, strict=True),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint32",
        )
        labels[~covered] = 0
        print(f"labels: {len(buildings)} buildings from {n_frag} fragments")

        like = replace(
            infos[order[0]],
            transform=transform,
            width=width,
            height=height,
            path=out_dir / "d4_sar.tif",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / "d4_sar.tif", like, sar, 0.0, None)
        _write(out_dir / "d4_optical.tif", like, opt, 0, None)
        _write(out_dir / "d4_labels.tif", like, labels, None, "NEAREST")
        _write(out_dir / "d4_strips.tif", like, owner[0], 0, "NEAREST")
        buildings.to_file(out_dir / "d4_buildings.geojson", driver="GeoJSON")

        share = {
            o: float(np.isin(owner[0], [strip_no[s] for s in strips if orient[s] == o]).mean())
            for o in ("0", "1")
        }
        manifest = {
            "source": "val split tiles (data/splits/val.txt), SAR-Intensity + PS-RGBNIR",
            "grid": {
                "crs": TILE_CRS,
                "transform": list(transform)[:6],
                "width": width,
                "height": height,
                "res_m": RES,
            },
            "tiles": len(val),
            "tiles_contributing": used,
            "strips": {str(strip_no[s]): {"id": s, "orientation": int(orient[s])} for s in strips},
            "priority": "orientation 1 first, then 0; strips in time order; first valid wins",
            "covered_fraction": round(float(covered.mean()), 4),
            "pixel_share_by_orientation": {k: round(v, 4) for k, v in share.items()},
            "max_paste_residual_px": round(residual, 4),
            "buildings": len(buildings),
            "building_pixel_fraction_of_covered": round(
                float((labels > 0).sum() / max(covered.sum(), 1)), 4
            ),
            "files_mib": {
                f.name: round(f.stat().st_size / 2**20, 1) for f in sorted(out_dir.iterdir())
            },
        }
        (out_dir / "d4.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({k: v for k, v in manifest.items() if k != "strips"}, indent=2))
        print(f"done in {time.perf_counter() - t0:.0f} s -> {out_dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _bounds(info: RasterInfo) -> tuple[float, float, float, float]:
    t = info.transform
    return (t.c, t.f + t.e * info.height, t.c + t.a * info.width, t.f)


if __name__ == "__main__":
    main()
