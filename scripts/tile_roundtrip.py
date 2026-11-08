"""Tile -> identity "model" -> stitch round trip on a large scene (Gate 1 evidence).

Reads one band of a scene tile by tile, feeds each tile unchanged to the stitcher,
writes the stitched scene as a COG on the source grid, then compares it with the source
block by block. With an identity model every strategy must reproduce the source on
covered pixels; the run also reports tile counts, timings and peak anonymous memory
(memmap pages are file-backed and reclaimable, so they are reported separately).
The output is deleted after the comparison unless ``--keep`` is given.

Usage: uv run python scripts/tile_roundtrip.py [--scene d3] [--band 1] [--nodata 0]
           [--strategy hann] [--tile 512 --overlap 64] [--out PATH]
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.config import load_io_config
from geoinfer.io.raster import RasterReader, inspect_raster
from geoinfer.tiling.config import load_tiling_config
from geoinfer.tiling.stitcher import Stitcher
from geoinfer.tiling.tiler import read_tile, tile_grid

OUT_ROOT = Path("/mnt/d/geo/processed/sn6/roundtrip")


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _anon_mib() -> float:
    """Current anonymous (non file-backed) resident memory, from /proc."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def main() -> None:
    tc = load_tiling_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="d3", help="key under scenes: in configs/data.yaml")
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--nodata", type=float, default=None, help="treat this value as nodata")
    p.add_argument("--strategy", default=tc.strategy)
    p.add_argument("--tile", type=int, default=tc.tile)
    p.add_argument("--overlap", type=int, default=tc.overlap)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--keep", action="store_true", help="keep the stitched output")
    a = p.parse_args()
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    src = Path(dcfg["scenes"][a.scene])
    out = a.out or OUT_ROOT / f"{a.scene}_b{a.band}_{a.strategy}_t{a.tile}_o{a.overlap}.tif"
    io_cfg = load_io_config()
    info = inspect_raster(src)
    h, w = info.height, info.width
    specs = tile_grid(h, w, a.tile, a.overlap)
    print(f"{src.name}: {w} x {h} px, {len(specs)} tiles (T {a.tile}, O {a.overlap})")

    t0 = time.perf_counter()
    kept = 0
    peak_anon = _anon_mib()
    with (
        RasterReader(src, max_window_px=a.tile * a.tile) as r,
        Stitcher(
            h,
            w,
            a.tile,
            a.overlap,
            a.strategy,
            memmap_min_px=tc.memmap_min_px,
            scratch_dir=tc.scratch_dir,
        ) as st,
    ):
        print(f"accumulators: {'memmap in ' + str(tc.scratch_dir) if st.uses_memmap else 'RAM'}")
        for i, spec in enumerate(specs, 1):
            t = read_tile(r, spec, [a.band], tc.pad_mode)
            valid = t.valid if a.nodata is None else t.valid & (t.data[0] != a.nodata)
            if tc.skip_nodata and not valid.any():
                continue
            st.add(spec, t.data.astype(np.float32), valid)  # identity "model"
            kept += 1
            if i % 50 == 0:
                peak_anon = max(peak_anon, _anon_mib())
            if i % 250 == 0:
                print(f"  {i}/{len(specs)}  {time.perf_counter() - t0:.0f} s")
        t_tiles = time.perf_counter() - t0
        st.write(out, info, io_cfg, tc.out_nodata)
        peak_anon = max(peak_anon, _anon_mib())
        t_write = time.perf_counter() - t0 - t_tiles

    t1 = time.perf_counter()
    max_err, max_rel, n_cov, n_src_valid, n_missed = 0.0, 0.0, 0, 0, 0
    with RasterReader(src) as rs, RasterReader(out) as ro:
        for r0 in range(0, h, io_cfg.cog.blocksize):
            win = Window(0, r0, w, min(io_cfg.cog.blocksize, h - r0))
            s = rs.read(win, [a.band]).data[0].astype(np.float32)
            o = ro.read(win).data[0]
            sv = np.ones(s.shape, bool) if a.nodata is None else s != a.nodata
            cov = ~np.isnan(o)
            if cov.any():
                err = np.abs(o[cov] - s[cov])
                max_err = max(max_err, float(err.max()))
                max_rel = max(max_rel, float((err / np.maximum(np.abs(s[cov]), 1.0)).max()))
            n_cov += int(cov.sum())
            n_src_valid += int(sv.sum())
            n_missed += int((sv & ~cov).sum())
    t_cmp = time.perf_counter() - t1

    print(f"tiles inferred {kept}, skipped {len(specs) - kept} (no valid pixel)")
    print(f"tiling + stitching {t_tiles:.0f} s, write {t_write:.0f} s, compare {t_cmp:.0f} s")
    print(f"source valid {n_src_valid / (h * w):.1%}, stitched covered {n_cov / (h * w):.1%}")
    print(f"valid source pixels missed: {n_missed}")
    print(f"max error on covered pixels: abs {max_err:.3g}, relative {max_rel:.2e}")
    print(f"peak anon RSS {peak_anon:.0f} MiB (peak RSS incl. memmap pages {_rss_mib():.0f} MiB)")
    print(f"output {out.name}: {out.stat().st_size / 2**20:.0f} MiB", end="")
    if a.keep:
        print(f", kept at {out}")
    else:
        out.unlink()
        print(", deleted (use --keep to keep it)")


if __name__ == "__main__":
    main()
