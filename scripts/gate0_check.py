"""Gate 0: open one SpaceNet 6 SAR tile and its optical twin, print metadata, save a quick-look.

Usage: uv run python scripts/gate0_check.py [--sample ~/geo-data/sample] [--out docs/figures]
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio

from geoinfer.io.quicklook import render_quicklook

TILE_ID = re.compile(r"_(\d{14}_\d{14}_tile_\d+)\.tif$")


def find_pair(root: Path) -> tuple[Path, Path]:
    """Return the first (SAR-Intensity, PS-RGBNIR) pair that shares a tile ID."""
    sar = {TILE_ID.search(p.name).group(1): p for p in root.rglob("SAR-Intensity/*.tif")}  # type: ignore[union-attr]
    for opt in sorted(root.rglob("PS-RGBNIR/*.tif")):
        m = TILE_ID.search(opt.name)
        if m and m.group(1) in sar:
            return sar[m.group(1)], opt
    raise FileNotFoundError(f"no SAR/optical pair under {root}")


def describe(path: Path) -> None:
    with rasterio.open(path) as src:
        print(f"  {path.name}")
        print(f"    crs={src.crs}  size={src.width}x{src.height}  bands={src.count}")
        print(f"    dtype={src.dtypes[0]}  nodata={src.nodata}  res={src.res}")
        print(f"    bounds={tuple(round(b, 1) for b in src.bounds)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=Path, default=Path.home() / "geo-data" / "sample")
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = p.parse_args()

    sar, opt = find_pair(a.sample)
    print("SAR / optical pair:")
    describe(sar)
    describe(opt)

    with rasterio.open(sar) as s, rasterio.open(opt) as o:
        same_grid = s.crs == o.crs and s.transform.almost_equals(o.transform)
    print(f"same CRS + geotransform: {same_grid}")

    with tempfile.TemporaryDirectory() as tmp:
        sar_img = render_quicklook(sar, Path(tmp) / "sar.png", [1], max_size=450, sar_db=True)
        opt_img = render_quicklook(opt, Path(tmp) / "opt.png", [1, 2, 3], max_size=450)
    side = np.hstack([cv2.cvtColor(sar_img, cv2.COLOR_GRAY2RGB), opt_img])
    a.out.mkdir(parents=True, exist_ok=True)
    combined = a.out / "gate0_sar_optical.png"
    cv2.imwrite(str(combined), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    print(f"quick-look written: {combined}")


if __name__ == "__main__":
    main()
