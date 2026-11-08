"""Quick-look figure of D4: overview (SAR, optical, strips) and a full-resolution crop
with building outlines on SAR and optical. Reads overviews / one window only.

Usage: uv run python scripts/d4_quicklook.py [--out docs/figures/d4_overview.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from rasterio.windows import Window, bounds

from geoinfer.io.raster import RasterReader

DOWN = 8  # overview factor for the scene panels


def _stretch(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    return np.where(valid, np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1), 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/figures/d4_overview.png"))
    p.add_argument("--row", type=int, default=7400, help="crop top row")
    p.add_argument("--col", type=int, default=1000, help="crop left column")
    p.add_argument("--size", type=int, default=400)
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    d4 = Path(cfg["scenes"]["d4"])
    with rasterio.open(d4 / "d4_sar.tif") as s:
        shape = (s.height // DOWN, s.width // DOWN)
        sar = s.read(out_shape=(4, *shape)).astype(np.float64)
    with rasterio.open(d4 / "d4_optical.tif") as o:
        opt = o.read([1, 2, 3], out_shape=(3, *shape)).astype(np.float64)
    with rasterio.open(d4 / "d4_strips.tif") as st:
        strips = st.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.nearest)
    sv = np.all(sar != 0, axis=0)

    win = Window(a.col, a.row, a.size, a.size)
    with RasterReader(d4 / "d4_sar.tif") as r:
        sar_c = r.read(win).data.astype(np.float64)
        tr = r.info.transform
    with RasterReader(d4 / "d4_optical.tif") as r:
        opt_c = r.read(win, [1, 2, 3]).data.astype(np.float64)
    b = bounds(win, tr)
    bld = gpd.read_file(d4 / "d4_buildings.geojson", bbox=b)

    fig, ax = plt.subplots(1, 5, figsize=(16, 9), dpi=80, width_ratios=[1, 1, 1, 3, 3])
    ax[0].imshow(_stretch(sar.mean(0), sv), cmap="gray")
    ax[0].set_title("SAR dB (band mean)")
    rgb = np.stack([_stretch(c, sv) for c in opt], axis=-1)
    ax[1].imshow(rgb)
    ax[1].set_title("optical RGB")
    sm = np.ma.masked_equal(strips, 0)
    ax[2].imshow(sm % 20, cmap="tab20", interpolation="nearest")
    ax[2].set_title(f"owner strip ({len(np.unique(strips)) - 1})")
    for x in ax[:3]:
        x.add_patch(
            plt.Rectangle(
                (a.col / DOWN, a.row / DOWN), a.size / DOWN, a.size / DOWN, fill=False, ec="red"
            )
        )
    ext = (b[0], b[2], b[1], b[3])
    cv = np.all(sar_c != 0, axis=0)
    ax[3].imshow(_stretch(sar_c.mean(0), cv), cmap="gray", extent=ext)
    ax[4].imshow(np.stack([_stretch(c, cv) for c in opt_c], -1), extent=ext)
    for x, name in ((ax[3], "SAR"), (ax[4], "optical")):
        bld.boundary.plot(ax=x, color="yellow", linewidth=0.7)
        x.set_xlim(b[0], b[2])
        x.set_ylim(b[1], b[3])
        x.set_title(f"{name} {a.size} px crop + labels (0.5 m)")
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle("D4: val-tile SAR + optical mosaic (2621 x 15451 px), labels from tile GeoJSONs")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"figure: {a.out}")


if __name__ == "__main__":
    main()
