"""Build geographic train / tune / val splits from the extracted SN6 train tiles.

Reads tile headers only (no pixels), writes data/splits/{train,tune,val}.txt and
splits.json, and a split map figure.

Usage: uv run python scripts/build_splits.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import yaml
from shapely.geometry.base import BaseGeometry

from geoinfer.train.splits import assign_splits, split_regions, tile_footprints, write_splits

TILE_CRS = "EPSG:32631"
COLORS = {"train": "tab:blue", "tune": "tab:orange", "val": "tab:green"}


def _extent(path: Path) -> BaseGeometry:
    return gpd.read_file(path).to_crs(TILE_CRS).union_all()


def _strip(tile_id: str) -> str:
    return tile_id.split("_tile_")[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--figure", type=Path, default=Path("docs/figures/splits_map.png"))
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    sc = cfg["splits"]
    raw = Path(cfg["raw_root"])
    interim = Path(cfg["interim_root"]).expanduser()

    sar = sorted((interim / cfg["modalities"]["sar"]).glob("*.tif"))
    opt_dir = interim / cfg["modalities"]["optical"]
    rx = re.compile(cfg["tile_id_regex"])
    opt_ids = {m.group(1) for f in opt_dir.glob("*.tif") if (m := rx.search(f.name))}
    fps = tile_footprints(sar, cfg["tile_id_regex"], TILE_CRS)
    missing_opt = sorted(set(fps) - opt_ids)
    print(f"SAR tiles: {len(fps)}  without optical twin: {len(missing_opt)}")
    fps = {k: v for k, v in fps.items() if k in opt_ids}

    train_ext, val_ext = _extent(raw / sc["train_extent"]), _extent(raw / sc["val_extent"])
    regions = split_regions(train_ext, val_ext, sc["tune_strip_fraction"])
    assigned, dropped = assign_splits(fps, regions, sc["tolerance_m"])

    test_path = raw / "extents_masks/SN6_AOI_11_Rotterdam_Test-Extent.geojson"
    if test_path.exists():
        test_ext = _extent(test_path)
        n_test = sum(fp.intersects(test_ext) for fp in fps.values())
        print(f"tiles touching the Test extent: {n_test}")

    meta = {
        "rule": sc["rule"],
        "tolerance_m": sc["tolerance_m"],
        "tune_strip_fraction": sc["tune_strip_fraction"],
        "source": f"{cfg['modalities']['sar']} tile headers, tiles with an optical twin",
        "tiles_considered": len(fps),
        "regions_bounds": {k: [round(v, 1) for v in r.bounds] for k, r in regions.items()},
    }
    info = write_splits(assigned, dropped, Path(sc["out_dir"]), meta)
    for name, ids in assigned.items():
        strips = Counter(_strip(t) for t in ids)
        print(f"{name:>6}: {len(ids):5d} tiles from {len(strips):3d} strips")
    print(f"dropped: {len(dropped)} (straddle a boundary or outside all regions)")
    print(f"wrote {sc['out_dir']}/ ({', '.join(info['sha256'])})")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    for name, reg in regions.items():
        gpd.GeoSeries([reg]).boundary.plot(ax=ax, color=COLORS[name], linewidth=1.5)
    for name, ids in assigned.items():
        gpd.GeoSeries([fps[t] for t in ids]).plot(
            ax=ax, facecolor=COLORS[name], edgecolor="none", alpha=0.08
        )
    if dropped:
        gpd.GeoSeries([fps[t] for t in dropped]).boundary.plot(ax=ax, color="grey", linewidth=0.2)
    handles = [
        plt.Line2D([], [], color=c, lw=6, alpha=0.5, label=f"{n} ({len(assigned[n])})")
        for n, c in COLORS.items()
    ]
    handles.append(plt.Line2D([], [], color="grey", lw=1, label=f"dropped ({len(dropped)})"))
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.set_title("SN6 Rotterdam: geographic splits (tile footprints, EPSG:32631)")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.ticklabel_format(style="plain")
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.figure)
    print(f"figure: {a.figure}")


if __name__ == "__main__":
    main()
