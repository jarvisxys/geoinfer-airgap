"""Rasterise building footprints into the mask cache for a tile list.

Masks are uint8 GeoTIFFs (1 = building) on each SAR tile's grid, written once to
``<root>/masks/<tile_id>.tif`` (``train.dataset.ensure_mask``).

Usage: uv run python scripts/build_masks.py [--root ~/geo-data/subset]
           [--lists data/splits/subset_train.txt ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.config import load_io_config
from geoinfer.io.raster import RasterReader
from geoinfer.train.dataset import ensure_mask, index_tiles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    split_dir = Path(dcfg["splits"]["out_dir"])
    p.add_argument("--root", type=Path, default=Path(dcfg["subset"]["root"]).expanduser())
    p.add_argument(
        "--lists",
        type=Path,
        nargs="+",
        default=[split_dir / f"subset_{s}.txt" for s in ("train", "tune", "val")],
    )
    p.add_argument(
        "--interim-root",
        type=Path,
        default=Path(dcfg["interim_root"]).expanduser(),
        help="extracted SN6 AOI folder (holds SummaryData/)",
    )
    a = p.parse_args()
    interim = a.interim_root.expanduser()
    orient_lines = (interim / dcfg["orientations"]).read_text().splitlines()
    orient = {k: int(v) for k, v in (ln.split() for ln in orient_lines if ln.strip())}
    ids = sorted({t for f in a.lists for t in f.read_text().split()})
    recs = index_tiles(
        a.root,
        ids,
        dcfg["modalities"]["sar"],
        dcfg["modalities"]["optical"],
        dcfg["tile_id_regex"],
        orient,
    )
    io_cfg = load_io_config()
    t0 = time.perf_counter()
    frac, empty = [], 0
    for rec in recs:
        path = ensure_mask(rec, a.root / "masks", io_cfg)
        with RasterReader(path) as r:
            m = r.read(Window(0, 0, r.info.width, r.info.height)).data[0]
        frac.append(float(m.mean()))
        empty += int(m.sum() == 0)
    print(
        f"{len(recs)} masks in {a.root / 'masks'} ({time.perf_counter() - t0:.1f} s); "
        f"building fraction mean {np.mean(frac):.3f}, tiles without buildings {empty}"
    )


if __name__ == "__main__":
    main()
