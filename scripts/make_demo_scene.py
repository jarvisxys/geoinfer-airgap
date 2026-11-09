"""Cut the demo scene for the service smoke / air-gap test out of the D4 mosaic.

Picks the band of ``demo_scene.rows`` rows (configs/service.yaml) where one SAR look
direction owns the most valid pixels, and writes to ``INBOX``:

- ``demo_optical.tif``: the optical band (unchanged);
- ``demo_sar.tif``: the SAR band with pixels of the other look direction set to nodata
  (0), so the scene has a single orientation (decision D5);
- ``demo_labels.tif`` next to the inbox (``INBOX/../demo_labels.tif``): building ids, for
  scoring the service output. Prints the band and the orientation to register.

Every read is a window of at most 1024 rows; outputs are COGs on the band's grid.

Usage: uv run python scripts/make_demo_scene.py INBOX
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from geoinfer.io.config import load_io_config
from geoinfer.io.raster import RasterReader, RasterWriter

BLOCK = 1024


def _orientation_counts(strips: Path, lookup: np.ndarray, rows: int) -> np.ndarray:
    """Per-row pixel counts of look direction 0 and 1, shape (rows, 2)."""
    out = np.zeros((rows, 2), np.int64)
    with RasterReader(strips) as r:
        for r0 in range(0, rows, BLOCK):
            h = min(BLOCK, rows - r0)
            ids = r.read(Window(0, r0, r.info.width, h), [1]).data[0].astype(np.int64)
            o = lookup[ids]  # -1 for nodata
            out[r0 : r0 + h, 0] = (o == 0).sum(axis=1)
            out[r0 : r0 + h, 1] = (o == 1).sum(axis=1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inbox", type=Path)
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(Path("configs/service.yaml").read_text())["demo_scene"]
    src = Path(yaml.safe_load(Path("configs/data.yaml").read_text())["scenes"][cfg["source"]])
    key = cfg["source"]
    meta = json.loads((src / f"{key}.json").read_text())
    n_ids = max(int(k) for k in meta["strips"]) + 1
    lookup = np.full(n_ids, -1, np.int8)
    for k, v in meta["strips"].items():
        lookup[int(k)] = int(v["orientation"])
    height, width = meta["grid"]["height"], meta["grid"]["width"]
    rows, step = int(cfg["rows"]), int(cfg["step"])
    counts = np.cumsum(
        np.vstack([[0, 0], _orientation_counts(src / f"{key}_strips.tif", lookup, height)]), 0
    )
    best = max(
        (
            (r0, o, int(counts[r0 + rows, o] - counts[r0, o]))
            for r0 in range(0, height - rows + 1, step)
            for o in (0, 1)
        ),
        key=lambda t: t[2],
    )
    r0, orient, n = best
    print(
        f"band rows {r0}-{r0 + rows}: look direction {orient} on {n / (rows * width):.0%} of pixels"
    )
    io_cfg = load_io_config()
    a.inbox.mkdir(parents=True, exist_ok=True)
    win_all = Window(0, r0, width, rows)
    outputs = {
        "optical": (src / f"{key}_optical.tif", a.inbox / "demo_optical.tif"),
        "sar": (src / f"{key}_sar.tif", a.inbox / "demo_sar.tif"),
        "labels": (src / f"{key}_labels.tif", a.inbox.parent / "demo_labels.tif"),
    }
    with RasterReader(src / f"{key}_strips.tif") as rs:
        for name, (sp, dp) in outputs.items():
            with RasterReader(sp) as r:
                like = replace(
                    r.info, transform=window_transform(win_all, r.info.transform), height=rows
                )
                with RasterWriter(
                    dp,
                    like,
                    io_cfg,
                    nodata=r.info.nodata,
                    overview_resampling="NEAREST" if name == "labels" else None,
                ) as w:
                    for b0 in range(0, rows, BLOCK):
                        h = min(BLOCK, rows - b0)
                        win = Window(0, r0 + b0, width, h)
                        data = r.read(win).data
                        if name == "sar":
                            ids = rs.read(win, [1]).data[0].astype(np.int64)
                            data = np.where(lookup[ids] == orient, data, 0).astype(data.dtype)
                        w.write(data, Window(0, b0, width, h))
            print(f"wrote {dp}")
    print(json.dumps({"rows": [r0, r0 + rows], "orientation": orient}))


if __name__ == "__main__":
    main()
