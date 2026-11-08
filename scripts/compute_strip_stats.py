"""Per-strip SAR statistics for the strip-scoped S2 arms (decision D4).

SN6 SAR-Intensity brightness differs by up to about 29 dB between acquisition strips.
For every arm with ``scope: strip`` this script streams all extracted D2 SAR tiles
(every split: the statistics use input pixels only, never labels, as a deployed system
would on a new scene), applies the arm's speckle filter, and accumulates one histogram
per strip. Percentiles and clipped moments per strip go to ``strip_stats.json`` next to
``stats.json``; strips with too few valid pixels are written as null and fall back to
the arm's global ``fallback`` arm.

Usage: uv run python scripts/compute_strip_stats.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.config import load_io_config
from geoinfer.io.paths import assert_writable
from geoinfer.io.raster import RasterReader
from geoinfer.preprocess.sar import sar_chain
from geoinfer.preprocess.stats import BandHistogram, strip_arm_stats


def _jsonable(stats: dict[str, Any] | None) -> dict[str, list[float]] | None:
    if stats is None:
        return None
    return {k: np.round(np.asarray(v, dtype=float), 4).tolist() for k, v in stats.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    dcfg: dict[str, Any] = yaml.safe_load(a.data_config.read_text())
    sc = cfg["sar"]
    out_dir = Path(cfg["stats"]["out_dir"]).expanduser()
    out = out_dir / sc["strip_stats"]["out"]
    assert_writable(out, load_io_config().read_only_roots)
    enl = float(json.loads((out_dir / "stats.json").read_text())["sar"]["enl"]["value"])
    arms = {k: v for k, v in sc["arms"].items() if v.get("scope") == "strip"}
    if not arms:
        raise SystemExit("no arm with scope: strip in the config")
    lee_sizes = sorted({arm["lee"] for arm in arms.values()}, key=lambda s: s or 0)

    rx = re.compile(dcfg["tile_id_regex"])
    folder = Path(dcfg["interim_root"]).expanduser() / dcfg["modalities"]["sar"]
    tiles = sorted((m.group(1), f) for f in folder.glob("*.tif") if (m := rx.search(f.name)))
    hb = sc["hist"]
    hists: dict[Any, dict[str, BandHistogram]] = {lee: {} for lee in lee_sizes}
    n_tiles: dict[str, int] = defaultdict(int)
    print(f"{len(tiles)} tiles, arms {list(arms)}, ENL {enl}")
    t0 = time.perf_counter()
    for i, (tid, path) in enumerate(tiles, 1):
        strip = tid.split("_tile_")[0]
        with RasterReader(path) as r:
            db = r.read(Window(0, 0, r.info.width, r.info.height)).data
        valid = np.all(db != 0, axis=0)
        n_tiles[strip] += 1
        if not valid.any():
            continue
        for lee in lee_sizes:
            h = hists[lee].setdefault(strip, BandHistogram(4, hb["min"], hb["max"], hb["step"]))
            h.add(sar_chain(db, valid, lee, enl), valid)
        if i % 250 == 0 or i == len(tiles):
            print(f"  {i}/{len(tiles)}  {time.perf_counter() - t0:.0f} s")

    min_px = int(sc["strip_stats"]["min_valid_px"])
    result: dict[str, Any] = {
        "source": "all extracted D2 SAR tiles (input pixels only, no labels)",
        "tiles": len(tiles),
        "enl": enl,
        "percentiles": sc["percentiles"],
        "min_valid_px": min_px,
        "arms": {},
    }
    for name, arm in arms.items():
        hs = hists[arm["lee"]]
        per = strip_arm_stats(hs, tuple(sc["percentiles"]), arm["domain"], arm["clip"], min_px)
        means = np.array([s["mean"] for s in per.values() if s is not None])
        result["arms"][name] = {
            "lee": arm["lee"],
            "domain": arm["domain"],
            "clip": arm["clip"],
            "fallback": arm["fallback"],
            "strips": {k: _jsonable(v) for k, v in sorted(per.items())},
            "valid_px": {k: int(h.counts[0].sum()) for k, h in sorted(hs.items())},
            "tiles": dict(sorted(n_tiles.items())),
        }
        n_fb = sum(v is None for v in per.values())
        spread = (means.max(axis=0) - means.min(axis=0)).round(2).tolist()
        print(
            f"{name}: {len(per)} strips, {n_fb} fall back; strip mean spread per band {spread} dB"
        )
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
