"""Fill the Lee-filtered SAR cache for tile lists (one process per tile, all cores).

Each tile's Lee-filtered dB (window ``--lee``, train-split ENL) is stored as float32
``<cache>/<tile_id>.lee<k>.enl<enl>.npy``; training then skips the per-sample filter.

Usage: uv run python scripts/build_sar_cache.py --root ROOT --cache ROOT/sar_cache
           --lists data/splits/train.txt ... [--lee 5 7] [--workers 64]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.raster import RasterReader
from geoinfer.train.dataset import chain_cache_path, ensure_sar_chain
from geoinfer.train.train import SAR_DIR, TILE_REGEX


def _one(job: tuple[str, str, int, float, str]) -> tuple[str, bool]:
    tile_id, sar_path, lee, enl, cache = job
    if chain_cache_path(Path(cache), tile_id, lee, enl).exists():
        return tile_id, False
    with RasterReader(Path(sar_path)) as r:
        db = r.read(Window(0, 0, r.info.width, r.info.height)).data
    ensure_sar_chain(tile_id, db, np.all(db != 0, axis=0), lee, enl, Path(cache))
    return tile_id, True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--lists", type=Path, nargs="+", required=True)
    p.add_argument("--lee", type=int, nargs="+", default=[5, 7])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    pcfg: dict[str, Any] = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
    stats = json.loads((Path(pcfg["stats"]["out_dir"]) / "stats.json").read_text())
    enl = float(stats["sar"]["enl"]["value"])
    rx = re.compile(TILE_REGEX)
    by_id = {m.group(1): f for f in (a.root / SAR_DIR).glob("*.tif") if (m := rx.search(f.name))}
    ids = sorted({t for lst in a.lists for t in lst.read_text().split()})
    jobs = [(t, str(by_id[t]), k, enl, str(a.cache)) for k in a.lee for t in ids]
    t0, done = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (_, new) in enumerate(ex.map(_one, jobs, chunksize=4), 1):
            done += new
            if i % 500 == 0 or i == len(jobs):
                print(
                    f"  {i}/{len(jobs)} ({done} new) {time.perf_counter() - t0:.0f} s", flush=True
                )
    size = sum(f.stat().st_size for f in a.cache.glob("*.npy")) / 2**30
    print(f"cache {a.cache}: {len(ids)} tiles x Lee {a.lee}, ENL {enl}, {size:.1f} GiB")


if __name__ == "__main__":
    main()
