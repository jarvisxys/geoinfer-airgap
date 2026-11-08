"""Train-split statistics for the S2 normalisation study.

Pass 1 estimates the SAR ENL (if ``sar.enl`` is null in the config) on every
``enl_estimate.tile_stride``-th train tile. Pass 2 streams the train tiles once, one at
a time, into fixed-bin histograms: optical uint16 and one dB histogram per SAR arm
(stored, Lee 5x5, Lee 7x7). Percentiles, clip ranges and z-score moments are read off
the histograms.

Outputs (in ``stats.out_dir``): ``stats.json`` and ``hist_*.npz``.

Usage: uv run python scripts/compute_stats.py [--split data/splits/train.txt]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.config import load_io_config
from geoinfer.io.paths import assert_writable
from geoinfer.io.raster import RasterReader
from geoinfer.preprocess.sar import db_to_linear, estimate_enl, sar_chain
from geoinfer.preprocess.stats import BandHistogram, optical_stats, sar_arm_stats


def _index(folder: Path, pattern: str) -> dict[str, Path]:
    rx = re.compile(pattern)
    return {m.group(1): f for f in folder.glob("*.tif") if (m := rx.search(f.name))}


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with RasterReader(path) as rd:
        out = rd.read(Window(0, 0, rd.info.width, rd.info.height))  # one 900 px tile
    return out.data, out.valid


def _hist(n_bands: int, spec: dict[str, float]) -> BandHistogram:
    return BandHistogram(n_bands, spec["min"], spec["max"], spec["step"])


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.round(6).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--split", type=Path, default=Path("data/splits/train.txt"))
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    dcfg: dict[str, Any] = yaml.safe_load(a.data_config.read_text())
    sc, oc = cfg["sar"], cfg["optical"]
    out_dir = Path(cfg["stats"]["out_dir"]).expanduser()
    assert_writable(out_dir, load_io_config().read_only_roots)

    interim = Path(dcfg["interim_root"]).expanduser()
    sar_idx = _index(interim / dcfg["modalities"]["sar"], dcfg["tile_id_regex"])
    opt_idx = _index(interim / dcfg["modalities"]["optical"], dcfg["tile_id_regex"])
    ids = a.split.read_text().split()[:: cfg["stats"]["tile_stride"]]
    print(f"{len(ids)} tiles from {a.split}")

    # pass 1: ENL of the speckle, on linear intensity
    t0 = time.perf_counter()
    if sc["enl"] is None:
        ec = sc["enl_estimate"]
        per_tile: list[list[float]] = []
        for tid in ids[:: ec["tile_stride"]]:
            db, valid = _read(sar_idx[tid])
            lin = db_to_linear(db, valid)
            try:
                per_tile.append([estimate_enl(b, valid, ec["window"], ec["quantile"]) for b in lin])
            except ValueError:
                continue  # no fully valid window in this tile
        enl_bands = np.median(np.array(per_tile), axis=0)
        enl_value = float(np.median(enl_bands))
        enl_info: dict[str, Any] = {
            "source": "estimate_enl, median over tiles then bands",
            "per_band": enl_bands.round(3).tolist(),
            "tiles": len(per_tile),
        }
        print(
            f"ENL per band {enl_info['per_band']} -> {enl_value:.2f} "
            f"({len(per_tile)} tiles, {time.perf_counter() - t0:.0f} s)"
        )
    else:
        enl_value = float(sc["enl"])
        enl_info = {"source": "config"}
    enl_info["value"] = round(enl_value, 3)

    # pass 2: one streaming read of the split into histograms
    t0 = time.perf_counter()
    opt_hist = _hist(4, oc["hist"])
    lee_sizes = sorted({arm["lee"] for arm in sc["arms"].values()}, key=lambda s: s or 0)
    sar_hist = {lee: _hist(4, sc["hist"]) for lee in lee_sizes}
    for i, tid in enumerate(ids, 1):
        opt, ov = _read(opt_idx[tid])
        opt_hist.add(opt, ov)
        db, sv = _read(sar_idx[tid])
        for lee, h in sar_hist.items():
            h.add(sar_chain(db, sv, lee, enl_value), sv)
        if i % 200 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}  {time.perf_counter() - t0:.0f} s")

    out_dir.mkdir(parents=True, exist_ok=True)
    opt_hist.save(out_dir / "hist_optical.npz")
    for lee, h in sar_hist.items():
        h.save(out_dir / f"hist_sar_{'db' if lee is None else f'lee{lee}'}.npz")

    pct = tuple(sc["percentiles"])
    arms = {
        name: _jsonable(sar_arm_stats(sar_hist[arm["lee"]], pct, arm["domain"], arm["clip"]))
        | {"lee": arm["lee"], "domain": arm["domain"], "clip": arm["clip"]}
        for name, arm in sc["arms"].items()
    }
    stats = {
        "split": str(a.split),
        "split_sha256": hashlib.sha256(a.split.read_bytes()).hexdigest(),
        "tiles": len(ids),
        "optical": _jsonable(optical_stats(opt_hist, tuple(oc["percentiles"])))
        | {"percentiles": oc["percentiles"], "clipped": opt_hist.clipped.tolist()},
        "sar": {
            "percentiles": sc["percentiles"],
            "enl": enl_info,
            "arms": arms,
            "clipped": {str(k): h.clipped.tolist() for k, h in sar_hist.items()},
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_dir}/stats.json")


if __name__ == "__main__":
    main()
