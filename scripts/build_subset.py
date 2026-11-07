"""Copy a compact ~200-tile working subset (SAR + optical COGs + labels) to WSL ext4.

Needs data/splits/ from build_splits.py. Resumable: tiles already converted are skipped.
Refuses to start if the target disk would drop below ``subset.min_free_gb``.

Usage: uv run python scripts/build_subset.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from geoinfer.io.cog import to_cog
from geoinfer.io.config import DEFAULT_IO_CONFIG, load_io_config
from geoinfer.train.splits import select_subset, tile_footprints

TILE_CRS = "EPSG:32631"
GIB = 1024**3


def _by_id(folder: Path, rx: re.Pattern[str], suffix: str) -> dict[str, Path]:
    return {m.group(1): f for f in folder.glob(f"*{suffix}") if (m := rx.search(f.name))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--io-config", type=Path, default=DEFAULT_IO_CONFIG)
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    io_cfg = load_io_config(a.io_config)
    interim = Path(cfg["interim_root"]).expanduser()
    root = Path(cfg["subset"]["root"]).expanduser()
    split_dir = Path(cfg["splits"]["out_dir"])
    sar_name, opt_name = cfg["modalities"]["sar"], cfg["modalities"]["optical"]

    tif_rx = re.compile(cfg["tile_id_regex"])
    geo_rx = re.compile(cfg["tile_id_regex"].replace(r"\.tif$", r"\.geojson$"))
    sar = _by_id(interim / sar_name, tif_rx, ".tif")
    opt = _by_id(interim / opt_name, tif_rx, ".tif")
    labels = _by_id(interim / "geojson_buildings", geo_rx, ".geojson")

    counts: dict[str, int] = cfg["subset"]["counts"]
    assigned = {s: (split_dir / f"{s}.txt").read_text().split() for s in counts}
    wanted = sorted({t for ids in assigned.values() for t in ids})
    fps = tile_footprints([sar[t] for t in wanted], cfg["tile_id_regex"], TILE_CRS)
    chosen = select_subset(fps, assigned, counts)
    tiles = sorted(t for ids in chosen.values() for t in ids)

    need = sum(sar[t].stat().st_size + opt[t].stat().st_size for t in tiles)  # upper bound
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    min_free = cfg["subset"]["min_free_gb"] * GIB
    print(f"{len(tiles)} tiles, at most {need / GIB:.1f} GiB; {free / GIB:.1f} GiB free on target")
    if free - need < min_free:
        raise SystemExit(f"refusing: would leave < {cfg['subset']['min_free_gb']} GiB free")

    for i, t in enumerate(tiles, 1):
        for name, src in ((sar_name, sar[t]), (opt_name, opt[t])):
            dst = root / name / src.name
            if not dst.exists():
                to_cog(src, dst, io_cfg)
        if t in labels:
            lab = root / "geojson_buildings" / labels[t].name
            lab.parent.mkdir(exist_ok=True)
            if not lab.exists():
                shutil.copyfile(labels[t], lab)
        if i % 20 == 0 or i == len(tiles):
            print(f"  {i}/{len(tiles)}")

    for s, ids in chosen.items():
        (split_dir / f"subset_{s}.txt").write_text("".join(f"{t}\n" for t in ids))
    used = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    manifest = {
        "counts": {s: len(ids) for s, ids in chosen.items()},
        "size_gib": round(used / GIB, 2),
        "source": str(interim),
    }
    (root / "subset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"subset at {root}: {manifest}")


if __name__ == "__main__":
    main()
