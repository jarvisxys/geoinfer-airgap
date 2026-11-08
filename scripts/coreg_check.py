"""SAR vs optical co-registration check on the working subset (pipeline stage 2 evidence).

For each subset tile, estimates the shift of the SAR edge map relative to the optical
one (smoothed Sobel magnitude + masked NCC) and writes a per-tile CSV, a summary per
SAR look direction, and a scatter figure. Building layover displaces SAR edges
towards the sensor, so a look-direction-dependent offset is expected; a global
mis-registration would show as the same offset in both directions.

Usage: uv run python scripts/coreg_check.py [--csv docs/figures/coreg_shifts.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.raster import RasterReader
from geoinfer.preprocess.coreg import coregistration_shift

SPLITS = ("train", "tune", "val")
COLORS = {0: "tab:blue", 1: "tab:red"}


def _index(folder: Path, pattern: str) -> dict[str, Path]:
    rx = re.compile(pattern)
    return {m.group(1): f for f in folder.glob("*.tif") if (m := rx.search(f.name))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    p.add_argument("--csv", type=Path, default=Path("docs/figures/coreg_shifts.csv"))
    p.add_argument("--figure", type=Path, default=Path("docs/figures/coreg_shifts.png"))
    a = p.parse_args()
    dcfg: dict[str, Any] = yaml.safe_load(a.data_config.read_text())
    cc: dict[str, Any] = yaml.safe_load(a.config.read_text())["coreg"]

    root = Path(dcfg["subset"]["root"]).expanduser()
    sar_idx = _index(root / dcfg["modalities"]["sar"], dcfg["tile_id_regex"])
    opt_idx = _index(root / dcfg["modalities"]["optical"], dcfg["tile_id_regex"])
    split_dir = Path(dcfg["splits"]["out_dir"])
    orient_file = Path(dcfg["interim_root"]).expanduser() / dcfg["orientations"]
    orient = dict(line.split() for line in orient_file.read_text().splitlines() if line.strip())

    rows: list[dict[str, Any]] = []
    skipped = 0
    for split in SPLITS:
        for tid in (split_dir / f"subset_{split}.txt").read_text().split():
            with RasterReader(sar_idx[tid]) as s, RasterReader(opt_idx[tid]) as o:
                si, oi = s.info, o.info
                same_grid = (si.width, si.height) == (oi.width, oi.height) and (
                    si.transform.almost_equals(oi.transform, precision=cc["grid_tol_m"])
                )
                if not same_grid:
                    raise ValueError(f"{tid}: SAR and optical grids differ")
                full = Window(0, 0, si.width, si.height)
                sar = s.read(full).data
                opt = o.read(full).data
            valid = np.all(sar != 0, axis=0) & np.any(opt != 0, axis=0)
            frac = float(valid.mean())
            if frac < cc["min_valid"]:
                skipped += 1
                continue
            r = coregistration_shift(
                sar,
                opt,
                valid,
                cc["window"],
                cc["sobel_ksize"],
                cc["optical_bands"],
                cc["blur_sigma"],
                cc["max_shift"],
                cc["overlap_ratio"],
            )
            rows.append(
                {
                    "tile": tid,
                    "split": split,
                    "orientation": int(orient[tid.split("_tile_")[0]]),
                    "valid": round(frac, 3),
                    "dx_px": round(r.dx, 3),
                    "dy_px": round(r.dy, 3),
                    "shift_px": round(float(np.hypot(r.dx, r.dy)), 3),
                    "response": round(r.response, 4),
                }
            )

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    d = np.array([r["shift_px"] for r in rows])
    edge = cc["max_shift"] - 0.5
    at_edge = np.mean([max(abs(r["dx_px"]), abs(r["dy_px"])) >= edge for r in rows])
    print(f"tiles: {len(rows)} measured, {skipped} skipped (valid < {cc['min_valid']})")
    print(f"peak at the search boundary (unreliable): {at_edge:.1%}")
    print(f"response (NCC) median {np.median([r['response'] for r in rows]):.3f}")
    print(f"|shift| median {np.median(d):.2f} px, p90 {np.percentile(d, 90):.2f} px")
    for o in (0, 1):
        sel = [r for r in rows if r["orientation"] == o]
        if sel:
            mdx = np.median([r["dx_px"] for r in sel])
            mdy = np.median([r["dy_px"] for r in sel])
            print(f"orientation {o}: {len(sel):3d} tiles, median dx {mdx:+.2f}, dy {mdy:+.2f} px")
    print(f"wrote {a.csv}")

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
    for o, color in COLORS.items():
        sel = [r for r in rows if r["orientation"] == o]
        ax.scatter(
            [r["dx_px"] for r in sel],
            [r["dy_px"] for r in sel],
            s=12,
            alpha=0.7,
            color=color,
            label=f"look direction {o} ({len(sel)})",
        )
    lim = max(2.0, float(np.percentile(d, 98)) * 1.1)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls="--", color="grey"))
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("dx (px, 0.5 m)")
    ax.set_ylabel("dy (px, 0.5 m)")
    ax.set_title("SAR vs optical shift per tile (masked NCC of edges)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.figure)
    print(f"figure: {a.figure}")


if __name__ == "__main__":
    main()
