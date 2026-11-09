"""Score the smoke-test jobs (outputs/smoke.json) against the demo scene labels.

Pixel IoU of each job's mask.tif against ``demo_labels.tif`` on pixels the mask covers
(value != 255) that have a label-raster pixel. Windowed reads only.

Usage: uv run python scripts/score_service_outputs.py DATA   (GEOINFER_DATA of the stack)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    a = p.parse_args()
    smoke = json.loads((a.data / "outputs" / "smoke.json").read_text())
    with rasterio.open(a.data / "demo_labels.tif") as lab:
        for job in smoke["jobs"]:
            if job["status"] != "ready":
                print(f"{job['mode']:8s} {job['status']}")
                continue
            with rasterio.open(a.data / "outputs" / job["job"] / "mask.tif") as m:
                mask = m.read(1)
                win = from_bounds(*m.bounds, lab.transform).round_offsets().round_lengths()
            gt = lab.read(1, window=win, boundless=True, fill_value=0) > 0
            ok = mask != 255
            pr, g = (mask == 1) & ok, gt & ok
            iou = (pr & g).sum() / max(1, (pr | g).sum())
            print(
                f"{job['mode']:8s} {job['model']}: pixel IoU {iou:.4f} on {ok.mean():.0%} "
                f"of the AOI, {job['buildings']} buildings, {job['n_tiles']} tiles, "
                f"{job['timings']['total_s']} s"
            )


if __name__ == "__main__":
    main()
