"""Synthetic analyst requests for the serving experiments (plan section 4, E2-E4).

One fixed sequence of ``requests.n`` jobs (configs/loadtest.yaml), so every load level and
policy sees the same work. Arrival offsets are a unit-rate Poisson process (exponential
gaps, mean 1 s); the replay divides them by the target rate. Each request draws a sensor
mode by weight, an AOI side for that mode, a uniform AOI centre that keeps the square
inside the scene (scene CRS), and a priority (``high_priority_share`` high). AOIs are
written as EPSG:4326 GeoJSON polygons.

Usage: uv run python scripts/gen_requests.py [--inbox DIR]   (default data_dir/inbox)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from shapely.geometry import box, mapping

from geoinfer.service.worker.pipeline import to_crs


def scene_extent(path: Path) -> tuple[Any, tuple[float, float, float, float]]:
    """CRS and bounds of a raster (header only)."""
    with rasterio.open(path) as r:
        return r.crs, tuple(r.bounds)  # type: ignore[return-value]


def generate(cfg: dict[str, Any], extents: dict[str, tuple[Any, tuple[float, ...]]]) -> list[dict]:
    """The request list (see module docstring); deterministic for a given seed."""
    rq = cfg["requests"]
    rng = np.random.default_rng(int(rq["seed"]))
    modes = list(rq["modes"])
    w = np.array([rq["modes"][m]["weight"] for m in modes], float)
    t, out = 0.0, []
    for i in range(int(rq["n"])):
        t += float(rng.exponential(1.0))
        mode = modes[int(rng.choice(len(modes), p=w / w.sum()))]
        spec = rq["modes"][mode]
        crs, (x0, y0, x1, y1) = extents[spec["scene"]]
        sides = [s for s in spec["sides_km"] if s * 1000 <= min(x1 - x0, y1 - y0)]
        side = float(rng.choice(sides)) * 1000
        cx = float(rng.uniform(x0 + side / 2, x1 - side / 2))
        cy = float(rng.uniform(y0 + side / 2, y1 - side / 2))
        aoi = to_crs(
            box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), crs, "EPSG:4326"
        )
        high = bool(rng.random() < float(rq["high_priority_share"]))
        out.append(
            {
                "i": i,
                "t": round(t, 4),
                "scene_id": spec["scene"],
                "sensor_mode": mode,
                "task": "segment_buildings",
                "side_km": side / 1000,
                "priority": int(rq["priorities"]["high" if high else "normal"]),
                "aoi": mapping(aoi),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/loadtest.yaml"))
    p.add_argument("--inbox", type=Path, default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    inbox = a.inbox or Path(cfg["data_dir"]) / "inbox"
    extents = {k: scene_extent(inbox / v["path"]) for k, v in cfg["scenes"].items()}
    reqs = generate(cfg, extents)
    out = Path(cfg["requests"]["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in reqs))
    sides: dict[float, int] = {}
    for r in reqs:
        sides[r["side_km"]] = sides.get(r["side_km"], 0) + 1
    print(f"wrote {len(reqs)} requests to {out}; AOI sides {dict(sorted(sides.items()))}")


if __name__ == "__main__":
    main()
