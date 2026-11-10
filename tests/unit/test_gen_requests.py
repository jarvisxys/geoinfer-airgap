from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape

from geoinfer.service.worker.pipeline import to_crs

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gen_requests", REPO / "scripts" / "gen_requests.py")
assert spec is not None
assert spec.loader is not None
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

CRS = "EPSG:32631"
EXT = {
    "big": (CRS, (594000.0, 5750000.0, 604000.0, 5760000.0)),
    "small": (CRS, (594000.0, 5750000.0, 595300.0, 5752000.0)),
}
CFG = {
    "requests": {
        "seed": 42,
        "n": 400,
        "high_priority_share": 0.2,
        "priorities": {"high": 5, "normal": 0},
        "modes": {
            "optical": {"weight": 0.6, "scene": "big", "sides_km": [0.5, 1, 2, 5]},
            "sar": {"weight": 0.4, "scene": "small", "sides_km": [0.5, 1, 2]},
        },
    }
}


def test_requests_are_deterministic_poisson_and_inside_their_scene() -> None:
    a, b = gen.generate(CFG, EXT), gen.generate(CFG, EXT)
    assert a == b
    t = np.array([r["t"] for r in a])
    gaps = np.diff(np.concatenate([[0.0], t]))
    assert (gaps > 0).all()
    assert abs(gaps.mean() - 1.0) < 0.15  # unit rate
    assert 0.12 < np.mean([r["priority"] == 5 for r in a]) < 0.28
    for r in a:
        crs, bounds = EXT[r["scene_id"]]
        aoi = to_crs(shape(r["aoi"]), "EPSG:4326", crs)
        assert box(*bounds).buffer(1.0).contains(aoi)
        assert abs(aoi.area - (r["side_km"] * 1000) ** 2) / aoi.area < 0.01


def test_sides_too_big_for_the_scene_are_never_drawn() -> None:
    small = {r["side_km"] for r in gen.generate(CFG, EXT) if r["scene_id"] == "small"}
    assert small <= {0.5, 1.0}  # the 2 km side does not fit a 1.3 km wide scene
    assert {r["side_km"] for r in gen.generate(CFG, EXT) if r["scene_id"] == "big"} == {
        0.5,
        1.0,
        2.0,
        5.0,
    }
