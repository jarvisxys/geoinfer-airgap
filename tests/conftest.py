"""Shared fixtures: tiny synthetic GeoTIFFs written to tmp_path."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from geoinfer.io.config import IOConfig, load_io_config
from geoinfer.train.dataset import Normalizer, index_tiles

# SpaceNet 6 Rotterdam tiles are in UTM zone 31N.
CRS = "EPSG:32631"
TRANSFORM = from_origin(594_000.0, 5_757_000.0, 0.5, 0.5)

MakeTiff = Callable[..., Path]

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def io_cfg(tmp_path: Path) -> IOConfig:
    """Repo ``configs/io.yaml`` with the read-only root moved to ``tmp_path/raw``."""
    cfg = load_io_config(REPO_ROOT / "configs" / "io.yaml")
    return replace(cfg, read_only_roots=(tmp_path / "raw",))


@pytest.fixture
def make_tiff(tmp_path: Path) -> MakeTiff:
    """Factory that writes ``array`` (bands, h, w) to a GeoTIFF and returns its path."""

    def _make(
        array: npt.NDArray[np.generic], name: str = "img.tif", nodata: float | None = None
    ) -> Path:
        path = tmp_path / name
        count, h, w = array.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=count,
            dtype=array.dtype,
            crs=CRS,
            transform=TRANSFORM,
            nodata=nodata,
        ) as dst:
            dst.write(array)
        return path

    return _make


# --- a tiny two-strip SN6-like scene (dataset and training tests) ---------------------


RX = r"_(\d{14}_\d{14}_tile_\d+)\.tif$"
STRIP_A, STRIP_B = "20190804111224_20190804111453", "20190822070610_20190822070846"
N = 64
X0, Y0 = 594_000.0, 5_757_000.0


def _write(path: Path, arr: np.ndarray, nodata: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=N,
        width=N,
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=CRS,
        transform=from_origin(X0, Y0, 0.5, 0.5),
        nodata=nodata,
    ) as d:
        d.write(arr)


@pytest.fixture
def scene(tmp_path: Path) -> dict:
    """Two tiny SN6-like tiles (one per strip) with SAR, optical and labels."""
    root = tmp_path / "subset"
    rng = np.random.default_rng(0)
    ids = []
    for strip, k, offset in ((STRIP_A, 1, 0.0), (STRIP_B, 2, 10.0)):
        tid = f"{strip}_tile_{k}"
        ids.append(tid)
        sar = (25 + offset + 3 * rng.standard_normal((4, N, N))).astype(np.float32)
        sar[:, :, :8] = 0  # nodata strip on the west edge
        _write(
            root / "SAR-Intensity" / f"SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{tid}.tif", sar, 0
        )
        opt = rng.integers(100, 900, (4, N, N)).astype(np.uint16)
        opt[:, :, :8] = 0
        _write(root / "PS-RGBNIR" / f"SN6_Train_AOI_11_Rotterdam_PS-RGBNIR_{tid}.tif", opt, 0)
        # one building in the north-east quarter: asymmetric, so flips are visible
        b = box(X0 + 20.0, Y0 - 10.0, X0 + 28.0, Y0 - 4.0)  # cols 40-56, rows 8-20
        gj = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32631"}},
            "features": [
                {"type": "Feature", "properties": {"Building_ID": 7}, "geometry": mapping(b)}
            ],
        }
        g = root / "geojson_buildings" / f"SN6_Train_AOI_11_Rotterdam_Buildings_{tid}.geojson"
        g.parent.mkdir(parents=True, exist_ok=True)
        g.write_text(json.dumps(gj))
    stats = {
        "optical": {"p_lo": [100.0] * 4, "p_hi": [900.0] * 4, "mean": [0.5] * 4, "std": [0.25] * 4},
        "sar": {
            "enl": {"value": 2.0},
            "arms": {
                "db": {
                    "p_lo": [15.0] * 4,
                    "p_hi": [45.0] * 4,
                    "mean": [30.0] * 4,
                    "std": [5.0] * 4,
                    "lee": None,
                    "domain": "db",
                    "clip": True,
                }
            },
        },
    }
    strip = {
        "arms": {
            "db_strip": {
                "lee": None,
                "domain": "db",
                "clip": True,
                "fallback": "db",
                "strips": {
                    STRIP_A: {
                        "p_lo": [10.0] * 4,
                        "p_hi": [40.0] * 4,
                        "mean": [25.0] * 4,
                        "std": [3.0] * 4,
                    },
                    STRIP_B: None,
                },
            }
        }
    }
    sd = tmp_path / "stats"
    sd.mkdir()
    (sd / "stats.json").write_text(json.dumps(stats))
    (sd / "strip_stats.json").write_text(json.dumps(strip))
    orient = {STRIP_A: 0, STRIP_B: 1}
    arms = {
        "db": {"lee": None, "domain": "db", "clip": True},
        "db_strip": {"lee": None, "domain": "db", "clip": True, "scope": "strip", "fallback": "db"},
    }
    return {"root": root, "ids": ids, "stats": sd, "orient": orient, "arms": arms}


def _records(s: dict) -> list:
    return index_tiles(s["root"], s["ids"], "SAR-Intensity", "PS-RGBNIR", RX, s["orient"])


def _norm(s: dict, optical: str = "global", sar: str = "db") -> Any:
    return Normalizer.from_files(
        s["stats"] / "stats.json", s["stats"] / "strip_stats.json", optical, sar, s["arms"]
    )
