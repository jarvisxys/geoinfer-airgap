from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import inspect_raster
from geoinfer.train.dataset import (
    Normalizer,
    SN6Tiles,
    ensure_mask,
    index_tiles,
    rasterize_labels,
)
from tests.conftest import CRS

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


def _records(s: dict):
    return index_tiles(s["root"], s["ids"], "SAR-Intensity", "PS-RGBNIR", RX, s["orient"])


def _norm(s: dict, optical: str = "global", sar: str = "db") -> Normalizer:
    return Normalizer.from_files(
        s["stats"] / "stats.json", s["stats"] / "strip_stats.json", optical, sar, s["arms"]
    )


def test_index_pairs_tiles_by_id_with_strip_and_orientation(scene: dict) -> None:
    recs = _records(scene)
    assert [r.tile_id for r in recs] == scene["ids"]
    assert recs[0].strip == STRIP_A
    assert (recs[0].orientation, recs[1].orientation) == (0, 1)
    assert recs[0].labels is not None
    os.remove(recs[1].optical)
    with pytest.raises(FileNotFoundError, match="PS-RGBNIR"):
        _records(scene)


def test_rasterize_labels_on_the_tile_grid(scene: dict) -> None:
    rec = _records(scene)[0]
    m = rasterize_labels(rec.labels, inspect_raster(rec.sar))
    assert m.dtype == np.uint8
    assert m.sum() == 16 * 12  # 8 m x 6 m at 0.5 m
    assert m[8:20, 40:56].all()


def test_mask_cache_keeps_crs_and_transform_and_is_reused(scene: dict, io_cfg: IOConfig) -> None:
    rec = _records(scene)[0]
    masks = scene["root"] / "masks"
    p = ensure_mask(rec, masks, io_cfg)
    info, ref = inspect_raster(p), inspect_raster(rec.sar)
    assert info.crs == ref.crs
    assert info.transform == ref.transform
    assert info.dtypes == ("uint8",)
    t = p.stat().st_mtime_ns
    assert ensure_mask(rec, masks, io_cfg) == p
    assert p.stat().st_mtime_ns == t  # cached, not rewritten


def test_mask_of_tile_without_labels_is_empty(scene: dict, io_cfg: IOConfig) -> None:
    rec = _records(scene)[0]
    rec.labels.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    m = rasterize_labels(rec.labels, inspect_raster(rec.sar))
    assert m.sum() == 0


def test_normalizer_optical_and_sar_arms(scene: dict) -> None:
    x = np.full((4, 2, 2), 500, dtype=np.uint16)
    v = np.ones((2, 2), bool)
    np.testing.assert_allclose(_norm(scene).optical(x, v), 0.0)  # stretch 0.5 -> z 0
    mm = _norm(scene, optical="minmax").optical(x, v)
    assert mm.min() >= 0
    assert mm.max() <= 1
    db = np.full((4, 2, 2), 25.0, dtype=np.float32)
    np.testing.assert_allclose(_norm(scene).sar(db, v, STRIP_A), -1.0)  # (25 - 30) / 5
    strip = _norm(scene, sar="db_strip")
    np.testing.assert_allclose(strip.sar(db, v, STRIP_A), 0.0)  # strip A's own mean
    np.testing.assert_allclose(strip.sar(db, v, STRIP_B), -1.0)  # B falls back to global
    with pytest.raises(ValueError, match="optical"):
        _norm(scene, optical="median")


@pytest.mark.parametrize(("modality", "channels"), [("optical", 4), ("sar", 4), ("fused", 8)])
def test_dataset_channels_and_shapes(
    scene: dict, io_cfg: IOConfig, modality: str, channels: int
) -> None:
    ds = SN6Tiles(
        _records(scene),
        modality,
        _norm(scene),
        scene["root"] / "masks",
        io_cfg,
        crop=32,
        train=True,
        hflip=False,
        align_orientation=False,
        seed=1,
    )
    s = ds[0]
    assert s["image"].shape == (channels, 32, 32)
    assert s["image"].dtype == np.float32
    assert s["mask"].shape == (32, 32)
    assert s["valid"].shape == (32, 32)
    assert s["tile_id"] == scene["ids"][0]
    assert len(ds) == 2


def test_random_crop_is_reproducible_per_epoch(scene: dict, io_cfg: IOConfig) -> None:
    kw = dict(
        modality="sar",
        normalizer=_norm(scene),
        masks_dir=scene["root"] / "masks",
        io_cfg=io_cfg,
        crop=32,
        train=True,
        hflip=True,
        align_orientation=False,
        seed=3,
    )
    a, b = SN6Tiles(_records(scene), **kw), SN6Tiles(_records(scene), **kw)
    np.testing.assert_array_equal(a[0]["image"], b[0]["image"])
    assert a[0]["offset"] == b[0]["offset"]
    seen = set()
    for epoch in range(8):
        a.set_epoch(epoch)
        seen.add((a[0]["offset"], a[0]["hflipped"]))
    assert len(seen) > 1  # the draw changes with the epoch
    c = a[0]
    assert 0 <= c["offset"][0] <= 32
    assert 0 <= c["offset"][1] <= 32


def test_eval_pads_to_a_multiple_of_32_and_marks_padding(scene: dict, io_cfg: IOConfig) -> None:
    ds = SN6Tiles(
        _records(scene),
        "fused",
        _norm(scene),
        scene["root"] / "masks",
        io_cfg,
        crop=None,
        train=False,
        hflip=False,
        align_orientation=False,
        seed=0,
    )
    s = ds[0]
    assert s["image"].shape == (8, 64, 64)  # 64 is already a multiple of 32
    assert not s["valid"][:, :8].any()  # nodata columns
    assert s["valid"][:, 8:].all()
    assert s["mask"][8:20, 40:56].all()


def test_flips_move_image_mask_and_valid_together(scene: dict, io_cfg: IOConfig) -> None:
    common = dict(
        normalizer=_norm(scene),
        masks_dir=scene["root"] / "masks",
        io_cfg=io_cfg,
        crop=None,
        train=False,
        seed=0,
    )
    plain = SN6Tiles(_records(scene), "sar", hflip=False, align_orientation=False, **common)
    aligned = SN6Tiles(_records(scene), "sar", hflip=False, align_orientation=True, **common)
    # strip A (orientation 0) unchanged, strip B (orientation 1) flipped north-south
    np.testing.assert_array_equal(aligned[0]["mask"], plain[0]["mask"])
    np.testing.assert_array_equal(aligned[1]["mask"], plain[1]["mask"][::-1])
    np.testing.assert_array_equal(aligned[1]["image"], plain[1]["image"][:, ::-1])
    np.testing.assert_array_equal(aligned[1]["valid"], plain[1]["valid"][::-1])
    assert aligned[1]["flipped_ns"]
    assert not aligned[0]["flipped_ns"]
