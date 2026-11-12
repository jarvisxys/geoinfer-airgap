from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geoinfer.io.config import IOConfig
from geoinfer.service.api.imagery import (
    inspect_image,
    lonlat_to_px,
    overlay_rings,
    prepare_image,
    px_to_lonlat,
    pyramid,
    safe_upload_name,
    stretch_stats,
    tile_png,
    utm_epsg,
)
from geoinfer.service.worker.pipeline import to_crs
from tests.conftest import CRS, TRANSFORM, MakeTiff

TRAIN = {"p_lo": [100.0] * 4, "p_hi": [2000.0] * 4}


def _write(path: Path, arr: np.ndarray, crs: str | None, transform: object, nodata=None) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as d:
        d.write(arr)
    return path


@pytest.fixture
def optical(make_tiff: MakeTiff) -> Path:
    a = np.full((4, 400, 600), 300, np.uint16)
    a[:, 100:200, 300:400] = 1800  # bright block: rows 100-200, cols 300-400
    return make_tiff(a, name="opt.tif")


def _decode(png: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None
    return img


def test_pyramid_levels_cover_the_scene() -> None:
    p = pyramid(600, 400, 256)
    assert p["max_zoom"] == 2  # 600 px needs 4 tiles of 256 at full resolution -> 2 levels down
    assert p["tile"] == 256
    assert pyramid(200, 100, 256)["max_zoom"] == 0


def test_tile_png_full_resolution_shows_the_bright_block(optical: Path) -> None:
    stats = stretch_stats(optical, "optical")
    img = _decode(tile_png(optical, 2, 1, 0, "optical", stats))  # cols 256-511, rows 0-255
    assert img.shape == (256, 256, 4)
    block = img[100:200, 300 - 256 : 400 - 256]
    rest = img[10:50, 160:200]
    assert block[..., :3].mean() > rest[..., :3].mean() + 100
    assert (img[..., 3] == 255).all()


def test_tile_png_outside_the_scene_is_transparent(optical: Path) -> None:
    stats = stretch_stats(optical, "optical")
    img = _decode(tile_png(optical, 2, 2, 0, "optical", stats))  # cols 512-767: 512-599 inside
    assert (img[:, :88, 3] == 255).all()
    assert (img[:, 100:, 3] == 0).all()
    whole = _decode(tile_png(optical, 0, 0, 0, "optical", stats))  # 600 x 400 scaled by 1/4
    assert (whole[:100, :150, 3] == 255).all()
    assert (whole[110:, 160:, 3] == 0).all()


def test_tile_png_sar_is_grey(make_tiff: MakeTiff) -> None:
    sar = make_tiff(
        (20 + 10 * np.random.default_rng(0).random((4, 64, 64))).astype(np.float32), name="sar.tif"
    )
    img = _decode(tile_png(sar, 0, 0, 0, "sar", stretch_stats(sar, "sar")))
    rgb = img[:64, :64, :3]
    assert (rgb[..., 0] == rgb[..., 1]).all()


def test_pixel_and_lonlat_round_trip() -> None:
    ring = [(0.0, 0.0), (600.0, 0.0), (600.0, 400.0), (0.0, 400.0)]
    ll = px_to_lonlat(ring, TRANSFORM, CRS)
    assert 4.3 < ll[0][0] < 4.4
    assert 51.9 < ll[0][1] < 52.0
    back = lonlat_to_px(ll, TRANSFORM, CRS)
    np.testing.assert_allclose(back, ring, atol=1e-6)


def test_utm_epsg() -> None:
    assert utm_epsg(4.4, 51.9) == 32631
    assert utm_epsg(77.2, 28.6) == 32643
    assert utm_epsg(151.2, -33.9) == 32756


def test_inspect_good_optical(optical: Path) -> None:
    r = inspect_image(optical, TRAIN)
    assert r["ok"] is True
    assert r["suggested_sensor"] == "optical"
    assert (r["width"], r["height"], r["bands"], r["gsd_m"]) == (600, 400, 4, 0.5)
    assert all(c["level"] in ("pass", "info") for c in r["checks"])  # info: not a COG


def test_inspect_reports_problems_in_plain_words(tmp_path: Path) -> None:
    rgb = _write(tmp_path / "rgb.tif", np.ones((3, 50, 50), np.uint8), CRS, TRANSFORM)
    r = inspect_image(rgb, TRAIN)
    assert r["ok"] is False
    assert any("4 bands" in c["message"] and c["level"] == "fail" for c in r["checks"])
    geo = _write(
        tmp_path / "geo.tif",
        np.ones((4, 50, 50), np.uint16),
        "EPSG:4326",
        from_origin(4.38, 51.92, 0.00001, 0.00001),
    )
    r = inspect_image(geo, TRAIN)
    assert any(c["id"] == "crs" and c["level"] == "fail" and c["fixable"] for c in r["checks"])
    coarse = _write(
        tmp_path / "coarse.tif",
        np.ones((4, 50, 50), np.uint16),
        CRS,
        from_origin(594000, 5757000, 2.0, 2.0),
    )
    r = inspect_image(coarse, TRAIN)
    assert any(c["id"] == "gsd" and c["level"] == "warn" and c["fixable"] for c in r["checks"])
    nocrs = _write(
        tmp_path / "nocrs.tif", np.ones((4, 20, 20), np.uint16), None, from_origin(0, 0, 1, 1)
    )
    r = inspect_image(nocrs, TRAIN)
    assert any(c["id"] == "crs" and c["level"] == "fail" and not c["fixable"] for c in r["checks"])


def test_prepare_reprojects_to_utm_at_half_metre(tmp_path: Path, io_cfg: IOConfig) -> None:
    a = np.zeros((4, 60, 80), np.uint16)
    a[:, 20:40, 30:50] = 1500
    src = _write(
        tmp_path / "geo.tif", a, "EPSG:4326", from_origin(4.38, 51.92, 0.00001, 0.00001), nodata=0
    )  # ~0.7 x 1.1 m pixels
    out = prepare_image(src, tmp_path / "geo_prepared.tif", 0.5, io_cfg)
    with rasterio.open(out) as r:
        assert r.crs.to_epsg() == 32631
        assert r.res == pytest.approx((0.5, 0.5))
        assert r.count == 4
        data = r.read()
    assert data.max() == 1500
    assert inspect_image(out, TRAIN)["ok"] is True


def test_safe_upload_name() -> None:
    assert safe_upload_name("My City (2024).TIF") == "My_City_2024.tif"
    assert safe_upload_name("../../etc/passwd.tif") == "passwd.tif"
    with pytest.raises(ValueError, match="GeoTIFF"):
        safe_upload_name("photo.jpg")
    with pytest.raises(ValueError, match="name"):
        safe_upload_name(".tif")


def test_overlay_rings_in_scene_pixels() -> None:
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    poly = to_crs(box(x0 + 10, y0 - 20, x0 + 30, y0 - 10), CRS, "EPSG:4326")
    feats = [
        {
            "geometry": poly.__geo_interface__,
            "properties": {"id": 1, "area_m2": 200.0, "score": 0.9},
        }
    ]
    rings = overlay_rings(feats, TRANSFORM, CRS)
    assert len(rings) == 1
    xs = [p[0] for p in rings[0]["rings"][0]]
    ys = [p[1] for p in rings[0]["rings"][0]]
    assert min(xs) == pytest.approx(20, abs=0.5)
    assert max(xs) == pytest.approx(60, abs=0.5)
    assert min(ys) == pytest.approx(20, abs=0.5)
    assert max(ys) == pytest.approx(40, abs=0.5)
    assert rings[0]["area_m2"] == 200.0
