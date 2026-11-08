from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from geoinfer.postprocess.config import PostprocessConfig
from geoinfer.postprocess.vectorize import (
    export_geojson,
    polygonize,
    simplify,
    to_geodataframe,
    vectorize_mask_blockwise,
    vectorize_raster,
)
from tests.conftest import CRS, TRANSFORM, MakeTiff

T = from_origin(500_000.0, 5_750_000.0, 0.5, 0.5)
CFG = PostprocessConfig(
    threshold=0.5,
    opening=3,
    closing=3,
    min_area_m2=20.0,
    connectivity=4,
    simplify_tolerance_m=0.5,
    block=64,
    geojson_crs="EPSG:4326",
)


def test_polygonize_square_area_and_position() -> None:
    m = np.zeros((20, 20), dtype=bool)
    m[2:6, 3:10] = True  # 4 x 7 px
    (poly,) = polygonize(m, T, connectivity=4)
    assert poly.area == pytest.approx(28 * 0.25)
    assert poly.bounds == pytest.approx((500_001.5, 5_749_997.0, 500_005.0, 5_749_999.0))


def test_polygonize_keeps_holes() -> None:
    m = np.zeros((20, 20), dtype=bool)
    m[2:12, 2:12] = True
    m[5:8, 5:8] = False
    (poly,) = polygonize(m, T, connectivity=4)
    assert len(poly.interiors) == 1
    assert poly.area == pytest.approx((100 - 9) * 0.25)


def test_polygonize_connectivity_for_diagonal_pixels() -> None:
    m = np.zeros((10, 10), dtype=bool)
    m[2:4, 2:4] = True
    m[4:6, 4:6] = True
    assert len(polygonize(m, T, connectivity=4)) == 2
    assert len(polygonize(m, T, connectivity=8)) == 1


def test_geodataframe_keeps_the_raster_crs() -> None:
    m = np.zeros((10, 10), dtype=bool)
    m[1:4, 1:4] = True
    gdf = to_geodataframe(polygonize(m, T, 4), CRS)
    assert gdf.crs.to_string() == CRS
    assert list(gdf.columns) == ["id", "area_m2", "geometry"]
    assert gdf.area_m2.iloc[0] == pytest.approx(9 * 0.25)
    empty = to_geodataframe([], CRS)
    assert len(empty) == 0
    assert empty.crs.to_string() == CRS


def test_simplify_reduces_vertices_and_keeps_shape() -> None:
    # 1-px staircase edge: many vertices, almost a triangle
    m = np.tril(np.ones((40, 40), dtype=bool))
    (poly,) = polygonize(m, T, 4)
    (simple,) = simplify([poly], tolerance=0.5)
    assert len(simple.exterior.coords) < len(poly.exterior.coords) / 5
    assert simple.is_valid
    assert abs(simple.area - poly.area) < 0.5 * poly.length  # within tolerance x perimeter
    assert simplify([Polygon()], 0.5) == []


@pytest.mark.parametrize("block", [13, 32, 50])
@pytest.mark.parametrize("seed", [0, 1])
def test_blockwise_vectorizing_equals_whole(block: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m = np.zeros((150, 130), dtype=bool)
    for _ in range(40):
        y, x = rng.integers(0, 140), rng.integers(0, 120)
        h, w = rng.integers(1, 30, size=2)
        m[y : y + h, x : x + w] = True
    m[rng.random(m.shape) < 0.02] ^= True  # holes, specks, diagonal contacts
    whole = polygonize(m, T, 4)
    blocks = vectorize_mask_blockwise(m, T, block, connectivity=4)
    assert len(blocks) == len(whole)
    assert sum(p.area for p in blocks) == pytest.approx(sum(p.area for p in whole))
    a = gpd.GeoSeries(whole).union_all()
    b = gpd.GeoSeries(blocks).union_all()
    assert a.symmetric_difference(b).area == pytest.approx(0.0, abs=1e-6)
    assert sorted(round(p.area, 3) for p in blocks) == sorted(round(p.area, 3) for p in whole)


def test_blockwise_refuses_8_connectivity() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        vectorize_mask_blockwise(np.zeros((4, 4), bool), T, 2, connectivity=8)


def test_export_geojson_in_wgs84_and_native(tmp_path: Path) -> None:
    m = np.zeros((40, 40), dtype=bool)
    m[5:25, 5:15] = True
    gdf = to_geodataframe(polygonize(m, TRANSFORM, 4), CRS)
    out = export_geojson(gdf, tmp_path / "b.geojson", "EPSG:4326")
    raw = json.loads(out.read_text())
    lon, lat = raw["features"][0]["geometry"]["coordinates"][0][0]
    assert 0 < lon < 10  # Rotterdam, lon/lat order
    assert 50 < lat < 55
    back = gpd.read_file(out).to_crs(CRS)
    assert back.geometry.area.iloc[0] == pytest.approx(gdf.geometry.area.iloc[0], rel=1e-6)
    native = gpd.read_file(export_geojson(gdf, tmp_path / "n.geojson", None))
    assert native.crs.to_string() == CRS
    assert native.geometry.area.iloc[0] == pytest.approx(200 * 0.25)


def test_vectorize_raster_reads_mask_cog_and_ignores_nodata(make_tiff: MakeTiff) -> None:
    m = np.zeros((1, 90, 70), dtype=np.uint8)
    m[0, 10:30, 10:30] = 1
    m[0, 50:60, 40:60] = 1
    m[0, 70:90, :] = 255  # no prediction
    path = make_tiff(m, name="mask.tif", nodata=255)
    gdf = vectorize_raster(path, replace(CFG, block=32, simplify_tolerance_m=0.0))
    assert gdf.crs.to_string() == CRS
    assert sorted(gdf.area_m2) == pytest.approx([200 * 0.25, 400 * 0.25])
    assert list(gdf.id) == [1, 2]
