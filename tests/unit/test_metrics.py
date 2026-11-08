from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import box

from geoinfer.eval.metrics import (
    PixelCounts,
    filter_by_valid,
    object_scores,
    pixel_scores,
    score_rasters,
)
from tests.conftest import CRS, MakeTiff

T = from_origin(500_000.0, 5_750_000.0, 0.5, 0.5)


def _gdf(geoms: list, crs: str = CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def test_pixel_counts_by_hand() -> None:
    pred = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    gt = np.array([[1, 0, 0], [1, 1, 0]], dtype=bool)
    c = PixelCounts.from_masks(pred, gt)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 2)
    s = pixel_scores(c)
    assert s["iou"] == pytest.approx(2 / 4)
    assert s["f1"] == pytest.approx(4 / 6)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)


def test_pixel_counts_respect_valid_and_add_up() -> None:
    rng = np.random.default_rng(0)
    pred, gt = rng.random((60, 50)) > 0.5, rng.random((60, 50)) > 0.6
    valid = rng.random((60, 50)) > 0.2
    whole = PixelCounts.from_masks(pred, gt, valid)
    parts = sum(
        (
            PixelCounts.from_masks(pred[r : r + 7], gt[r : r + 7], valid[r : r + 7])
            for r in range(0, 60, 7)
        ),
        PixelCounts(),
    )
    assert parts == whole
    assert whole.tp + whole.fp + whole.fn + whole.tn == valid.sum()


def test_perfect_and_empty_predictions() -> None:
    gt = np.zeros((10, 10), dtype=bool)
    gt[2:5, 2:5] = True
    assert pixel_scores(PixelCounts.from_masks(gt, gt))["iou"] == 1.0
    empty = pixel_scores(PixelCounts.from_masks(np.zeros_like(gt), gt))
    assert empty["iou"] == 0.0
    assert empty["recall"] == 0.0
    assert math.isnan(empty["precision"])  # no positive predictions
    nothing = pixel_scores(PixelCounts.from_masks(np.zeros_like(gt), np.zeros_like(gt)))
    assert math.isnan(nothing["iou"])


def test_score_rasters_blockwise_with_nodata(make_tiff: MakeTiff) -> None:
    rng = np.random.default_rng(1)
    pred = (rng.random((1, 70, 90)) > 0.5).astype(np.uint8)
    pred[0, :5, :] = 255  # no prediction
    gt = (rng.random((1, 70, 90)) > 0.5).astype(np.uint32) * rng.integers(1, 9, (1, 70, 90))
    p = make_tiff(pred, name="pred.tif", nodata=255)
    g = make_tiff(gt.astype(np.uint32), name="gt.tif")
    c = score_rasters(p, g, block=16)
    valid = pred[0] != 255
    assert c == PixelCounts.from_masks(pred[0] == 1, gt[0] > 0, valid)


def test_score_rasters_rejects_other_grids(make_tiff: MakeTiff, tmp_path: Path) -> None:
    a = make_tiff(np.zeros((1, 10, 10), np.uint8), name="a.tif")
    b = make_tiff(np.zeros((1, 10, 11), np.uint8), name="b.tif")
    with pytest.raises(ValueError, match="grid"):
        score_rasters(a, b)


def test_object_f1_identical_polygons() -> None:
    gt = _gdf([box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 8)])
    s = object_scores(gt, gt, min_area_m2=20)
    assert (s.tp, s.fp, s.fn) == (3, 0, 0)
    assert s.f1 == 1.0
    assert s.ious == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.parametrize(("dx", "matched"), [(3.0, True), (3.4, False)])
def test_object_match_needs_iou_of_one_half(dx: float, matched: bool) -> None:
    # 10 x 10 boxes shifted by dx: IoU = (10 - dx) / (10 + dx); 0.5 at dx = 10/3
    gt = _gdf([box(0, 0, 10, 10)])
    pred = _gdf([translate(box(0, 0, 10, 10), dx)])
    s = object_scores(pred, gt, min_area_m2=20)
    assert s.tp == int(matched)
    assert (s.fp, s.fn) == ((0, 0) if matched else (1, 1))


def test_one_prediction_matches_at_most_one_building() -> None:
    gt = _gdf([box(0, 0, 10, 10), box(10, 0, 20, 10)])
    pred = _gdf([box(0, 0, 20, 10)])  # covers both: IoU 0.5 with each
    s = object_scores(pred, gt, min_area_m2=20)
    assert (s.tp, s.fp, s.fn) == (1, 0, 1)


def test_greedy_matching_prefers_the_best_pair() -> None:
    gt = _gdf([box(0, 0, 10, 10)])
    pred = _gdf([translate(box(0, 0, 10, 10), 2.0), translate(box(0, 0, 10, 10), 1.0)])
    s = object_scores(pred, gt, min_area_m2=20)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)
    assert s.ious == pytest.approx([9 / 11])


def test_small_buildings_are_ignored_on_both_sides() -> None:
    gt = _gdf([box(0, 0, 10, 10), box(50, 50, 54, 54)])  # 100 m2 and 16 m2
    pred = _gdf([box(0, 0, 10, 10), box(80, 80, 84, 84)])  # 16 m2 false positive
    s = object_scores(pred, gt, min_area_m2=20)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)


def test_empty_sides() -> None:
    gt = _gdf([box(0, 0, 10, 10)])
    empty = _gdf([])
    s = object_scores(empty, gt, min_area_m2=20)
    assert (s.tp, s.fp, s.fn, s.recall, s.f1) == (0, 0, 1, 0.0, 0.0)
    assert math.isnan(s.precision)
    s2 = object_scores(gt, empty, min_area_m2=20)
    assert (s2.tp, s2.fp, s2.fn, s2.precision) == (0, 1, 0, 0.0)


def test_object_scores_reject_crs_mismatch_and_geographic_crs() -> None:
    gt = _gdf([box(0, 0, 10, 10)])
    with pytest.raises(ValueError, match="CRS"):
        object_scores(_gdf([box(0, 0, 10, 10)], crs="EPSG:32632"), gt, min_area_m2=20)
    with pytest.raises(ValueError, match="projected"):
        object_scores(
            _gdf([box(0, 0, 1, 1)], "EPSG:4326"), _gdf([box(0, 0, 1, 1)], "EPSG:4326"), 20
        )


def test_filter_by_valid_uses_the_centroid() -> None:
    valid = np.zeros((20, 20), dtype=bool)
    valid[:, :10] = True  # western half valid
    x0, y1 = T.c, T.f
    inside = box(x0 + 1, y1 - 3, x0 + 3, y1 - 1)  # cols 2-6
    straddle_in = box(x0 + 3, y1 - 3, x0 + 5.5, y1 - 1)  # centroid col 8.5 -> valid
    outside = box(x0 + 6, y1 - 3, x0 + 9, y1 - 1)  # centroid col 15 -> invalid
    off_grid = box(x0 - 50, y1 - 3, x0 - 40, y1 - 1)
    kept = filter_by_valid(_gdf([inside, straddle_in, outside, off_grid]), valid, T)
    assert len(kept) == 2
    assert kept.geometry.iloc[1].equals(straddle_in)
