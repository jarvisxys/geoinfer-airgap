from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from rasterio.windows import Window

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import RasterReader, inspect_raster
from geoinfer.postprocess.config import PostprocessConfig, load_postprocess_config
from geoinfer.postprocess.morphology import (
    binarize,
    clean_mask,
    clean_mask_blockwise,
    clean_raster,
    closing,
    halo_px,
    label_components,
    min_area_px,
    opening,
    remove_small,
)
from tests.conftest import CRS, REPO_ROOT, TRANSFORM, MakeTiff

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


def _square(n: int, y: int, x: int, s: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[y : y + s, x : x + s] = True
    return m


def test_binarize_thresholds_and_drops_invalid_and_nan() -> None:
    p = np.array([[0.2, 0.5, 0.9], [np.nan, 0.7, 0.51]], dtype=np.float32)
    valid = np.array([[True, True, True], [True, False, True]])
    np.testing.assert_array_equal(
        binarize(p, 0.5, valid), [[False, True, True], [False, False, True]]
    )
    assert binarize(p, 0.5).dtype == np.bool_


def test_opening_removes_specks_and_thin_lines_but_keeps_blocks() -> None:
    m = _square(40, 10, 10, 6)
    m[2, 30] = True  # single pixel
    m[30, 2:20] = True  # 1-px line
    out = opening(m, 3)
    np.testing.assert_array_equal(out, _square(40, 10, 10, 6))
    np.testing.assert_array_equal(opening(out, 3), out)  # idempotent


def test_closing_fills_small_holes_and_is_idempotent() -> None:
    m = _square(40, 10, 10, 8)
    m[13, 13] = False
    m[15, 14:16] = False
    out = closing(m, 3)
    np.testing.assert_array_equal(out, _square(40, 10, 10, 8))
    np.testing.assert_array_equal(closing(out, 3), out)


def test_morphology_keeps_objects_on_the_scene_edge() -> None:
    m = _square(20, 0, 0, 5)  # touches the top-left corner
    np.testing.assert_array_equal(opening(m, 3), m)
    np.testing.assert_array_equal(closing(m, 3), m)


@pytest.mark.parametrize("fn", [opening, closing])
def test_morphology_edge_cases(fn: Callable[[np.ndarray, int], np.ndarray]) -> None:
    empty, full = np.zeros((9, 9), bool), np.ones((9, 9), bool)
    dot = _square(9, 2, 2, 1)
    np.testing.assert_array_equal(fn(empty, 3), empty)
    np.testing.assert_array_equal(fn(full, 3), full)
    np.testing.assert_array_equal(fn(dot, 1), dot)  # size 1 is a no-op
    with pytest.raises(ValueError, match="odd"):
        fn(empty, 4)


def test_min_area_px_follows_the_pixel_size() -> None:
    assert min_area_px(from_origin(0, 0, 0.5, 0.5), 20.0) == 80
    assert min_area_px(from_origin(0, 0, 1.0, 1.0), 20.0) == 20
    assert min_area_px(from_origin(0, 0, 0.3, 0.3), 1.0) == 12  # ceil(11.1)


def test_remove_small_drops_exactly_the_components_below_the_limit() -> None:
    m = np.zeros((30, 30), dtype=bool)
    m[1:9, 1:11] = True  # 80 px: kept
    m[15:21, 15:28] = True  # 78 px ... set to 79 below
    m[21, 15] = True  # 79 px: dropped
    out = remove_small(m, 80, connectivity=4)
    assert out[1:9, 1:11].all()
    assert not out[15:22, 15:28].any()


def test_connectivity_decides_whether_diagonal_pixels_join() -> None:
    m = np.zeros((10, 10), dtype=bool)
    m[2:4, 2:4] = True
    m[4:6, 4:6] = True  # touches the first block only at a corner
    n4 = label_components(m, 4)[1]
    n8 = label_components(m, 8)[1]
    assert (n4, n8) == (2, 1)
    assert remove_small(m, 5, connectivity=4).sum() == 0  # two 4-px parts
    assert remove_small(m, 5, connectivity=8).sum() == 8  # one 8-px part
    with pytest.raises(ValueError, match="connectivity"):
        label_components(m, 6)


def test_clean_mask_runs_the_chain() -> None:
    p = np.zeros((60, 60), dtype=np.float32)
    p[10:30, 10:30] = 0.9  # building (400 px)
    p[18, 18] = 0.1  # roof hole
    p[50, 50] = 0.95  # speck
    p[40:44, 40:44] = 0.8  # 16 px blob, below 80 px
    out = clean_mask(p, None, CFG, from_origin(0, 0, 0.5, 0.5))
    np.testing.assert_array_equal(out, _square(60, 10, 10, 20))


def test_clean_mask_never_marks_invalid_pixels_as_building() -> None:
    # closing dilates first; it must not grow buildings into nodata
    p = np.zeros((40, 40), dtype=np.float32)
    p[5:35, 5:20] = 0.9
    p[5:35, 21:30] = 0.9
    valid = np.ones((40, 40), dtype=bool)
    valid[:, 20] = False  # 1-px nodata column between the two halves
    p[:, 20] = np.nan
    out = clean_mask(p, valid, CFG, from_origin(0, 0, 0.5, 0.5))
    assert not out[:, 20].any()
    assert out[5:35, 5:20].all()
    assert out[5:35, 21:30].all()


@pytest.mark.parametrize("block", [17, 64, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blockwise_cleaning_equals_whole_array(block: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.random((230, 190)).astype(np.float32)
    # blobs of many sizes plus speckle, so components straddle block edges
    p = np.zeros_like(noise)
    for _ in range(60):
        y, x = rng.integers(0, 220), rng.integers(0, 180)
        h, w = rng.integers(2, 25, size=2)
        p[y : y + h, x : x + w] = 0.9
    p = np.where(noise < 0.03, 1.0 - p, p)
    valid = rng.random(p.shape) > 0.01
    t = from_origin(0, 0, 0.5, 0.5)
    whole = clean_mask(p, valid, CFG, t)
    blocks = clean_mask_blockwise(p, valid, replace(CFG, block=block), t)
    np.testing.assert_array_equal(blocks, whole)


def test_blockwise_without_halo_would_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    # guards the equality test above: with no halo, block edges change the result
    import geoinfer.postprocess.morphology as morph

    p = np.zeros((120, 120), dtype=np.float32)
    p[30:40, 59:69] = 0.9  # 100 px building, split 50 / 50 by the block edge at col 64
    monkeypatch.setattr(morph, "halo_px", lambda cfg, t: 0)
    t = from_origin(0, 0, 0.5, 0.5)
    whole = clean_mask(p, None, CFG, t)
    assert whole.sum() == 100
    assert clean_mask_blockwise(p, None, CFG, t).sum() == 0  # both halves < 80 px


def test_halo_covers_morphology_and_component_reach() -> None:
    assert halo_px(CFG, from_origin(0, 0, 0.5, 0.5)) == 2 * 1 + 2 * 1 + 80


def test_clean_raster_keeps_crs_transform_and_marks_nodata(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    p = np.zeros((1, 150, 120), dtype=np.float32)
    p[0, 20:60, 20:60] = 0.9
    p[0, 100:110, 0:10] = np.nan  # no prediction
    src = make_tiff(p, name="prob.tif", nodata=float("nan"))
    dst = tmp_path / "out" / "mask.tif"
    clean_raster(src, dst, CFG, io_cfg)
    info = inspect_raster(dst)
    assert info.crs.to_string() == CRS
    assert info.transform == TRANSFORM
    assert info.dtypes == ("uint8",)
    assert info.nodata == 255
    with RasterReader(dst) as r:
        got = r.read(Window(0, 0, 120, 150))
    expected = clean_mask(p[0], ~np.isnan(p[0]), CFG, TRANSFORM).astype(np.uint8)
    expected[100:110, 0:10] = 255
    np.testing.assert_array_equal(got.data[0], expected)


def test_postprocess_config_loads_and_validates() -> None:
    cfg = load_postprocess_config(REPO_ROOT / "configs" / "postprocess.yaml")
    assert cfg.connectivity == 4
    assert cfg.min_area_m2 == 20
    assert cfg.geojson_crs == "EPSG:4326"
    with pytest.raises(ValueError, match="connectivity"):
        replace(cfg, connectivity=6).validate()
    with pytest.raises(ValueError, match="odd"):
        replace(cfg, opening=2).validate()
    with pytest.raises(ValueError, match="threshold"):
        replace(cfg, threshold=1.5).validate()
