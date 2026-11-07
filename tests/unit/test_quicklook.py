from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from geoinfer.io.quicklook import percentile_stretch, read_decimated, render_quicklook
from tests.conftest import MakeTiff


def test_stretch_maps_percentiles_to_full_range() -> None:
    band = np.linspace(0, 1000, 10_000, dtype=np.float32).reshape(100, 100)
    out = percentile_stretch(band, np.ones_like(band, dtype=bool), 0, 100)
    assert out.min() == 0
    assert out.max() == 255


def test_stretch_constant_band_is_mid_grey() -> None:
    band = np.full((8, 8), 7.0, dtype=np.float32)
    out = percentile_stretch(band, np.ones_like(band, dtype=bool))
    assert np.all(out == 128)


def test_stretch_keeps_nodata_black() -> None:
    band = np.arange(64, dtype=np.float32).reshape(8, 8)
    valid = np.ones_like(band, dtype=bool)
    valid[0, :] = False
    out = percentile_stretch(band, valid)
    assert np.all(out[0, :] == 0)


def test_read_decimated_caps_longest_side(make_tiff: MakeTiff) -> None:
    arr = np.random.default_rng(0).integers(0, 2000, (4, 300, 600), dtype=np.uint16)
    data, valid = read_decimated(make_tiff(arr), bands=[1, 2, 3], max_size=100)
    assert data.shape == (3, 50, 100)
    assert valid.shape == (50, 100)
    assert data.dtype == np.float32


def test_read_decimated_flags_nodata(make_tiff: MakeTiff) -> None:
    arr = np.ones((1, 10, 10), dtype=np.float32)
    arr[:, :, :5] = 0
    _, valid = read_decimated(make_tiff(arr, nodata=0), max_size=10)
    assert not valid[:, :5].any()
    assert valid[:, 5:].all()


@pytest.mark.parametrize(("bands", "ndim"), [([1], 2), ([3, 2, 1], 3)])
def test_render_writes_png(
    make_tiff: MakeTiff, tmp_path: Path, bands: list[int], ndim: int
) -> None:
    arr = np.random.default_rng(1).integers(1, 2000, (4, 64, 64), dtype=np.uint16)
    out = tmp_path / "sub" / "ql.png"
    img = render_quicklook(make_tiff(arr), out, bands=bands, max_size=32)
    assert img.ndim == ndim
    assert img.shape[:2] == (32, 32)
    assert cv2.imread(str(out)) is not None


def test_render_sar_db_ignores_non_positive(make_tiff: MakeTiff, tmp_path: Path) -> None:
    arr = np.random.default_rng(2).uniform(0.01, 1.0, (1, 32, 32)).astype(np.float32)
    arr[0, :4, :] = 0.0  # zero intensity = SAR nodata in SpaceNet 6
    img = render_quicklook(make_tiff(arr), tmp_path / "sar.png", sar_db=True, max_size=32)
    assert np.all(img[:4, :] == 0)
    assert img[4:, :].max() == 255
