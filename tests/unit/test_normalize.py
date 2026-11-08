from __future__ import annotations

import numpy as np
import pytest

from geoinfer.preprocess.normalize import minmax_per_tile, percentile_stretch, zscore


def _valid(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def test_minmax_maps_each_band_to_unit_range() -> None:
    x = np.stack([np.arange(16).reshape(4, 4), 100 + 2 * np.arange(16).reshape(4, 4)])
    out = minmax_per_tile(x.astype(np.uint16), _valid(4, 4))
    assert out.dtype == np.float32
    for b in out:
        assert b.min() == 0.0
        assert b.max() == 1.0
    np.testing.assert_allclose(out[0], out[1])


def test_minmax_ignores_invalid_pixels() -> None:
    x = np.arange(16, dtype=np.uint16).reshape(1, 4, 4)
    x[0, 0, 0] = 60_000  # nodata-like outlier
    valid = _valid(4, 4)
    valid[0, 0] = False
    out = minmax_per_tile(x, valid)
    assert out[0, 0, 0] == 0.0
    assert out[0, 0, 1] == 0.0  # 1 is the smallest valid value
    assert out[0, 3, 3] == 1.0


def test_minmax_constant_band_is_zero() -> None:
    out = minmax_per_tile(np.full((2, 3, 3), 7, dtype=np.uint16), _valid(3, 3))
    assert np.all(out == 0.0)


def test_percentile_stretch_clips_and_scales_per_band() -> None:
    x = np.array([[[0, 10, 20, 30]], [[0, 100, 200, 300]]], dtype=np.float32)
    out = percentile_stretch(x, _valid(1, 4), lo=np.array([10.0, 0.0]), hi=np.array([20.0, 200.0]))
    np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(out[1, 0], [0.0, 0.5, 1.0, 1.0])


def test_percentile_stretch_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="hi"):
        percentile_stretch(
            np.zeros((1, 2, 2)), _valid(2, 2), lo=np.array([5.0]), hi=np.array([5.0])
        )


def test_zscore_and_invalid_to_zero() -> None:
    x = np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    out = zscore(x, valid, mean=np.array([3.0]), std=np.array([2.0]))
    np.testing.assert_allclose(out[0], [[-1.0, 0.0], [1.0, 0.0]])


def test_zscore_rejects_zero_std() -> None:
    with pytest.raises(ValueError, match="std"):
        zscore(np.zeros((1, 2, 2)), _valid(2, 2), mean=np.array([0.0]), std=np.array([0.0]))
