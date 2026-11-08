from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

from geoinfer.tiling.mosaic import paste_first_valid, paste_offset, snap_grid


def test_snap_grid_covers_bounds_on_the_lattice() -> None:
    transform, width, height = snap_grid((10.3, 20.1, 15.2, 23.9), res=0.5)
    assert transform == from_origin(10.0, 24.0, 0.5, 0.5)
    assert (width, height) == (11, 8)  # x 10.0..15.5, y 20.0..24.0
    with pytest.raises(ValueError, match="bounds"):
        snap_grid((1.0, 0.0, 1.0, 2.0), res=0.5)


@pytest.mark.parametrize(
    ("x0", "y0", "row", "col"),
    [(12.0, 22.0, 4, 4), (12.12, 21.9, 4, 4), (12.24, 22.3, 3, 4), (11.76, 21.76, 4, 4)],
)
def test_paste_offset_rounds_to_nearest_pixel(x0: float, y0: float, row: int, col: int) -> None:
    dst = from_origin(10.0, 24.0, 0.5, 0.5)
    r, c, ry, rx = paste_offset(from_origin(x0, y0, 0.5, 0.5), dst)
    assert (r, c) == (row, col)
    assert abs(ry) <= 0.5
    assert abs(rx) <= 0.5
    assert c + rx == pytest.approx((x0 - 10.0) / 0.5)
    assert r + ry == pytest.approx((24.0 - y0) / 0.5)


def test_paste_offset_rejects_other_resolution_or_rotation() -> None:
    dst = from_origin(0.0, 10.0, 0.5, 0.5)
    with pytest.raises(ValueError, match="resolution"):
        paste_offset(from_origin(0.0, 10.0, 1.0, 1.0), dst)
    with pytest.raises(ValueError, match="rotation"):
        paste_offset(Affine(0.5, 0.1, 0.0, 0.0, -0.5, 10.0), dst)


def test_paste_first_valid_keeps_earlier_pixels_and_clips() -> None:
    dst = np.zeros((2, 6, 6), dtype=np.float32)
    filled = np.zeros((6, 6), dtype=bool)
    a = np.full((2, 4, 4), 1.0, dtype=np.float32)
    va = np.ones((4, 4), dtype=bool)
    va[0, 0] = False
    new_a = paste_first_valid(dst, filled, a, va, row=0, col=0)
    assert new_a.sum() == 15
    b = np.full((2, 4, 4), 2.0, dtype=np.float32)
    new_b = paste_first_valid(dst, filled, b, np.ones((4, 4), dtype=bool), row=2, col=4)
    # b does not overlap a (a ends at col 3) and is clipped to cols 4-5
    assert new_b.sum() == 4 * 2
    assert dst[0, 0, 0] == 0  # invalid in a, never filled
    assert np.all(dst[:, 1:4, 1:4] == 1)
    assert np.all(dst[:, 2:6, 4:6] == 2)
    c = np.full((2, 3, 3), 3.0, dtype=np.float32)
    new_c = paste_first_valid(dst, filled, c, np.ones((3, 3), dtype=bool), row=-1, col=-1)
    assert new_c.sum() == 1  # only (0, 0) was still empty
    assert dst[0, 0, 0] == 3
    assert filled.sum() == 15 + 8 + 1


def test_paste_first_valid_outside_destination_is_a_no_op() -> None:
    dst = np.zeros((1, 4, 4), dtype=np.uint16)
    filled = np.zeros((4, 4), dtype=bool)
    src = np.ones((1, 2, 2), dtype=np.uint16)
    new = paste_first_valid(dst, filled, src, np.ones((2, 2), dtype=bool), row=10, col=0)
    assert not new.any()
    assert new.shape == (4, 4)
    assert not filled.any()


def test_paste_first_valid_checks_shapes() -> None:
    dst = np.zeros((1, 4, 4), dtype=np.uint16)
    filled = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        paste_first_valid(
            dst, filled, np.ones((2, 2, 2), dtype=np.uint16), np.ones((2, 2), bool), 0, 0
        )
    with pytest.raises(ValueError, match="shape"):
        paste_first_valid(
            dst, filled, np.ones((1, 2, 2), dtype=np.uint16), np.ones((3, 2), bool), 0, 0
        )
