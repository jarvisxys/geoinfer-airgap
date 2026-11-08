from __future__ import annotations

import math

import numpy as np
import pytest

from geoinfer.eval.seams import border_band, label_edges, seam_lines, seam_score
from geoinfer.tiling.tiler import tile_grid


def test_seam_lines_without_overlap_are_tile_edges() -> None:
    rows, cols = seam_lines(300, 200, tile=64, overlap=0)
    np.testing.assert_array_equal(rows, [64, 128, 192, 256])
    np.testing.assert_array_equal(cols, [64, 128, 192])


def test_seam_lines_with_overlap_include_tile_and_crop_edges() -> None:
    rows, _ = seam_lines(200, 200, tile=64, overlap=16)
    # grid starts at -8, stride 48: tiles at -8, 40, 88, 136, 184
    tile_edges = {s.y0 for s in tile_grid(200, 200, 64, 16)} | {
        s.y0 + 64 for s in tile_grid(200, 200, 64, 16)
    }
    crop_edges = {k * 48 for k in range(1, 5)}  # crops partition at multiples of the stride
    expected = sorted(v for v in tile_edges | crop_edges if 0 < v < 200)
    np.testing.assert_array_equal(rows, expected)


def test_border_band_covers_pixels_within_width() -> None:
    band = border_band((100, 80), np.array([50]), np.array([], dtype=int), width=4)
    assert band.sum() == 8 * 80  # rows 46..53
    assert band[46:54].all()
    assert not band[45].any()
    assert not band[54].any()
    both = border_band((100, 80), np.array([50]), np.array([40]), width=4)
    assert both.sum() == 8 * 80 + 8 * 100 - 8 * 8


def test_constant_map_scores_one() -> None:
    rows, cols = seam_lines(128, 128, 32, 0)
    assert seam_score(np.full((128, 128), 0.3), rows, cols) == 1.0


def test_seamless_noise_scores_about_one() -> None:
    p = np.random.default_rng(0).random((512, 512))
    rows, cols = seam_lines(512, 512, 64, 0)
    assert seam_score(p, rows, cols) == pytest.approx(1.0, abs=0.05)


def test_per_tile_offsets_raise_the_score() -> None:
    rng = np.random.default_rng(1)
    p = 0.5 + 0.02 * rng.standard_normal((256, 256))
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            p[y : y + 64, x : x + 64] += rng.uniform(-0.1, 0.1)
    rows, cols = seam_lines(256, 256, 64, 0)
    assert seam_score(p, rows, cols) > 2.0


def test_excluded_and_invalid_pixels_are_not_counted() -> None:
    rng = np.random.default_rng(2)
    p = rng.random((128, 128))
    p[64:, :] += 5.0  # a step on row line 64
    rows, cols = np.array([64]), np.array([], dtype=int)
    assert seam_score(p, rows, cols) > 5
    excl = np.zeros((128, 128), bool)
    excl[60:68, :] = True
    assert math.isnan(seam_score(p, rows, cols, exclude=excl))  # the line is excluded
    cols2 = np.array([30])  # a seamless column line still scores about 1
    assert seam_score(p, np.array([], dtype=int), cols2, exclude=excl) == pytest.approx(
        1.0, abs=0.2
    )
    valid = np.ones((128, 128), bool)
    valid[64, :] = False
    assert math.isnan(seam_score(p, rows, cols, valid=valid))  # no valid pair on the line


def test_label_edges_marks_pixels_near_label_changes() -> None:
    lab = np.zeros((20, 20), np.uint16)
    lab[:, 10:] = 3
    e = label_edges(lab, radius=2)
    assert e[:, 8:12].all()
    assert not e[:, :8].any()
    assert not e[:, 12:].any()
    assert not label_edges(np.ones((5, 5), np.uint16), 2).any()
