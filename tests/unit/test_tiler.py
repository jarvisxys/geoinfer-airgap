from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from geoinfer.io.raster import RasterReader
from geoinfer.tiling.config import load_tiling_config
from geoinfer.tiling.tiler import iter_tiles, read_tile, reflect_index, tile_grid
from tests.conftest import REPO_ROOT, MakeTiff

SIZES = [(1000, 700), (512, 512), (100, 37), (257, 1)]


def _coverage(h: int, w: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Count how often each scene pixel falls in a (y0, x0, y1, x1) box, clipped."""
    count = np.zeros((h, w), dtype=np.int32)
    for y0, x0, y1, x1 in boxes:
        count[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)] += 1
    return count


@pytest.mark.parametrize(("h", "w"), SIZES)
def test_grid_without_overlap_covers_every_pixel_exactly_once(h: int, w: int) -> None:
    specs = tile_grid(h, w, tile=256, overlap=0)
    boxes = [(s.y0, s.x0, s.y0 + s.tile, s.x0 + s.tile) for s in specs]
    assert np.all(_coverage(h, w, boxes) == 1)


@pytest.mark.parametrize(("h", "w"), SIZES)
@pytest.mark.parametrize("overlap", [32, 64, 128])
def test_centre_crops_partition_the_scene(h: int, w: int, overlap: int) -> None:
    specs = tile_grid(h, w, tile=256, overlap=overlap)
    boxes = []
    for s in specs:
        c0, c1 = s.crop
        boxes.append((s.y0 + c0, s.x0 + c0, s.y0 + c1, s.x0 + c1))
    assert np.all(_coverage(h, w, boxes) == 1)
    full = [(s.y0, s.x0, s.y0 + s.tile, s.x0 + s.tile) for s in specs]
    assert np.all(_coverage(h, w, full) >= 1)


def test_grid_starts_half_an_overlap_outside_and_steps_by_stride() -> None:
    specs = tile_grid(600, 600, tile=256, overlap=64)
    ys = sorted({s.y0 for s in specs})
    assert ys[0] == -32
    assert np.all(np.diff(ys) == 192)
    assert specs[0].crop == (32, 224)
    assert {(s.row, s.col) for s in specs} == {(r, c) for r in range(4) for c in range(4)}


@pytest.mark.parametrize(
    ("tile", "overlap", "match"),
    [(0, 0, "tile"), (256, 256, "overlap"), (256, -2, "overlap"), (256, 33, "even")],
)
def test_grid_rejects_bad_parameters(tile: int, overlap: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tile_grid(100, 100, tile=tile, overlap=overlap)


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_reflect_index_matches_numpy_pad(n: int) -> None:
    ref = np.pad(np.arange(n), 40, mode="reflect") if n > 1 else np.zeros(n + 80, dtype=int)
    got = reflect_index(np.arange(-40, n + 40), n)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize(("h", "w"), [(300, 200), (40, 70)])
def test_read_tile_reflects_the_scene_at_its_edges(make_tiff: MakeTiff, h: int, w: int) -> None:
    rng = np.random.default_rng(0)
    scene = rng.integers(1, 1000, size=(2, h, w)).astype(np.uint16)
    path = make_tiff(scene)
    big = np.pad(scene, ((0, 0), (300, 300), (300, 300)), mode="reflect")
    with RasterReader(path) as r:
        for spec in tile_grid(h, w, tile=128, overlap=32):
            t = read_tile(r, spec)
            assert t.data.shape == (2, 128, 128)
            assert t.data.dtype == np.uint16
            ys, xs = spec.y0 + 300, spec.x0 + 300
            np.testing.assert_array_equal(t.data, big[:, ys : ys + 128, xs : xs + 128])
            inside = np.zeros((128, 128), dtype=bool)
            yy = np.arange(spec.y0, spec.y0 + 128)
            xx = np.arange(spec.x0, spec.x0 + 128)
            inside[np.ix_((yy >= 0) & (yy < h), (xx >= 0) & (xx < w))] = True
            np.testing.assert_array_equal(t.valid, inside)


def test_read_tile_marks_nodata_invalid(make_tiff: MakeTiff) -> None:
    scene = np.ones((1, 64, 64), dtype=np.float32)
    scene[0, :, :16] = 0
    with RasterReader(make_tiff(scene, nodata=0)) as r:
        t = read_tile(r, tile_grid(64, 64, tile=64, overlap=0)[0])
    assert not t.valid[:, :16].any()
    assert t.valid[:, 16:].all()


def test_iter_tiles_skips_all_nodata_tiles(make_tiff: MakeTiff) -> None:
    scene = np.zeros((1, 256, 256), dtype=np.float32)
    scene[0, 10:20, 10:20] = 5.0  # only the top-left 128 px tile has data
    with RasterReader(make_tiff(scene, nodata=0)) as r:
        kept = list(iter_tiles(r, tile=128, overlap=0, skip_nodata=True))
        every = list(iter_tiles(r, tile=128, overlap=0, skip_nodata=False))
    assert [(t.spec.row, t.spec.col) for t in kept] == [(0, 0)]
    assert len(every) == 4


def test_iter_tiles_never_reads_more_than_one_tile(make_tiff: MakeTiff) -> None:
    scene = np.arange(500 * 300, dtype=np.float32).reshape(1, 500, 300)
    with RasterReader(make_tiff(scene), max_window_px=128 * 128) as r:
        tiles = list(iter_tiles(r, tile=128, overlap=32, skip_nodata=False))
    assert len(tiles) == len(tile_grid(500, 300, tile=128, overlap=32))


def test_iter_tiles_reads_selected_bands(make_tiff: MakeTiff) -> None:
    scene = np.stack([np.full((64, 64), b, dtype=np.uint8) for b in (1, 2, 3)])
    with RasterReader(make_tiff(scene)) as r:
        (t,) = iter_tiles(r, tile=64, overlap=0, bands=[3, 1])
    assert t.data.shape == (2, 64, 64)
    assert t.data[0].max() == 3
    assert t.data[1].max() == 1


def test_tiling_config_loads_and_validates(tmp_path: Path) -> None:
    cfg = load_tiling_config(REPO_ROOT / "configs" / "tiling.yaml")
    assert cfg.overlap < cfg.tile
    assert cfg.overlap % 2 == 0
    assert cfg.pad_mode == "reflect"
    assert cfg.strategy in {"naive", "crop", "hann", "tukey"}
    with pytest.raises(ValueError, match="strategy"):
        replace(cfg, strategy="median").validate()
    with pytest.raises(ValueError, match="naive"):
        replace(cfg, strategy="naive", overlap=64).validate()


def test_read_tile_constant_mode_fills_zeros(make_tiff: MakeTiff) -> None:
    scene = np.full((1, 50, 50), 7, dtype=np.int16)
    with RasterReader(make_tiff(scene)) as r:
        spec = tile_grid(50, 50, tile=64, overlap=16)[0]  # starts at (-8, -8)
        t = read_tile(r, spec, pad_mode="constant")
        with pytest.raises(ValueError, match="pad_mode"):
            read_tile(r, spec, pad_mode="wrap")
    assert np.all(t.data[0, :8, :] == 0)
    assert np.all(t.data[0, 8:58, 8:58] == 7)
    assert np.array_equal(t.valid, t.data[0] == 7)
