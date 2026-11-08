from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.windows import Window

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import RasterReader, inspect_raster
from geoinfer.tiling.stitcher import Stitcher, blend_weights
from geoinfer.tiling.tiler import TileSpec, tile_grid
from tests.conftest import CRS, TRANSFORM, MakeTiff

STRATEGIES = ["naive", "crop", "hann", "tukey"]
GRIDS = [(64, 0), (64, 16), (128, 32), (96, 48)]
SIZES = [(200, 150), (64, 64), (37, 101)]
# naive stitching needs overlap 0
CASES = [(s, t, o) for s in STRATEGIES for t, o in GRIDS if not (s == "naive" and o)]


def _tile_of(scene: np.ndarray, spec: TileSpec, pad: int) -> np.ndarray:
    """Reflect-padded tile cut from a scene already padded by ``pad`` on every side."""
    y, x = spec.y0 + pad, spec.x0 + pad
    return scene[..., y : y + spec.tile, x : x + spec.tile]


def _stitch(scene: np.ndarray, tile: int, overlap: int, strategy: str, **kw: object) -> Stitcher:
    h, w = scene.shape[-2:]
    pad = 2 * tile
    big = np.pad(scene, [(0, 0)] * (scene.ndim - 2) + [(pad, pad), (pad, pad)], mode="reflect")
    st = Stitcher(
        h, w, tile, overlap, strategy, bands=1 if scene.ndim == 2 else scene.shape[0], **kw
    )  # type: ignore[arg-type]
    for spec in tile_grid(h, w, tile, overlap):
        st.add(spec, _tile_of(big, spec, pad))
    return st


def test_hann_weights_match_the_plan_formula() -> None:
    t = 32
    w = blend_weights(t, 8, "hann")
    x = np.arange(t)
    one = np.sin(np.pi * (x + 0.5) / t) ** 2
    assert w.dtype == np.float32
    np.testing.assert_allclose(w, np.outer(one, one), rtol=1e-6)
    np.testing.assert_allclose(w, w.T)
    np.testing.assert_allclose(w, w[::-1, ::-1], rtol=1e-6)


def test_crop_and_naive_weights() -> None:
    crop = blend_weights(32, 8, "crop")
    assert crop.sum() == 24 * 24
    assert np.all(crop[4:28, 4:28] == 1)
    assert np.all(blend_weights(32, 0, "naive") == 1)
    with pytest.raises(ValueError, match="naive"):
        blend_weights(32, 8, "naive")
    with pytest.raises(ValueError, match="strategy"):
        blend_weights(32, 8, "median")


@pytest.mark.parametrize(("tile", "overlap"), [(64, 16), (128, 32), (32, 2)])
def test_tukey_ramps_sum_to_one_across_the_overlap(tile: int, overlap: int) -> None:
    w = blend_weights(tile, overlap, "tukey")[tile // 2]  # one row
    assert np.all(w[overlap : tile - overlap] == 1)
    stride = tile - overlap
    np.testing.assert_allclose(w[stride:] + w[:overlap], 1.0, rtol=1e-6)
    assert np.all(blend_weights(tile, 0, "tukey") == 1)


@pytest.mark.parametrize(("strategy", "tile", "overlap"), CASES)
@pytest.mark.parametrize(("h", "w"), SIZES)
def test_constant_image_is_reproduced(
    strategy: str, tile: int, overlap: int, h: int, w: int
) -> None:
    st = _stitch(np.full((h, w), 0.37, dtype=np.float32), tile, overlap, strategy)
    out = st.read(Window(0, 0, w, h))
    assert out.shape == (1, h, w)
    assert np.all(st.weight_sum() > 0)  # every pixel covered
    assert np.max(np.abs(out - 0.37)) < 1e-6


@pytest.mark.parametrize(("strategy", "tile", "overlap"), CASES)
def test_identity_model_reproduces_a_random_image(strategy: str, tile: int, overlap: int) -> None:
    scene = np.random.default_rng(1).random((2, 150, 170)).astype(np.float32)
    out = _stitch(scene, tile, overlap, strategy).read(Window(0, 0, 170, 150))
    np.testing.assert_allclose(out, scene, atol=1e-6)


def test_invalid_pixels_get_no_weight() -> None:
    h = w = 100
    scene = np.full((h, w), 0.5, dtype=np.float32)
    bad = np.zeros((h, w), dtype=bool)
    bad[40:60, 40:60] = True
    pad = 128
    big = np.pad(scene, pad, mode="reflect")
    big_bad = np.pad(bad, pad, mode="reflect")
    st = Stitcher(h, w, 64, 16, "hann")
    for spec in tile_grid(h, w, 64, 16):
        tile_vals = _tile_of(big, spec, pad).copy()
        tile_bad = _tile_of(big_bad, spec, pad)
        tile_vals[tile_bad] = 99.0  # garbage where invalid
        st.add(spec, tile_vals, valid=~tile_bad)
    out = st.read(Window(0, 0, w, h))[0]
    assert np.all(np.isnan(out[bad]))
    assert np.max(np.abs(out[~bad] - 0.5)) < 1e-6


def test_memmap_matches_ram_and_is_cleaned_up(tmp_path: Path) -> None:
    scene = np.random.default_rng(2).random((90, 130)).astype(np.float32)
    ram = _stitch(scene, 64, 16, "hann")
    with _stitch(scene, 64, 16, "hann", memmap_min_px=1, scratch_dir=tmp_path) as mm:
        assert mm.uses_memmap
        assert any(tmp_path.iterdir())
        np.testing.assert_array_equal(
            mm.read(Window(0, 0, 130, 90)), ram.read(Window(0, 0, 130, 90))
        )
    assert not any(tmp_path.iterdir())
    assert not ram.uses_memmap


def test_add_rejects_wrong_tile_shape() -> None:
    st = Stitcher(50, 50, 32, 0, "naive")
    spec = tile_grid(50, 50, 32, 0)[0]
    with pytest.raises(ValueError, match="shape"):
        st.add(spec, np.zeros((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        st.add(spec, np.zeros((32, 32), dtype=np.float32), valid=np.ones((16, 16), dtype=bool))


def test_write_keeps_crs_transform_and_marks_uncovered_nodata(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    h, w = 120, 90
    ref = make_tiff(np.zeros((1, h, w), dtype=np.uint8))
    scene = np.random.default_rng(3).random((h, w)).astype(np.float32)
    pad = 128
    big = np.pad(scene, pad, mode="reflect")
    st = Stitcher(h, w, 64, 16, "hann")
    for spec in tile_grid(h, w, 64, 16):
        if spec.row == 0 and spec.col == 0:
            continue  # pretend this tile was skipped as nodata
        st.add(spec, _tile_of(big, spec, pad))
    dst = tmp_path / "out" / "prob.tif"
    st.write(dst, inspect_raster(ref), io_cfg, nodata=float("nan"))
    info = inspect_raster(dst)
    assert info.crs.to_string() == CRS
    assert info.transform == TRANSFORM
    assert (info.width, info.height, info.count, info.dtypes[0]) == (w, h, 1, "float32")
    with RasterReader(dst) as r:
        got = r.read(Window(0, 0, w, h))
    covered = st.weight_sum() > 0
    assert not covered[:10, :10].any()  # only the skipped tile covered the corner
    np.testing.assert_allclose(got.data[0][covered], scene[covered], atol=1e-6)
    assert np.all(np.isnan(got.data[0][~covered]))
    np.testing.assert_array_equal(got.valid, covered)
