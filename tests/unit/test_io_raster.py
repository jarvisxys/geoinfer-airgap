from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import (
    NotGeoreferencedError,
    RasterReader,
    RasterWriter,
    assert_writable,
    inspect_raster,
    read_window,
)
from tests.conftest import CRS, TRANSFORM, MakeTiff


def _ramp(count: int, h: int, w: int, dtype: str = "uint16") -> np.ndarray:
    return (np.arange(count * h * w) % 60_000 + 1).reshape(count, h, w).astype(dtype)


# --- inspect_raster -------------------------------------------------------------------


def test_inspect_reports_strip_tiff(make_tiff: MakeTiff) -> None:
    info = inspect_raster(make_tiff(_ramp(4, 70, 90), nodata=0))
    assert (info.width, info.height, info.count) == (90, 70, 4)
    assert info.dtypes == ("uint16",) * 4
    assert info.nodata == 0
    assert info.crs == rasterio.crs.CRS.from_string(CRS)
    assert info.transform == TRANSFORM
    assert not info.tiled
    assert info.overviews == 0
    assert info.compression is None
    assert not info.is_cog


def test_inspect_rejects_missing_crs(tmp_path: Path) -> None:
    path = tmp_path / "nocrs.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint8", transform=TRANSFORM
    ) as dst:
        dst.write(np.zeros((1, 8, 8), dtype=np.uint8))
    with pytest.raises(NotGeoreferencedError, match="CRS"):
        inspect_raster(path)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_inspect_rejects_missing_transform(tmp_path: Path) -> None:
    path = tmp_path / "notr.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint8", crs=CRS
    ) as dst:
        dst.write(np.zeros((1, 8, 8), dtype=np.uint8))
    with pytest.raises(NotGeoreferencedError, match="transform"):
        inspect_raster(path)


# --- assert_writable ------------------------------------------------------------------


def test_assert_writable_blocks_read_only_root(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(PermissionError):
        assert_writable(root / "sub" / "x.tif", (root,))
    with pytest.raises(PermissionError):
        assert_writable(root, (root,))


def test_assert_writable_allows_sibling_with_shared_prefix(tmp_path: Path) -> None:
    assert_writable(tmp_path / "raw2" / "x.tif", (tmp_path / "raw",))


def test_assert_writable_follows_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(PermissionError):
        assert_writable(link / "x.tif", (root,))


# --- RasterReader ---------------------------------------------------------------------


def test_reader_window_matches_slice_and_transform(make_tiff: MakeTiff) -> None:
    arr = _ramp(4, 100, 120)
    win = Window(col_off=30, row_off=20, width=50, height=40)
    with RasterReader(make_tiff(arr)) as rd:
        out = rd.read(win)
    np.testing.assert_array_equal(out.data, arr[:, 20:60, 30:80])
    assert out.transform == window_transform(win, TRANSFORM)
    assert out.valid.shape == (40, 50)
    assert out.valid.all()


def test_reader_band_subset(make_tiff: MakeTiff) -> None:
    arr = _ramp(4, 32, 32)
    with RasterReader(make_tiff(arr)) as rd:
        out = rd.read(Window(0, 0, 16, 16), bands=[3, 1])
    np.testing.assert_array_equal(out.data, arr[[2, 0], :16, :16])


def test_reader_valid_mask_is_false_only_where_all_bands_nodata(make_tiff: MakeTiff) -> None:
    arr = _ramp(2, 16, 16)
    arr[:, :4, :] = 0  # all bands nodata
    arr[0, 8, :] = 0  # one band only: still valid
    with RasterReader(make_tiff(arr, nodata=0)) as rd:
        out = rd.read(Window(0, 0, 16, 16))
    assert not out.valid[:4].any()
    assert out.valid[4:].all()


def test_reader_rejects_out_of_bounds_window(make_tiff: MakeTiff) -> None:
    with (
        RasterReader(make_tiff(_ramp(1, 32, 32))) as rd,
        pytest.raises(ValueError, match="outside"),
    ):
        rd.read(Window(20, 20, 16, 16))


@pytest.mark.parametrize("nodata", [0, None])
def test_reader_boundless_fills_and_masks_outside(make_tiff: MakeTiff, nodata: int | None) -> None:
    arr = _ramp(2, 32, 32)
    win = Window(col_off=-8, row_off=-4, width=16, height=16)
    with RasterReader(make_tiff(arr, nodata=nodata)) as rd:
        out = rd.read(win, boundless=True)
    assert out.data.shape == (2, 16, 16)
    np.testing.assert_array_equal(out.data[:, 4:, 8:], arr[:, :12, :8])
    assert np.all(out.data[:, :4, :] == 0)
    assert not out.valid[:4, :].any()
    assert not out.valid[:, :8].any()
    assert out.valid[4:, 8:].all()
    assert out.transform == window_transform(win, TRANSFORM)


def test_reader_refuses_oversized_window(make_tiff: MakeTiff) -> None:
    with (
        RasterReader(make_tiff(_ramp(1, 64, 64)), max_window_px=1000) as rd,
        pytest.raises(ValueError, match="max_window_px"),
    ):
        rd.read(Window(0, 0, 64, 64))


def test_reader_rejects_fractional_window(make_tiff: MakeTiff) -> None:
    with (
        RasterReader(make_tiff(_ramp(1, 32, 32))) as rd,
        pytest.raises(ValueError, match="integer"),
    ):
        rd.read(Window(0.5, 0, 8, 8))


def test_reader_requires_context(make_tiff: MakeTiff) -> None:
    rd = RasterReader(make_tiff(_ramp(1, 8, 8)))
    with pytest.raises(RuntimeError):
        rd.read(Window(0, 0, 4, 4))


def test_read_window_convenience(make_tiff: MakeTiff) -> None:
    arr = _ramp(1, 32, 32)
    out = read_window(make_tiff(arr), Window(4, 4, 8, 8))
    np.testing.assert_array_equal(out.data, arr[:, 4:12, 4:12])


# --- RasterWriter ---------------------------------------------------------------------


def _write_grid(wr: RasterWriter, arr: np.ndarray, step: int) -> None:
    _, h, w = arr.shape
    for r in range(0, h, step):
        for c in range(0, w, step):
            win = Window(c, r, min(step, w - c), min(step, h - r))
            wr.write(arr[:, r : r + int(win.height), c : c + int(win.width)], win)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("cog", [False, True])
def test_writer_grid_roundtrip_keeps_georef(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path, dtype: str, cog: bool
) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 600, 700)))
    arr = _ramp(2, 600, 700, dtype)
    dst = tmp_path / "out" / f"{dtype}.tif"
    with RasterWriter(dst, ref, io_cfg, count=2, dtype=dtype, cog=cog) as wr:
        _write_grid(wr, arr, 256)
    info = inspect_raster(dst)
    assert info.crs == ref.crs
    assert info.transform == ref.transform
    assert (info.width, info.height, info.count) == (700, 600, 2)
    assert info.tiled
    assert info.is_cog is cog
    with rasterio.open(dst) as src:
        np.testing.assert_array_equal(src.read(), arr)


def test_writer_defaults_to_reference_count_and_dtype(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    arr = _ramp(3, 40, 40)
    ref = inspect_raster(make_tiff(arr))
    with RasterWriter(tmp_path / "o.tif", ref, io_cfg, cog=False) as wr:
        wr.write(arr, Window(0, 0, 40, 40))
    info = inspect_raster(tmp_path / "o.tif")
    assert (info.count, info.dtypes) == (3, ("uint16",) * 3)


def test_writer_accepts_2d_for_single_band(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 16, 16)))
    mask = np.eye(16, dtype=np.uint8)
    with RasterWriter(tmp_path / "m.tif", ref, io_cfg, count=1, dtype="uint8", nodata=255) as wr:
        wr.write(mask, Window(0, 0, 16, 16))
    with rasterio.open(tmp_path / "m.tif") as src:
        np.testing.assert_array_equal(src.read(1), mask)
        assert src.nodata == 255


def test_writer_overview_resampling_override(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 1100, 1100)))
    dst = tmp_path / "mask.tif"
    with RasterWriter(dst, ref, io_cfg, dtype="uint8", overview_resampling="NEAREST") as wr:
        wr.write(np.ones((1, 1100, 1100), dtype=np.uint8), Window(0, 0, 1100, 1100))
    with rasterio.open(dst) as src:
        assert src.tags(ns="IMAGE_STRUCTURE")["OVERVIEW_RESAMPLING"] == "NEAREST"


def test_writer_rejects_dtype_mismatch(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 16, 16)))
    with (
        RasterWriter(tmp_path / "o.tif", ref, io_cfg, dtype="uint8") as wr,
        pytest.raises(ValueError, match="dtype"),
    ):
        wr.write(np.zeros((1, 16, 16), dtype=np.float32), Window(0, 0, 16, 16))


def test_writer_rejects_shape_mismatch_and_out_of_bounds(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 16, 16)))
    with RasterWriter(tmp_path / "o.tif", ref, io_cfg) as wr:
        with pytest.raises(ValueError, match="shape"):
            wr.write(np.zeros((1, 8, 8), dtype=np.uint16), Window(0, 0, 16, 16))
        with pytest.raises(ValueError, match="outside"):
            wr.write(np.zeros((1, 8, 8), dtype=np.uint16), Window(12, 12, 8, 8))
        wr.write(np.zeros((1, 16, 16), dtype=np.uint16), Window(0, 0, 16, 16))


@pytest.mark.parametrize("cog", [False, True])
def test_writer_cleans_up_on_error(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path, cog: bool
) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 16, 16)))
    out_dir = tmp_path / "out"
    with (
        pytest.raises(RuntimeError, match="boom"),
        RasterWriter(out_dir / "o.tif", ref, io_cfg, cog=cog),
    ):
        raise RuntimeError("boom")
    assert list(out_dir.iterdir()) == []


def test_writer_refuses_read_only_root(make_tiff: MakeTiff, io_cfg: IOConfig) -> None:
    ref = inspect_raster(make_tiff(_ramp(1, 16, 16)))
    with pytest.raises(PermissionError):
        RasterWriter(io_cfg.read_only_roots[0] / "o.tif", ref, io_cfg).__enter__()


def test_reader_valid_mask_handles_nan_nodata(make_tiff: MakeTiff) -> None:
    arr = np.ones((2, 8, 8), dtype=np.float32)
    arr[:, :2, :] = np.nan
    with RasterReader(make_tiff(arr, nodata=float("nan"))) as rd:
        out = rd.read(Window(0, 0, 8, 8))
    assert not out.valid[:2].any()
    assert out.valid[2:].all()
