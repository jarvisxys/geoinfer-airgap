from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

from geoinfer.io.cog import is_cog, main, to_cog
from geoinfer.io.config import IOConfig
from geoinfer.io.raster import inspect_raster
from tests.conftest import MakeTiff

SAMPLE = Path(os.environ.get("SAMPLE", Path.home() / "geo-data" / "sample"))


@pytest.mark.parametrize(("dtype", "predictor"), [("uint16", "2"), ("float32", "3")])
def test_to_cog_is_lossless_and_keeps_georef(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path, dtype: str, predictor: str
) -> None:
    arr = np.random.default_rng(0).integers(1, 4000, (4, 300, 900)).astype(dtype)
    src = make_tiff(arr, nodata=0)
    dst = to_cog(src, tmp_path / "out" / "c.tif", io_cfg)
    a, b = inspect_raster(src), inspect_raster(dst)
    assert (b.crs, b.transform, b.width, b.height, b.count) == (
        a.crs,
        a.transform,
        a.width,
        a.height,
        a.count,
    )
    assert b.dtypes == a.dtypes
    assert b.nodata == a.nodata
    assert b.compression == "DEFLATE"
    with rasterio.open(dst) as ds:
        np.testing.assert_array_equal(ds.read(), arr)
        assert ds.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"] == predictor


def test_to_cog_tiles_strip_input_and_builds_overviews(
    make_tiff: MakeTiff, io_cfg: IOConfig, tmp_path: Path
) -> None:
    src = make_tiff(np.ones((1, 2048, 2048), dtype=np.uint8))
    assert not inspect_raster(src).tiled
    info = inspect_raster(to_cog(src, tmp_path / "c.tif", io_cfg))
    assert info.tiled
    assert info.block_shape == (512, 512)
    assert info.overviews >= 2
    assert info.is_cog
    assert is_cog(tmp_path / "c.tif")


def test_is_cog_false_for_plain_tiff(make_tiff: MakeTiff) -> None:
    assert not is_cog(make_tiff(np.ones((1, 16, 16), dtype=np.uint8)))


def test_to_cog_refuses_read_only_root(make_tiff: MakeTiff, io_cfg: IOConfig) -> None:
    src = make_tiff(np.ones((1, 16, 16), dtype=np.uint8))
    with pytest.raises(PermissionError):
        to_cog(src, io_cfg.read_only_roots[0] / "c.tif", io_cfg)


def test_to_cog_refuses_in_place(make_tiff: MakeTiff, io_cfg: IOConfig) -> None:
    src = make_tiff(np.ones((1, 16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="itself"):
        to_cog(src, src, io_cfg)


def test_to_cog_leaves_nothing_on_failure(io_cfg: IOConfig, tmp_path: Path) -> None:
    bad = tmp_path / "not_a_raster.tif"
    bad.write_text("nope")
    out_dir = tmp_path / "out"
    with pytest.raises(rasterio.errors.RasterioError):
        to_cog(bad, out_dir / "c.tif", io_cfg)
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_cli_converts(make_tiff: MakeTiff, tmp_path: Path) -> None:
    src = make_tiff(np.ones((1, 64, 64), dtype=np.uint8))
    cfg = tmp_path / "io.yaml"
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "io.yaml"
    cfg.write_text(repo_cfg.read_text().replace("/mnt/d/geo/raw", str(tmp_path / "raw")))
    main([str(src), str(tmp_path / "c.tif"), "--config", str(cfg)])
    assert is_cog(tmp_path / "c.tif")


@pytest.mark.data
def test_sample_sar_tile_converts(io_cfg: IOConfig, tmp_path: Path) -> None:
    tiles = sorted(SAMPLE.rglob("SAR-Intensity/*.tif"))
    if not tiles:
        pytest.skip(f"no SAR tiles under {SAMPLE}")
    src = tiles[0]
    dst = to_cog(src, tmp_path / "sar_cog.tif", io_cfg)
    a, b = inspect_raster(src), inspect_raster(dst)
    assert (b.crs, b.transform, b.nodata, b.dtypes) == (a.crs, a.transform, a.nodata, a.dtypes)
    assert b.is_cog
    with rasterio.open(src) as s, rasterio.open(dst) as d:
        np.testing.assert_array_equal(s.read(), d.read())  # 900 px tile: small enough
