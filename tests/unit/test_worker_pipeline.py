from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
import rasterio.windows
from shapely.geometry import box

from geoinfer.io.config import IOConfig
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.service.db.schema import Scene
from geoinfer.service.worker.main import scene_inputs
from geoinfer.service.worker.pipeline import (
    SceneInputs,
    aoi_window,
    needs_flip,
    segment_aoi,
    to_crs,
)
from geoinfer.service.worker.predictor import OnnxPredictor, sigmoid
from geoinfer.tiling.config import TilingConfig, load_tiling_config
from geoinfer.train.dataset import Normalizer
from tests.conftest import CRS, TRANSFORM, MakeTiff

STATS = {
    "optical": {"p_lo": [100.0] * 4, "p_hi": [2000.0] * 4, "mean": [0.3] * 4, "std": [0.2] * 4},
    "sar": {
        "enl": {"value": 2.0},
        "arms": {
            "db": {"p_lo": [15.0] * 4, "p_hi": [45.0] * 4, "mean": [30.0] * 4, "std": [5.0] * 4}
        },
    },
}
ARMS = {"db": {"lee": None, "domain": "db", "clip": True}}
RECTS = [(50, 90, 60, 120), (150, 200, 250, 330)]  # rows, cols: 600 m2 and 1000 m2


class Pointwise:
    """``sigmoid(gain * x[:, 0])``; records every batch it sees."""

    def __init__(self, gain: float = 4.0) -> None:
        self.gain, self.batches = gain, []

    def __call__(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.batches.append(x.copy())
        return sigmoid(self.gain * x[:, 0])


def _norm() -> Normalizer:
    return Normalizer(STATS, None, "global", "db", ARMS)


@pytest.fixture
def tcfg(tmp_path: Path) -> TilingConfig:
    return replace(load_tiling_config(), tile=128, overlap=32, scratch_dir=tmp_path / "scratch")


@pytest.fixture
def optical(make_tiff: MakeTiff) -> Path:
    a = np.full((4, 300, 400), 200, np.uint16)
    for r0, r1, c0, c1 in RECTS:
        a[:, r0:r1, c0:c1] = 1500
    return make_tiff(a, name="opt.tif")


def _wgs84(bounds: tuple[float, float, float, float]) -> Any:
    return to_crs(box(*bounds), CRS, "EPSG:4326")


def _run(
    scene: SceneInputs,
    aoi: Any,
    tcfg: TilingConfig,
    io_cfg: IOConfig,
    out: Path,
    predictor: Callable[..., Any] | None = None,
    max_px: int = 10**8,
) -> Any:
    return segment_aoi(
        scene,
        aoi,
        predictor or Pointwise(),
        _norm(),
        0.5,
        tcfg,
        load_postprocess_config(),
        io_cfg,
        out,
        batch_size=3,
        max_aoi_px=max_px,
    )


def test_needs_flip() -> None:
    assert needs_flip("sar", 1)
    assert needs_flip("fused", 1)
    assert not needs_flip("sar", 0)
    assert not needs_flip("optical", 1)
    with pytest.raises(ValueError, match="orientation"):
        needs_flip("sar", None)


def test_aoi_window_rounds_outward_and_clips() -> None:
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    win = aoi_window(box(x0 + 10.2, y0 - 20.0, x0 + 30.0, y0 - 5.1), TRANSFORM, 400, 300)
    assert (win.col_off, win.row_off, win.width, win.height) == (20, 10, 40, 30)
    clipped = aoi_window(box(x0 - 50, y0 - 20, x0 + 10, y0 + 50), TRANSFORM, 400, 300)
    assert (clipped.col_off, clipped.row_off, clipped.width, clipped.height) == (0, 0, 20, 40)
    with pytest.raises(ValueError, match="outside"):
        aoi_window(box(x0 - 50, y0 + 10, x0 - 10, y0 + 50), TRANSFORM, 400, 300)


def test_segment_full_scene_finds_both_buildings(
    optical: Path, tcfg: TilingConfig, io_cfg: IOConfig, tmp_path: Path
) -> None:
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    aoi = _wgs84((x0, y0 - 150, x0 + 200, y0))
    res = _run(SceneInputs("optical", optical, None, None), aoi, tcfg, io_cfg, tmp_path / "out")
    gdf = res.buildings
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == 2
    assert sorted(np.round(gdf["area_m2"])) == [600, 1000]
    assert (gdf["score"] > 0.9).all()
    assert res.n_tiles > 0
    assert res.timings["total_s"] >= res.timings["infer_s"] >= 0
    assert res.geojson_path.exists()
    with rasterio.open(res.mask_path) as m:
        assert m.crs == rasterio.crs.CRS.from_string(CRS)
        win = aoi_window(to_crs(aoi, "EPSG:4326", CRS), TRANSFORM, 400, 300)
        assert m.transform == rasterio.windows.transform(win, TRANSFORM)
        mask = m.read(1)
    r0, r1, c0, c1 = RECTS[0]
    sub = mask[r0 - win.row_off : r1 - win.row_off, c0 - win.col_off : c1 - win.col_off]
    assert (sub == 1).all()
    assert set(np.unique(mask)) <= {0, 1, 255}


def test_segment_partial_aoi_keeps_only_buildings_inside(
    optical: Path, tcfg: TilingConfig, io_cfg: IOConfig, tmp_path: Path
) -> None:
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    aoi = _wgs84((x0 + 20, y0 - 60, x0 + 80, y0 - 15))  # around the first rectangle only
    res = _run(SceneInputs("optical", optical, None, None), aoi, tcfg, io_cfg, tmp_path / "o")
    assert len(res.buildings) == 1
    with rasterio.open(res.mask_path) as m:
        assert m.width < 400
        assert m.height < 300
        assert m.nodata == 255


def test_segment_rejects_large_aoi(
    optical: Path, tcfg: TilingConfig, io_cfg: IOConfig, tmp_path: Path
) -> None:
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    aoi = _wgs84((x0, y0 - 150, x0 + 200, y0))
    with pytest.raises(ValueError, match="AOI"):
        _run(SceneInputs("optical", optical, None, None), aoi, tcfg, io_cfg, tmp_path, max_px=1000)


def test_segment_without_valid_pixels_writes_empty_results(
    make_tiff: MakeTiff, tcfg: TilingConfig, io_cfg: IOConfig, tmp_path: Path
) -> None:
    path = make_tiff(np.zeros((4, 100, 100), np.uint16), name="empty.tif")
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    pred = Pointwise()
    res = _run(
        SceneInputs("optical", path, None, None),
        _wgs84((x0, y0 - 50, x0 + 50, y0)),
        tcfg,
        io_cfg,
        tmp_path / "e",
        predictor=pred,
    )
    assert len(res.buildings) == 0
    assert res.n_tiles == 0
    assert not pred.batches
    assert res.geojson_path.exists()
    assert res.mask_path.exists()


def test_orientation_flip_feeds_flipped_tiles_and_unflips_predictions(
    make_tiff: MakeTiff, tcfg: TilingConfig, io_cfg: IOConfig, tmp_path: Path
) -> None:
    rng = np.random.default_rng(1)
    sar = (20 + 5 * rng.random((4, 128, 160))).astype(np.float32)
    sar[:, 20:60, 30:90] = 42.0  # a bright asymmetric block
    path = make_tiff(sar, name="sar.tif")
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    aoi = _wgs84((x0, y0 - 64, x0 + 80, y0))
    p0, p1 = Pointwise(), Pointwise()
    r0 = _run(SceneInputs("sar", None, path, 0), aoi, tcfg, io_cfg, tmp_path / "a", predictor=p0)
    r1 = _run(SceneInputs("sar", None, path, 1), aoi, tcfg, io_cfg, tmp_path / "b", predictor=p1)
    for b0, b1 in zip(p0.batches, p1.batches, strict=True):
        np.testing.assert_array_equal(b1, b0[:, :, ::-1, :])
    with rasterio.open(r0.mask_path) as a, rasterio.open(r1.mask_path) as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))
    assert len(r0.buildings) == len(r1.buildings) == 1


def test_fused_input_stacks_optical_then_sar(
    make_tiff: MakeTiff, tcfg: TilingConfig, io_cfg: IOConfig, tmp_path: Path
) -> None:
    opt = make_tiff(np.full((4, 100, 100), 1500, np.uint16), name="o.tif")
    sar = make_tiff(np.full((4, 100, 100), 30.0, np.float32), name="s.tif")
    x0, y0 = TRANSFORM.c, TRANSFORM.f
    pred = Pointwise()
    _run(
        SceneInputs("fused", opt, sar, 0),
        _wgs84((x0, y0 - 50, x0 + 50, y0)),
        tcfg,
        io_cfg,
        tmp_path / "f",
        predictor=pred,
    )
    x = pred.batches[0]
    assert x.shape[1:] == (8, 128, 128)
    assert x.dtype == np.float32
    assert x.flags.c_contiguous
    assert x[0, 0, 70, 70] > 1.0
    assert abs(x[0, 4, 70, 70]) < 1e-6


def test_onnx_predictor_matches_numpy(tiny_onnx: Callable[..., Path]) -> None:
    path = tiny_onnx(channels=4, gain=2.0, bias=-1.0)
    pred = OnnxPredictor(path, ("NoSuchProvider", "CPUExecutionProvider"))
    assert pred.providers == ["CPUExecutionProvider"]
    assert pred.in_channels == 4
    x = np.random.default_rng(0).standard_normal((3, 4, 64, 96)).astype(np.float32)
    np.testing.assert_allclose(pred(x), sigmoid(2.0 * x[:, 0] - 1.0), atol=1e-6)


def test_scene_inputs_follow_the_model_mode(tmp_path: Path) -> None:
    fused = Scene(id="f", path="o.tif", sar_path="s.tif", sensor="fused", orientation=1)
    assert scene_inputs(fused, "optical", tmp_path).sar is None
    sar = scene_inputs(fused, "sar", tmp_path)
    assert (sar.optical, sar.sar, sar.orientation) == (None, tmp_path / "s.tif", 1)
    f = scene_inputs(fused, "fused", tmp_path)
    assert (f.optical, f.sar) == (tmp_path / "o.tif", tmp_path / "s.tif")
    only_sar = Scene(id="s", path="s.tif", sensor="sar", orientation=0)
    assert scene_inputs(only_sar, "sar", tmp_path).sar == tmp_path / "s.tif"
    with pytest.raises(ValueError, match="cannot serve"):
        scene_inputs(only_sar, "optical", tmp_path)


@pytest.mark.gpu
def test_onnx_predictor_runs_on_cuda(tiny_onnx: Callable[..., Path]) -> None:
    pred = OnnxPredictor(tiny_onnx(), ("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert pred.providers[0] == "CUDAExecutionProvider"
    x = np.ones((2, 4, 64, 64), np.float32)
    np.testing.assert_allclose(pred(x), sigmoid(np.full((2, 64, 64), 4.0)), atol=1e-5)
