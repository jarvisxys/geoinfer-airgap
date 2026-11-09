from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from geoinfer.bench.gpu import GpuSample, GpuSampler
from geoinfer.bench.timing import km2_per_min, summarize, time_batches


def test_time_batches_warms_up_then_times_each_batch() -> None:
    calls: list[int] = []

    def fn(x: np.ndarray) -> np.ndarray:
        calls.append(len(x))
        time.sleep(0.001)
        return x

    tiles = np.zeros((10, 1, 4, 4), np.float32)
    lat = time_batches(fn, tiles, batch=4, warmup=3, n_batches=5)
    assert len(calls) == 8  # 3 warm-up + 5 timed
    assert set(calls) == {4}  # batches wrap around the tile pool, always full
    assert lat.shape == (5,)
    assert (lat >= 0.001).all()


def test_summarize_reports_rates_and_percentiles() -> None:
    reps = [np.full(10, 0.1), np.full(10, 0.2)]  # two repeats of 10 batches of 4 tiles
    s = summarize(reps, batch=4, tile=512, overlap=64, gsd_m=0.5)
    assert s["tiles_per_s"] == pytest.approx(30.0)  # mean of 40 and 20
    assert s["tiles_per_s_std"] == pytest.approx(10.0)
    assert s["p50_ms"] == pytest.approx(150.0)
    assert s["p95_ms"] == pytest.approx(200.0)
    assert s["km2_per_min"] == pytest.approx(km2_per_min(30.0, 512, 64, 0.5), rel=1e-4)


def test_km2_per_min_counts_the_tile_stride_area() -> None:
    # 448 x 448 px of new ground per tile at 0.5 m = 0.050176 km^2
    assert km2_per_min(1.0, 512, 64, 0.5) == pytest.approx(0.050176 * 60)


def test_gpu_sampler_summarises_samples() -> None:
    seq = itertools.cycle(
        [GpuSample(50.0, 1000.0, 1500.0, 60.0), GpuSample(100.0, 3000.0, 1200.0, 70.0)]
    )
    with GpuSampler(read=lambda: next(seq), hz=200) as s:
        time.sleep(0.1)
    out = s.summary()
    assert out["samples"] >= 4
    assert out["util_mean"] == pytest.approx(75.0, abs=15)
    assert out["mem_peak_mib"] == 3000.0
    assert out["clock_min_mhz"] == 1200.0
    assert out["temp_max_c"] == 70.0


def test_gpu_sampler_without_samples_reports_none() -> None:
    s = GpuSampler(read=lambda: None, hz=50)
    with s:
        time.sleep(0.05)
    assert s.summary() == {"samples": 0}


def test_fp16_conversion_keeps_float32_io(tiny_onnx: Callable[..., Path], tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from geoinfer.bench.engines import fp16_model

    src = tiny_onnx(channels=4, gain=2.0)
    dst = fp16_model(src, tmp_path / "fp16.onnx")
    m = onnx.load(dst)
    assert m.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert m.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert any(i.data_type == onnx.TensorProto.FLOAT16 for i in m.graph.initializer)
    assert fp16_model(src, dst) == dst  # cached


@pytest.mark.gpu
def test_ort_fp16_engine_matches_fp32(tiny_onnx: Callable[..., Path], tmp_path: Path) -> None:
    from geoinfer.bench.engines import make_engine

    src = tiny_onnx(channels=4, gain=2.0, bias=-0.5)
    x = np.random.default_rng(0).standard_normal((2, 4, 64, 64)).astype(np.float32)
    e32 = make_engine("ort_fp32", onnx_path=src, cache_dir=tmp_path)
    e16 = make_engine("ort_fp16", onnx_path=src, cache_dir=tmp_path)
    np.testing.assert_allclose(e16(x), e32(x), atol=2e-3)
