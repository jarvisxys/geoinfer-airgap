from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from geoinfer.preprocess.stats import BandHistogram
from geoinfer.viz.dip_figures import (
    coreg_overlay_figure,
    histogram_figure,
    homogeneous_window,
    optical_seam_figure,
    sar_patch_figure,
    strip_calibration_figure,
    tiling_overlay_figure,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def _scene(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(200, 800, w)[None, :] + 50 * rng.standard_normal((h, w))
    return np.clip(np.stack([base] * 4), 1, 65535).astype(np.uint16)


def test_per_tile_minmax_seams_worse_than_global_stretch() -> None:
    scene = _scene(100, 160)
    a = scene[:, :, :100].copy()
    b = scene[:, :, 60:].copy()
    b[:, 10:20, 80:90] = 5000  # a bright roof only in tile b changes its min-max
    stats = {"p_lo": [150.0] * 4, "p_hi": [900.0] * 4}
    fig, m = optical_seam_figure(a, b, (0, 60), stats)
    assert isinstance(fig, Figure)
    s = m["seam_score"]
    assert s["per-tile min-max (baseline)"] > 1.5 * s["global 2-98 % stretch (train split)"]


def test_histogram_figure_reports_each_arm() -> None:
    rng = np.random.default_rng(1)
    ho = BandHistogram(4, 0, 2000, 1)
    ho.add(rng.integers(50, 1500, (4, 50, 50)), np.ones((50, 50), bool))
    hs = BandHistogram(4, 0.0, 100.0, 0.01)
    hs.add(rng.normal(30, 7, (4, 50, 50)), np.ones((50, 50), bool))
    arm = {"p_lo": [15.0] * 4, "p_hi": [45.0] * 4, "mean": [30.0] * 4, "std": [6.0] * 4}
    ostats = {"p_lo": [80.0] * 4, "p_hi": [1400.0] * 4, "mean": [0.5] * 4, "std": [0.3] * 4}
    fig, m = histogram_figure(ho, ostats, {"db": hs, "db_lee7": hs}, {"db": arm, "db_lee7": arm})
    assert isinstance(fig, Figure)
    assert set(m) == {"db", "db_lee7"}


def test_homogeneous_window_finds_the_flat_block() -> None:
    rng = np.random.default_rng(2)
    lin = rng.gamma(1.0, 1.0, (128, 128)) * (1 + 5 * rng.random((128, 128)))
    lin[64:96, 32:64] = rng.gamma(50.0, 1 / 50.0, (32, 32))  # low variation
    assert homogeneous_window(lin, np.ones((128, 128), bool), 32) == (64, 32)
    with pytest.raises(ValueError, match="valid"):
        homogeneous_window(lin, np.zeros((128, 128), bool), 32)


def test_sar_patch_figure_lee_raises_enl() -> None:
    rng = np.random.default_rng(3)
    lin = rng.gamma(2.0, 0.5, (4, 160, 160))  # ENL-2 speckle on a flat scene
    lin[:, 40:120, 40:120] *= 8.0  # one bright block for edges
    db = 10 * np.log10(lin)
    fig, m = sar_patch_figure(db, np.ones((160, 160), bool), enl_value=2.0)
    assert isinstance(fig, Figure)
    assert m["linear"]["enl"] == pytest.approx(2.0, rel=0.35)
    assert m["Lee 7"]["enl"] > m["Lee 5"]["enl"] > m["dB"]["enl"]
    assert m["Lee 7"]["epi"] < m["Lee 5"]["epi"] < 1.0


def test_strip_calibration_figure_spread() -> None:
    rng = np.random.default_rng(4)
    means = {"s1": [20.0] * 4, "s2": [35.0] * 4, "s3": [27.0] * 4}
    stats = {
        k: {"p_lo": [v[0] - 10] * 4, "p_hi": [v[0] + 10] * 4, "mean": v, "std": [4.0] * 4}
        for k, v in means.items()
    }
    glob = {"p_lo": [10.0] * 4, "p_hi": [45.0] * 4, "mean": [27.0] * 4, "std": [8.0] * 4}
    samples = {k: rng.normal(v[0], 4, 1000) for k, v in means.items()}
    fig, m = strip_calibration_figure(means, glob["mean"], samples, stats, glob)
    assert isinstance(fig, Figure)
    assert m == {"strips": 3, "mean_spread_db": 15.0}


def test_coreg_overlay_identical_scenes_correlate() -> None:
    scene = _scene(200, 200, seed=5).astype(np.float64)
    valid = np.ones((200, 200), bool)
    fig, m = coreg_overlay_figure(scene, scene[:3], valid, ksize=3, blur_sigma=1.0, crop=150)
    assert isinstance(fig, Figure)
    assert m["edge_correlation_at_zero_shift"] > 0.99


def test_tiling_overlay_counts_tiles() -> None:
    img = np.zeros((100, 60, 3))
    fig, m = tiling_overlay_figure(img, 0.1, (1000, 600), tile=256, overlap=64, band_px=32)
    assert isinstance(fig, Figure)
    assert m["tiles_o0"] == 4 * 3
