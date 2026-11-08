from __future__ import annotations

import numpy as np
import pytest

from geoinfer.preprocess.normalize import apply_sar_arm, select_arm_stats
from geoinfer.preprocess.stats import BandHistogram, strip_arm_stats


def _hist(mu: float, n: int, seed: int) -> BandHistogram:
    rng = np.random.default_rng(seed)
    h = BandHistogram(2, 0.0, 100.0, 0.01)
    x = rng.normal(mu, 3.0, size=(2, 1, n))
    h.add(x, np.ones((1, n), dtype=bool))
    return h


def test_strip_arm_stats_follow_each_strip_and_skip_small_ones() -> None:
    hists = {"A": _hist(20.0, 50_000, 0), "B": _hist(30.0, 50_000, 1), "C": _hist(25.0, 50, 2)}
    out = strip_arm_stats(hists, (1.0, 99.0), "db", True, min_valid_px=1000)
    assert out["C"] is None
    a, b = out["A"], out["B"]
    assert a is not None
    assert b is not None
    np.testing.assert_allclose(a["mean"], 20.0, atol=0.1)
    np.testing.assert_allclose(b["mean"] - a["mean"], 10.0, atol=0.15)
    np.testing.assert_allclose(a["std"], 3.0 * 0.97, rtol=0.05)  # clipped at p1 / p99
    assert np.all(a["p_lo"] < a["mean"])
    assert np.all(a["mean"] < a["p_hi"])


def test_select_arm_stats_prefers_the_strip_and_falls_back() -> None:
    glob = {"p_lo": [1.0], "p_hi": [9.0], "mean": [5.0], "std": [2.0]}
    strip = {"p_lo": [2.0], "p_hi": [8.0], "mean": [4.0], "std": [1.0]}
    per = {"s1": strip, "s2": None}
    assert select_arm_stats("s1", glob, per) == (strip, "strip")
    assert select_arm_stats("s2", glob, per) == (glob, "fallback")
    assert select_arm_stats("unknown", glob, per) == (glob, "fallback")
    assert select_arm_stats(None, glob, per) == (glob, "fallback")


def test_apply_sar_arm_clips_then_standardises() -> None:
    x = np.array([[[0.0, 10.0, 20.0, 50.0]]], dtype=np.float32)  # (1 band, 1, 4)
    valid = np.array([[True, True, True, False]])
    stats = {"p_lo": [5.0], "p_hi": [15.0], "mean": [10.0], "std": [2.5]}
    out = apply_sar_arm(x, valid, stats, domain="db", clip=True)
    np.testing.assert_allclose(out[0, 0], [-2.0, 0.0, 2.0, 0.0])
    unclipped = apply_sar_arm(x, valid, stats, domain="db", clip=False)
    np.testing.assert_allclose(unclipped[0, 0], [-4.0, 0.0, 4.0, 0.0])
    assert out.dtype == np.float32


def test_apply_sar_arm_linear_domain() -> None:
    x = np.array([[[10.0, 20.0]]], dtype=np.float32)  # dB -> 10 and 100 linear
    valid = np.ones((1, 2), dtype=bool)
    stats = {"p_lo": [0.0], "p_hi": [1e9], "mean": [55.0], "std": [45.0]}
    out = apply_sar_arm(x, valid, stats, domain="linear", clip=False)
    np.testing.assert_allclose(out[0, 0], [-1.0, 1.0], rtol=1e-6)
    with pytest.raises(ValueError, match="domain"):
        apply_sar_arm(x, valid, stats, domain="log", clip=False)  # type: ignore[arg-type]
