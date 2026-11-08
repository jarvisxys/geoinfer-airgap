from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoinfer.preprocess.normalize import percentile_stretch, zscore
from geoinfer.preprocess.stats import BandHistogram, optical_stats, sar_arm_stats

RNG = np.random.default_rng(0)


def _valid(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def test_integer_histogram_percentiles_are_exact_to_one_bin() -> None:
    x = RNG.integers(100, 3000, (3, 64, 64)).astype(np.uint16)
    h = BandHistogram(3, lo=0, hi=65536, step=1)
    h.add(x, _valid(64, 64))
    got = h.percentile([2.0, 98.0])
    want = np.percentile(x.reshape(3, -1), [2.0, 98.0], axis=1).T
    np.testing.assert_allclose(got, want, atol=1.0)


def test_float_histogram_percentiles_within_one_bin() -> None:
    x = RNG.normal(30.0, 6.0, (2, 128, 128)).astype(np.float32)
    h = BandHistogram(2, lo=0.0, hi=100.0, step=0.01)
    h.add(x, _valid(128, 128))
    want = np.percentile(x.reshape(2, -1), [1.0, 50.0, 99.0], axis=1).T
    np.testing.assert_allclose(h.percentile([1.0, 50.0, 99.0]), want, atol=0.01)


def test_histogram_excludes_invalid_and_clips_out_of_range() -> None:
    x = np.array([[[-5.0, 50.0], [150.0, 7.0]]])
    valid = np.array([[True, True], [True, False]])
    h = BandHistogram(1, lo=0.0, hi=100.0, step=1.0)
    h.add(x, valid)
    assert h.counts.sum() == 3
    assert h.counts[0, 0] == 1  # -5 lands in the first bin
    assert h.counts[0, -1] == 1  # 150 lands in the last bin
    assert h.clipped.tolist() == [2]


def test_histogram_merge_equals_whole() -> None:
    x = RNG.normal(30.0, 6.0, (2, 64, 64))
    whole = BandHistogram(2, 0.0, 100.0, 0.01)
    whole.add(x, _valid(64, 64))
    a, b = BandHistogram(2, 0.0, 100.0, 0.01), BandHistogram(2, 0.0, 100.0, 0.01)
    a.add(x[:, :32], _valid(32, 64))
    b.add(x[:, 32:], _valid(32, 64))
    a.merge(b)
    np.testing.assert_array_equal(a.counts, whole.counts)


def test_histogram_merge_rejects_other_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        BandHistogram(1, 0.0, 100.0, 0.01).merge(BandHistogram(1, 0.0, 100.0, 0.1))


def test_histogram_save_load_roundtrip(tmp_path: Path) -> None:
    h = BandHistogram(2, 0.0, 10.0, 0.5)
    h.add(RNG.uniform(0, 10, (2, 8, 8)), _valid(8, 8))
    h.save(tmp_path / "h.npz")
    g = BandHistogram.load(tmp_path / "h.npz")
    np.testing.assert_array_equal(g.counts, h.counts)
    assert (g.lo, g.hi, g.step) == (h.lo, h.hi, h.step)


def test_optical_stats_make_train_data_zero_mean_unit_std() -> None:
    x = RNG.gamma(3.0, 300.0, (4, 128, 128)).astype(np.uint16)
    v = _valid(128, 128)
    h = BandHistogram(4, 0, 65536, 1)
    h.add(x, v)
    s = optical_stats(h, (2.0, 98.0))
    z = zscore(percentile_stretch(x, v, s["p_lo"], s["p_hi"]), v, s["mean"], s["std"])
    np.testing.assert_allclose(z.reshape(4, -1).mean(axis=1), 0.0, atol=2e-3)
    np.testing.assert_allclose(z.reshape(4, -1).std(axis=1), 1.0, atol=2e-3)


def test_sar_db_arm_clips_then_standardises() -> None:
    x = RNG.normal(25.0, 5.0, (2, 128, 128))
    v = _valid(128, 128)
    h = BandHistogram(2, 0.0, 100.0, 0.01)
    h.add(x, v)
    s = sar_arm_stats(h, (1.0, 99.0), domain="db", clip=True)
    clipped = np.clip(x, s["p_lo"][:, None, None], s["p_hi"][:, None, None])
    z = zscore(clipped, v, s["mean"], s["std"])
    np.testing.assert_allclose(z.reshape(2, -1).mean(axis=1), 0.0, atol=5e-3)
    np.testing.assert_allclose(z.reshape(2, -1).std(axis=1), 1.0, atol=5e-3)


def test_sar_linear_arm_maps_percentiles_and_moments() -> None:
    x = RNG.normal(25.0, 3.0, (1, 128, 128))
    h = BandHistogram(1, 0.0, 100.0, 0.01)
    h.add(x, _valid(128, 128))
    db = sar_arm_stats(h, (1.0, 99.0), domain="db", clip=False)
    lin = sar_arm_stats(h, (1.0, 99.0), domain="linear", clip=False)
    np.testing.assert_allclose(lin["p_lo"], 10 ** (db["p_lo"] / 10), rtol=1e-9)
    lin_x = 10 ** (x / 10)
    assert lin["mean"][0] == pytest.approx(lin_x.mean(), rel=5e-3)
    assert lin["std"][0] == pytest.approx(lin_x.std(), rel=1e-2)
