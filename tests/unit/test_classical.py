from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.filters import threshold_otsu

from geoinfer.models.classical import otsu_from_hist, otsu_per_label, otsu_threshold

LO, HI, BINS = 0.0, 60.0, 600


def _bimodal(n: int, w: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(n * w)
    return np.concatenate([rng.normal(12, 2.5, n - k), rng.normal(32, 3.5, k)])


@pytest.mark.parametrize("weight", [0.5, 0.2, 0.05])
def test_otsu_matches_skimage_on_bimodal_data(weight: float) -> None:
    x = _bimodal(200_000, weight)
    t = otsu_threshold(x, None, BINS, LO, HI)
    assert 15 < t < 30
    assert t == pytest.approx(threshold_otsu(x, nbins=BINS), abs=2 * (HI - LO) / BINS)


def test_otsu_from_hist_returns_a_bin_centre() -> None:
    counts = np.array([5, 9, 1, 0, 0, 2, 8, 6])
    edges = np.linspace(0, 8, 9)
    t = otsu_from_hist(counts, edges)
    assert t in {1.5, 2.5, 3.5, 4.5}  # a bin centre between the two clusters
    assert math.isnan(otsu_from_hist(np.array([0, 7, 0]), np.arange(4.0)))  # one value only
    with pytest.raises(ValueError, match="edges"):
        otsu_from_hist(np.ones(3), np.arange(3.0))


def test_otsu_threshold_respects_valid_and_range() -> None:
    x = _bimodal(50_000, 0.3).reshape(250, 200)
    valid = np.ones(x.shape, bool)
    x[:10] = 1e6  # garbage rows, masked out
    valid[:10] = False
    t = otsu_threshold(x, valid, BINS, LO, HI)
    assert 15 < t < 30
    with pytest.raises(ValueError, match="range"):
        otsu_threshold(x, valid, BINS, 5.0, 5.0)


def test_otsu_per_label_follows_each_labels_distribution() -> None:
    rng = np.random.default_rng(3)
    a, b = _bimodal(40_000, 0.3, 1), _bimodal(40_000, 0.3, 2) + 8.0
    x = np.concatenate([a, b, rng.normal(20, 1, 100)])
    labels = np.concatenate([np.full(a.size, 1), np.full(b.size, 2), np.full(100, 3)])
    valid = labels != 3  # label 3 is all invalid
    t = otsu_per_label(x, labels, valid, n_labels=5, bins=BINS, lo=LO, hi=HI)
    assert t.shape == (5,)
    assert math.isnan(t[0])  # label 0 has no pixels
    assert math.isnan(t[3])  # only invalid pixels
    assert math.isnan(t[4])
    assert t[2] - t[1] == pytest.approx(8.0, abs=1.0)
    assert t[1] == pytest.approx(otsu_threshold(a, None, BINS, LO, HI), abs=1e-9)
