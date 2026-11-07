from __future__ import annotations

import numpy as np
import pytest

from geoinfer.preprocess.clouds import apply_clouds, cloud_fraction, cloud_mask, fractal_noise


def test_fractal_noise_is_seeded_and_in_unit_range() -> None:
    a = fractal_noise((64, 80), beta=2.2, seed=1)
    assert a.shape == (64, 80)
    assert a.dtype == np.float32
    assert a.min() == 0.0
    assert a.max() == 1.0
    np.testing.assert_array_equal(a, fractal_noise((64, 80), beta=2.2, seed=1))
    assert not np.array_equal(a, fractal_noise((64, 80), beta=2.2, seed=2))


def test_higher_beta_gives_smoother_noise() -> None:
    def roughness(beta: float) -> float:
        n = fractal_noise((128, 128), beta=beta, seed=0)
        return float(np.abs(np.diff(n, axis=1)).mean())

    assert roughness(3.0) < roughness(2.0) < roughness(1.0)


@pytest.mark.parametrize("coverage", [0.2, 0.5, 0.8])
def test_cloud_mask_hits_target_coverage(coverage: float) -> None:
    alpha = cloud_mask((256, 256), coverage, beta=2.2, softness=0.15, seed=0)
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0
    assert cloud_fraction(alpha) == pytest.approx(coverage, abs=0.01)


@pytest.mark.parametrize(("coverage", "value"), [(0.0, 0.0), (1.0, 1.0)])
def test_cloud_mask_extremes(coverage: float, value: float) -> None:
    alpha = cloud_mask((32, 32), coverage, beta=2.2, softness=0.15, seed=0)
    assert np.all(alpha == value)


def test_cloud_mask_rejects_bad_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        cloud_mask((8, 8), 1.5, beta=2.2, softness=0.15, seed=0)


def test_apply_clouds_zero_coverage_is_identity() -> None:
    x = np.random.default_rng(0).integers(1, 3000, (4, 32, 32)).astype(np.uint16)
    alpha = np.zeros((32, 32), dtype=np.float32)
    out = apply_clouds(x, np.ones((32, 32), dtype=bool), alpha, brightness=[4000] * 4)
    np.testing.assert_array_equal(out, x)
    assert out.dtype == np.uint16


def test_apply_clouds_blends_towards_brightness_and_keeps_nodata() -> None:
    x = np.full((2, 4, 4), 1000, dtype=np.uint16)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    x[:, 0, 0] = 0
    alpha = np.full((4, 4), 0.5, dtype=np.float32)
    alpha[3, 3] = 1.0
    out = apply_clouds(x, valid, alpha, brightness=[3000, 5000])
    assert out[0, 1, 1] == 2000
    assert out[1, 1, 1] == 3000
    assert out[0, 3, 3] == 3000
    assert np.all(out[:, 0, 0] == 0)


def test_apply_clouds_clips_to_dtype_range() -> None:
    x = np.full((1, 2, 2), 200, dtype=np.uint8)
    out = apply_clouds(x, np.ones((2, 2), bool), np.ones((2, 2), np.float32), brightness=[400])
    assert np.all(out == 255)
