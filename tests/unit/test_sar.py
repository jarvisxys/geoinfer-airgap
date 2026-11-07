from __future__ import annotations

import cv2
import numpy as np
import pytest

from geoinfer.preprocess.sar import (
    db_to_linear,
    edge_preservation_index,
    enl,
    estimate_enl,
    lee_filter,
    linear_to_db,
    sar_chain,
)

L = 4.0  # looks of the synthetic speckle


def _speckle(shape: tuple[int, ...], looks: float = L, seed: int = 0) -> np.ndarray:
    """Unit-mean gamma speckle (multiplicative, ENL = looks)."""
    return np.random.default_rng(seed).gamma(looks, 1.0 / looks, shape)


def _valid(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


# --- dB <-> linear ----------------------------------------------------------------------


def test_db_linear_roundtrip_keeps_nodata() -> None:
    db = np.array([[[0.0, 10.0], [23.5, 1e-5]]], dtype=np.float32)
    valid = np.array([[False, True], [True, True]])
    lin = db_to_linear(db, valid)
    assert lin[0, 0, 0] == 0.0
    assert lin[0, 0, 1] == pytest.approx(10.0)
    assert lin[0, 1, 1] == pytest.approx(1.0, rel=1e-5)  # SN6 floor -> 0 dB
    back = linear_to_db(lin, valid)
    np.testing.assert_allclose(back[0][valid], db[0][valid], atol=1e-4)
    assert back[0, 0, 0] == 0.0


# --- Lee filter -------------------------------------------------------------------------


@pytest.mark.parametrize("size", [5, 7])
def test_lee_leaves_constant_image_unchanged(size: int) -> None:
    x = np.full((2, 40, 50), 123.456, dtype=np.float32)
    out = lee_filter(x, _valid(40, 50), size, enl=L)
    np.testing.assert_allclose(out, x, rtol=1e-6)


def test_lee_does_not_bleed_nodata() -> None:
    x = np.full((1, 30, 30), 50.0, dtype=np.float32)
    valid = _valid(30, 30)
    valid[10:20, 10:20] = False
    x[0, ~valid] = 0.0
    out = lee_filter(x, valid, 7, enl=L)
    np.testing.assert_allclose(out[0][valid], 50.0, rtol=1e-6)
    assert np.all(out[0][~valid] == 0.0)


def test_lee_reduces_speckle_and_raises_enl() -> None:
    x = (100.0 * _speckle((1, 256, 256))).astype(np.float32)
    out5 = lee_filter(x, _valid(256, 256), 5, enl=L)
    out7 = lee_filter(x, _valid(256, 256), 7, enl=L)
    assert enl(out5[0]) > 3 * enl(x[0])
    assert enl(out7[0]) > enl(out5[0])  # bigger window, more smoothing
    assert out5.mean() == pytest.approx(x.mean(), rel=0.02)  # radiometry preserved


def test_lee_preserves_edges_better_than_box_blur() -> None:
    clean = np.full((128, 128), 10.0)
    clean[:, 64:] = 1000.0
    x = (clean * _speckle((128, 128), looks=16, seed=1)).astype(np.float32)
    lee = lee_filter(x[None], _valid(128, 128), 7, enl=16)[0]
    box = cv2.blur(x, (7, 7), borderType=cv2.BORDER_REFLECT_101)
    to_db = lambda a: 10 * np.log10(a)  # noqa: E731
    v = _valid(128, 128)
    assert edge_preservation_index(to_db(x), to_db(lee), v) > edge_preservation_index(
        to_db(x), to_db(box), v
    )


def test_lee_accepts_2d_and_returns_float32() -> None:
    out = lee_filter(np.ones((10, 10)), _valid(10, 10), 5, enl=L)
    assert out.shape == (10, 10)
    assert out.dtype == np.float32


@pytest.mark.parametrize(("size", "looks"), [(4, L), (1, L), (5, 0.0)])
def test_lee_rejects_bad_parameters(size: int, looks: float) -> None:
    with pytest.raises(ValueError, match=r"size|enl"):
        lee_filter(np.ones((1, 10, 10)), _valid(10, 10), size, enl=looks)


# --- ENL --------------------------------------------------------------------------------


def test_enl_of_gamma_speckle_matches_looks() -> None:
    assert enl(_speckle((512, 512))) == pytest.approx(L, rel=0.05)


def test_enl_respects_mask() -> None:
    x = _speckle((256, 256))
    x[:, :128] *= np.linspace(1, 50, 128)  # strong texture on the left
    mask = np.zeros_like(x, dtype=bool)
    mask[:, 128:] = True
    assert enl(x, mask) == pytest.approx(L, rel=0.05)


def test_estimate_enl_on_pure_speckle() -> None:
    est = estimate_enl(100.0 * _speckle((512, 512)), _valid(512, 512), window=32, quantile=0.05)
    assert est == pytest.approx(L, rel=0.10)


def test_estimate_enl_finds_homogeneous_area() -> None:
    x = 100.0 * _speckle((512, 512), seed=3)
    yy, xx = np.mgrid[0:512, 0:256]
    x[:, :256] *= 1 + 20 * (np.sin(xx / 3.0) * np.cos(yy / 5.0)) ** 2  # textured half
    est = estimate_enl(x, _valid(512, 512), window=32, quantile=0.05)
    assert est == pytest.approx(L, rel=0.15)


def test_estimate_enl_needs_valid_windows() -> None:
    valid = np.zeros((64, 64), dtype=bool)
    with pytest.raises(ValueError, match="window"):
        estimate_enl(np.ones((64, 64)), valid, window=32, quantile=0.05)


# --- edge preservation index ------------------------------------------------------------


def test_epi_identity_blur_and_constant() -> None:
    x = np.random.default_rng(4).normal(size=(64, 64))
    v = _valid(64, 64)
    assert edge_preservation_index(x, x, v) == pytest.approx(1.0)
    assert edge_preservation_index(x, cv2.blur(x, (5, 5)), v) < 0.5
    c = np.ones((8, 8))
    assert edge_preservation_index(c, c, _valid(8, 8)) == 1.0


def test_epi_ignores_pairs_touching_nodata() -> None:
    x = np.zeros((4, 4))
    x[:, 2:] = 10.0
    v = _valid(4, 4)
    f = x.copy()
    f[:, 1] = 99.0  # changed pixels sit in an invalid column
    v[:, 1] = False
    assert edge_preservation_index(x, f, v) == pytest.approx(1.0)


# --- full chain -------------------------------------------------------------------------


def test_sar_chain_without_lee_is_identity_on_valid() -> None:
    db = (20 + 5 * np.random.default_rng(5).random((4, 16, 16))).astype(np.float32)
    valid = _valid(16, 16)
    valid[0, 0] = False
    db[:, 0, 0] = 0.0
    np.testing.assert_array_equal(sar_chain(db, valid, lee=None, enl_value=L), db)


def test_sar_chain_with_lee_keeps_constant_db() -> None:
    db = np.full((4, 32, 32), 25.0, dtype=np.float32)
    out = sar_chain(db, _valid(32, 32), lee=5, enl_value=L)
    np.testing.assert_allclose(out, 25.0, atol=1e-4)
