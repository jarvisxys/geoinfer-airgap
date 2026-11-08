from __future__ import annotations

import cv2
import numpy as np
import pytest

from geoinfer.preprocess.coreg import (
    coregistration_shift,
    edge_magnitude,
    masked_ncc_shift,
)

N = 256  # analysed window
PAD = 64  # margin so shifts bring in real content, not wrap-around


def _texture(n: int, seed: int) -> np.ndarray:
    """Broadband random texture (structure from ~1 px to ~10 px)."""
    rng = np.random.default_rng(seed)
    img = sum(cv2.GaussianBlur(rng.normal(size=(n, n)), (0, 0), s) * s for s in (0.7, 2, 6))
    return np.asarray(img, dtype=np.float64)


def _shifted_pair(dx: float, dy: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Centre crops of a scene and of the scene moved by (dx, dy): mov(x) = ref(x - d)."""
    big = _texture(N + 2 * PAD, seed)
    ky = np.fft.fftfreq(big.shape[0])[:, None]
    kx = np.fft.fftfreq(big.shape[1])[None, :]
    moved = np.real(np.fft.ifft2(np.fft.fft2(big) * np.exp(-2j * np.pi * (kx * dx + ky * dy))))
    crop = (slice(PAD, PAD + N), slice(PAD, PAD + N))
    return big[crop], moved[crop]


def _edges(img: np.ndarray) -> np.ndarray:
    return edge_magnitude(img, np.ones(img.shape, dtype=bool), ksize=3)


def test_edge_magnitude_is_zero_on_flat_and_near_nodata() -> None:
    img = np.zeros((32, 32))
    img[:, 16:] = 10.0
    valid = np.ones((32, 32), dtype=bool)
    valid[:4, :] = False
    e = edge_magnitude(img, valid, ksize=3)
    assert e.dtype == np.float32
    assert e[10, 16] > 0  # the step
    assert np.all(e[10, :10] == 0)  # flat area
    assert np.all(e[:5, :] == 0)  # nodata rows and the row next to them


KW = {"ksize": 3, "blur_sigma": 1.0, "max_shift": 12, "overlap_ratio": 0.3}


def _ncc(ref: np.ndarray, mov: np.ndarray, valid: np.ndarray | None = None, max_shift: int = 12):
    v = np.ones(ref.shape, dtype=bool) if valid is None else valid
    return masked_ncc_shift(ref, mov, v, v, max_shift=max_shift, overlap_ratio=0.3)


def test_masked_ncc_identity_is_zero_shift() -> None:
    img = _edges(_texture(N, 1))
    dx, dy, peak = _ncc(img, img)
    assert abs(dx) < 1e-6
    assert abs(dy) < 1e-6
    assert peak == pytest.approx(1.0)


@pytest.mark.parametrize(("dx", "dy"), [(3.0, -2.0), (-7.0, 5.0), (0.0, 9.0)])
def test_masked_ncc_recovers_integer_shift(dx: float, dy: float) -> None:
    ref, mov = _shifted_pair(dx, dy)
    got_dx, got_dy, peak = _ncc(_edges(ref), _edges(mov))
    assert got_dx == pytest.approx(dx, abs=0.05)
    assert got_dy == pytest.approx(dy, abs=0.05)
    assert peak > 0.9


@pytest.mark.parametrize(("dx", "dy"), [(3.25, -1.5), (-0.4, 0.7), (0.5, 0.5), (-2.75, 1.1)])
def test_masked_ncc_recovers_subpixel_shift(dx: float, dy: float) -> None:
    ref, mov = _shifted_pair(dx, dy)
    smooth = [cv2.GaussianBlur(x, (0, 0), 1.5) for x in (ref, mov)]
    got_dx, got_dy, _ = _ncc(*(_edges(x) for x in smooth))
    assert got_dx == pytest.approx(dx, abs=0.1)
    assert got_dy == pytest.approx(dy, abs=0.1)


def test_masked_ncc_is_not_pulled_to_zero_by_a_shared_mask() -> None:
    # SN6 tiles are 20-50 % nodata, identical in both sensors. Plain phase correlation
    # locks onto the shared mask outline and reports zero shift; masked NCC must not.
    ref, mov = _shifted_pair(4.0, -3.0, seed=7)
    yy, xx = np.mgrid[:N, :N]
    valid = xx + 0.6 * yy > 0.45 * N  # diagonal strip edge, ~40 % nodata
    e_ref = edge_magnitude(ref, valid, 3)
    e_mov = edge_magnitude(mov, valid, 3)
    keep = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dx, dy, _ = masked_ncc_shift(e_ref, e_mov, keep, keep, max_shift=12, overlap_ratio=0.3)
    assert dx == pytest.approx(4.0, abs=0.1)
    assert dy == pytest.approx(-3.0, abs=0.1)


def test_masked_ncc_peak_is_low_for_unrelated_images() -> None:
    a, b = _edges(_texture(N, 2)), _edges(_texture(N, 3))
    assert _ncc(a, b)[2] < 0.2


def test_masked_ncc_search_is_limited_to_max_shift() -> None:
    ref, mov = _shifted_pair(10.0, 0.0)
    dx, dy, _ = _ncc(_edges(ref), _edges(mov), max_shift=4)
    assert abs(dx) <= 4.0
    assert abs(dy) <= 4.0


def test_masked_ncc_rejects_bad_input() -> None:
    a = np.zeros((8, 8))
    v = np.ones((8, 8), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        masked_ncc_shift(a, np.zeros((8, 9)), v, v, max_shift=2, overlap_ratio=0.3)
    with pytest.raises(ValueError, match="max_shift"):
        masked_ncc_shift(a, a, v, v, max_shift=0, overlap_ratio=0.3)
    with pytest.raises(ValueError, match="overlap_ratio"):
        masked_ncc_shift(a, a, v, v, max_shift=2, overlap_ratio=0.0)
    with pytest.raises(ValueError, match="overlap"):
        masked_ncc_shift(a, a, ~v, v, max_shift=2, overlap_ratio=0.3)


def test_coregistration_shift_handles_inverted_contrast_modality() -> None:
    ref, mov = _shifted_pair(2.5, -1.0, seed=4)
    optical = np.stack([ref, ref, ref])  # (3, h, w)
    noise = 0.02 * np.random.default_rng(5).normal(size=ref.shape)
    sar = np.stack([-2.0 * mov + noise] * 4)  # inverted contrast, other scale
    valid = np.ones(ref.shape, dtype=bool)
    res = coregistration_shift(sar, optical, valid, window=1024, optical_bands=[1, 2, 3], **KW)
    assert res.dx == pytest.approx(2.5, abs=0.1)
    assert res.dy == pytest.approx(-1.0, abs=0.1)
    assert res.response > 0.5
    assert res.window == N  # tile smaller than the window: used whole


def test_coregistration_shift_with_nodata_recovers_shift() -> None:
    ref, mov = _shifted_pair(-3.0, 2.0, seed=8)
    yy = np.mgrid[:N, :N][0]
    valid = yy < 0.6 * N  # bottom 40 % nodata in both sensors
    optical = np.stack([np.where(valid, ref, 0.0)] * 3)
    sar = np.stack([np.where(valid, mov, 0.0)] * 4)
    res = coregistration_shift(sar, optical, valid, window=1024, optical_bands=[1, 2, 3], **KW)
    assert res.dx == pytest.approx(-3.0, abs=0.1)
    assert res.dy == pytest.approx(2.0, abs=0.1)


def test_coregistration_shift_uses_central_window() -> None:
    scene = _texture(300, seed=6)
    stack = np.stack([scene] * 4)
    valid = np.ones((300, 300), dtype=bool)
    res = coregistration_shift(stack, stack[:3], valid, window=128, optical_bands=[1, 2, 3], **KW)
    assert res.window == 128
    assert abs(res.dx) < 0.05
    assert abs(res.dy) < 0.05
