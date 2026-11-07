from __future__ import annotations

import cv2
import numpy as np
import pytest

from geoinfer.preprocess.coreg import (
    coregistration_shift,
    edge_magnitude,
    phase_correlation_shift,
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


def test_phase_correlation_identity_is_zero_shift() -> None:
    img = _edges(_texture(N, 1))
    dx, dy, resp = phase_correlation_shift(img, img)
    assert abs(dx) < 1e-6
    assert abs(dy) < 1e-6
    assert resp == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("dx", "dy"), [(3.25, -1.5), (-0.4, 0.7), (7.0, 2.0), (0.5, 0.5), (-2.75, 1.1)]
)
def test_phase_correlation_recovers_subpixel_shift_on_edges(dx: float, dy: float) -> None:
    ref, mov = _shifted_pair(dx, dy)
    got_dx, got_dy, resp = phase_correlation_shift(_edges(ref), _edges(mov))
    assert got_dx == pytest.approx(dx, abs=0.1)
    assert got_dy == pytest.approx(dy, abs=0.1)
    assert resp > 0.2


def test_response_is_low_for_unrelated_images() -> None:
    a, b = _edges(_texture(N, 2)), _edges(_texture(N, 3))
    _, _, resp_unrelated = phase_correlation_shift(a, b)
    _, _, resp_same = phase_correlation_shift(a, a)
    assert resp_unrelated < 0.1 * resp_same


def test_coregistration_shift_handles_inverted_contrast_modality() -> None:
    ref, mov = _shifted_pair(2.5, -1.0, seed=4)
    optical = np.stack([ref, ref, ref])  # (3, h, w)
    noise = 0.02 * np.random.default_rng(5).normal(size=ref.shape)
    sar = np.stack([-2.0 * mov + noise] * 4)  # inverted contrast, other scale
    valid = np.ones(ref.shape, dtype=bool)
    res = coregistration_shift(sar, optical, valid, window=1024, ksize=3, optical_bands=[1, 2, 3])
    assert res.dx == pytest.approx(2.5, abs=0.15)
    assert res.dy == pytest.approx(-1.0, abs=0.15)
    assert res.window == N  # tile smaller than the window: used whole


def test_coregistration_shift_uses_central_window() -> None:
    scene = _texture(300, seed=6)
    stack = np.stack([scene] * 4)
    res = coregistration_shift(
        stack,
        stack[:3],
        np.ones((300, 300), dtype=bool),
        window=128,
        ksize=3,
        optical_bands=[1, 2, 3],
    )
    assert res.window == 128
    assert abs(res.dx) < 0.05
    assert abs(res.dy) < 0.05


def test_phase_correlation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        phase_correlation_shift(np.zeros((8, 8)), np.zeros((8, 9)))
