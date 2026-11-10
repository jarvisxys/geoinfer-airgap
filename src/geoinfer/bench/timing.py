"""Batch timing and the throughput statistics of the S4 / E1 tables."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def time_batches(
    fn: Callable[[npt.NDArray[np.float32]], Any],
    tiles: npt.NDArray[np.float32],
    batch: int,
    warmup: int,
    n_batches: int,
) -> npt.NDArray[np.float64]:
    """Seconds per batch for ``n_batches`` full batches after ``warmup`` untimed ones.

    Batches are cut from ``tiles`` cyclically, so a small pool serves any batch size. The
    engine must return host memory (a numpy array), so each timing includes the copy back
    and GPU work cannot hide behind an asynchronous launch.
    """
    n = len(tiles)
    idx = np.arange(batch)

    def get(k: int) -> npt.NDArray[np.float32]:
        return np.ascontiguousarray(tiles[(idx + k * batch) % n])

    for k in range(warmup):
        fn(get(k))
    out = np.empty(n_batches)
    for k in range(n_batches):
        x = get(warmup + k)
        t0 = time.perf_counter()
        fn(x)
        out[k] = time.perf_counter() - t0
    return out


def km2_per_min(tiles_per_s: float, tile: int, overlap: int, gsd_m: float) -> float:
    """Ground area per minute: each tile adds ``(T - O)^2`` new pixels of the grid."""
    return tiles_per_s * ((tile - overlap) * gsd_m) ** 2 / 1e6 * 60.0


def summarize(
    repeats: Sequence[npt.NDArray[np.float64]], batch: int, tile: int, overlap: int, gsd_m: float
) -> dict[str, float]:
    """Mean +- std tiles/s over repeats; p50 / p95 batch latency over all timed batches."""
    rates = np.array([batch * len(r) / r.sum() for r in repeats])
    pooled = np.concatenate(list(repeats))
    tps = float(rates.mean())
    return {
        "tiles_per_s": round(tps, 2),
        "tiles_per_s_std": round(float(rates.std()), 2),
        "km2_per_min": round(km2_per_min(tps, tile, overlap, gsd_m), 3),
        "p50_ms": round(float(np.percentile(pooled, 50)) * 1e3, 2),
        "p95_ms": round(float(np.percentile(pooled, 95)) * 1e3, 2),
    }
