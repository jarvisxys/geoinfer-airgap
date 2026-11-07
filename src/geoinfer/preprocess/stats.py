"""Streaming per-band histograms and the train-split statistics derived from them.

Histograms have fixed bins, so tiles can be accumulated one at a time (never more than
one tile in RAM) and partial histograms from several processes can be merged.
Percentiles and moments are then read off the histogram:

* optical uint16 uses unit integer bins, so percentiles are exact;
* SAR dB uses 0.01 dB bins, so percentiles are within 0.01 dB.

Percentiles commute with monotonic maps, so the linear-intensity percentiles of the
SAR baseline arm are ``10 ** (p_dB / 10)`` without a second pass over the data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
Stats = dict[str, F64]


class BandHistogram:
    """Fixed-bin histogram per band over ``[lo, hi)``.

    Values below ``lo`` or at/above ``hi`` are counted in the first/last bin and also
    tallied in :attr:`clipped`. Unit bins on an integer ``lo`` are treated as exact
    integer values (discrete data such as uint16 optical).

    Parameters
    ----------
    n_bands
        Number of bands.
    lo, hi, step
        Bin range and width.
    """

    def __init__(self, n_bands: int, lo: float, hi: float, step: float) -> None:
        if hi <= lo or step <= 0:
            raise ValueError("need hi > lo and step > 0")
        self.lo, self.hi, self.step = float(lo), float(hi), float(step)
        self.n_bins = round((self.hi - self.lo) / self.step)
        self.discrete = self.step == 1.0 and self.lo.is_integer()
        self.counts = np.zeros((n_bands, self.n_bins), dtype=np.int64)
        self.clipped = np.zeros(n_bands, dtype=np.int64)

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(self.counts.shape[0])

    def add(self, x: npt.NDArray[Any], valid: npt.NDArray[np.bool_]) -> None:
        """Accumulate the valid pixels of ``x`` ``(bands, h, w)``."""
        if x.shape[0] != self.n_bands:
            raise ValueError(f"expected {self.n_bands} bands, got {x.shape[0]}")
        for b in range(self.n_bands):
            v = x[b][valid].astype(np.float64)
            self.clipped[b] += int(np.count_nonzero((v < self.lo) | (v >= self.hi)))
            idx = np.clip(np.floor((v - self.lo) / self.step), 0, self.n_bins - 1)
            self.counts[b] += np.bincount(idx.astype(np.int64), minlength=self.n_bins)

    def merge(self, other: BandHistogram) -> None:
        """Add another histogram with identical bins into this one."""
        same = (self.lo, self.hi, self.step, self.counts.shape) == (
            other.lo,
            other.hi,
            other.step,
            other.counts.shape,
        )
        if not same:
            raise ValueError("cannot merge histograms with different bins or band counts")
        self.counts += other.counts
        self.clipped += other.clipped

    def _values(self) -> F64:
        """Representative value of each bin (left edge if discrete, else centre)."""
        offset = 0.0 if self.discrete else 0.5
        return np.asarray(self.lo + self.step * (np.arange(self.n_bins) + offset), np.float64)

    def percentile(self, q: Sequence[float]) -> F64:
        """Per-band percentiles, numpy ``linear`` method; returns ``(bands, len(q))``.

        Within a bin, order statistics are spread uniformly (continuous data) or sit on
        the bin value (discrete data).
        """
        out = np.zeros((self.n_bands, len(q)))
        for b in range(self.n_bands):
            cnt = self.counts[b]
            cum = np.cumsum(cnt)
            n = int(cum[-1])
            if n == 0:
                out[b] = np.nan
                continue

            def order_stat(k: int, cnt: npt.NDArray[np.int64] = cnt, cum: Any = cum) -> float:
                i = int(np.searchsorted(cum, k, side="right"))
                if self.discrete:
                    return self.lo + self.step * i
                before = cum[i - 1] if i > 0 else 0
                return float(self.lo + self.step * (i + (k - before + 0.5) / cnt[i]))

            for j, qq in enumerate(q):
                r = qq / 100.0 * (n - 1)
                k0 = int(np.floor(r))
                k1 = min(k0 + 1, n - 1)
                v0, v1 = order_stat(k0), order_stat(k1)
                out[b, j] = v0 + (r - k0) * (v1 - v0)
        return out

    def moments(self, f: Callable[[F64, int], F64]) -> tuple[F64, F64]:
        """Mean and std per band of ``f(value, band)`` applied to the binned data."""
        vals = self._values()
        mean, std = np.zeros(self.n_bands), np.zeros(self.n_bands)
        for b in range(self.n_bands):
            w = self.counts[b].astype(np.float64)
            fv = f(vals, b)
            mean[b] = np.sum(w * fv) / w.sum()
            std[b] = np.sqrt(np.sum(w * (fv - mean[b]) ** 2) / w.sum())
        return mean, std

    def save(self, path: str | Path) -> None:
        """Write the histogram to an ``.npz`` file."""
        np.savez_compressed(
            path, counts=self.counts, clipped=self.clipped, bins=[self.lo, self.hi, self.step]
        )

    @classmethod
    def load(cls, path: str | Path) -> BandHistogram:
        """Read a histogram written by :meth:`save`."""
        with np.load(path) as z:
            lo, hi, step = (float(v) for v in z["bins"])
            h = cls(int(z["counts"].shape[0]), lo, hi, step)
            h.counts = z["counts"].astype(np.int64)
            h.clipped = z["clipped"].astype(np.int64)
        return h


def optical_stats(h: BandHistogram, percentiles: tuple[float, float]) -> Stats:
    """Global stretch points and the mean / std of the stretched data.

    Returns
    -------
    dict
        ``p_lo``, ``p_hi`` (black / white points) and ``mean``, ``std`` of
        ``clip((x - p_lo) / (p_hi - p_lo), 0, 1)``, one value per band.
    """
    p = h.percentile(list(percentiles))
    lo, hi = p[:, 0], p[:, 1]
    mean, std = h.moments(lambda v, b: np.clip((v - lo[b]) / (hi[b] - lo[b]), 0.0, 1.0))
    return {"p_lo": lo, "p_hi": hi, "mean": mean, "std": std}


def sar_arm_stats(
    h: BandHistogram,
    percentiles: tuple[float, float],
    domain: Literal["db", "linear"],
    clip: bool,
) -> Stats:
    """Statistics for one S2 SAR arm from a histogram of its dB values.

    Parameters
    ----------
    h
        Histogram of the arm's dB values (after Lee, for the Lee arms).
    percentiles
        Clip range, e.g. ``(1, 99)``.
    domain
        ``db`` keeps dB; ``linear`` maps everything through ``10 ** (x / 10)``.
    clip
        Clip to the percentile range before the moments are taken.

    Returns
    -------
    dict
        ``p_lo``, ``p_hi``, ``mean``, ``std`` per band, in the arm's domain.
    """
    p = h.percentile(list(percentiles))
    lo, hi = p[:, 0], p[:, 1]

    def f(v: F64, b: int) -> F64:
        x = np.clip(v, lo[b], hi[b]) if clip else v
        return np.power(10.0, x / 10.0) if domain == "linear" else x

    mean, std = h.moments(f)
    if domain == "linear":
        lo, hi = np.power(10.0, lo / 10.0), np.power(10.0, hi / 10.0)
    return {"p_lo": lo, "p_hi": hi, "mean": mean, "std": std}
