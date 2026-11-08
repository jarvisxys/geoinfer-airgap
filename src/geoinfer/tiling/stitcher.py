"""Blend overlapping tile predictions back into one scene (pipeline step 7, study S1).

Each tile's values ``v`` are accumulated with a per-pixel weight ``w``:
``P(p) = sum_k w_k(p) v_k(p) / sum_k w_k(p)``. Pixels with no weight (never covered,
or invalid in every covering tile) become nodata. Unlike the plan's ``+ epsilon`` form,
the division is only done where the weight sum is positive, so low-weight pixels near
the scene corners are not biased towards 0 and a constant image is reproduced exactly
up to float32 rounding.

Strategies (``w`` for a ``T x T`` tile with overlap ``O``):

- ``naive``: 1 everywhere; tiles are pasted, ``O`` must be 0.
- ``crop``: 1 on the centre crop ``[O/2, T - O/2)``, 0 elsewhere.
- ``hann``: ``sin^2(pi (x + 0.5) / T) sin^2(pi (y + 0.5) / T)`` (plan formula).
- ``tukey``: 1 in the middle, cosine ramps over the ``O`` px at each side; neighbouring
  ramps sum to 1 across the overlap, so network border pixels get little weight while
  the tile interior keeps full weight.

Accumulators are float32 and live in RAM, or in disk memmaps for large scenes.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from rasterio.windows import Window

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import RasterInfo, RasterWriter
from geoinfer.tiling.config import STRATEGIES, check_grid
from geoinfer.tiling.tiler import TileSpec

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

DEFAULT_MEMMAP_MIN_PX = 50_000_000


def _ramp(tile: int, overlap: int) -> npt.NDArray[np.float64]:
    w = np.ones(tile)
    if overlap:
        u = (np.arange(overlap) + 0.5) / overlap
        r = 0.5 * (1.0 - np.cos(np.pi * u))
        w[:overlap] = r
        w[tile - overlap :] = r[::-1]
    return w


def blend_weights(tile: int, overlap: int, strategy: str) -> FloatArray:
    """``(T, T)`` blending weights for one stitching strategy (see module docstring)."""
    check_grid(tile, overlap)
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy == "naive":
        if overlap:
            raise ValueError("naive stitching pastes tiles; it needs overlap 0")
        one = np.ones(tile)
    elif strategy == "crop":
        one = np.zeros(tile)
        one[overlap // 2 : tile - overlap // 2] = 1.0
    elif strategy == "hann":
        one = np.sin(np.pi * (np.arange(tile) + 0.5) / tile) ** 2
    else:
        one = _ramp(tile, overlap)
    return np.outer(one, one).astype(np.float32)


class Stitcher:
    """Weighted accumulator for tile predictions over one scene.

    Parameters
    ----------
    height, width
        Scene size in pixels.
    tile, overlap
        Grid parameters the tiles were cut with.
    strategy
        Blending strategy, see :func:`blend_weights`.
    bands
        Channels per tile prediction.
    memmap_min_px
        Scenes with more pixels than this accumulate in disk memmaps.
    scratch_dir
        Where memmaps go; the system temp dir by default. Removed on :meth:`close`.
    """

    def __init__(
        self,
        height: int,
        width: int,
        tile: int,
        overlap: int,
        strategy: str,
        *,
        bands: int = 1,
        memmap_min_px: int = DEFAULT_MEMMAP_MIN_PX,
        scratch_dir: Path | None = None,
    ) -> None:
        self.height, self.width, self.tile, self.bands = height, width, tile, bands
        self.weights = blend_weights(tile, overlap, strategy)
        self._tmp: Path | None = None
        if height * width > memmap_min_px:
            if scratch_dir is not None:
                scratch_dir.mkdir(parents=True, exist_ok=True)
            self._tmp = Path(tempfile.mkdtemp(prefix="stitch-", dir=scratch_dir))
            self._acc: FloatArray = np.memmap(
                self._tmp / "acc.f32", np.float32, "w+", shape=(bands, height, width)
            )
            self._wsum: FloatArray = np.memmap(
                self._tmp / "wsum.f32", np.float32, "w+", shape=(height, width)
            )
        else:
            self._acc = np.zeros((bands, height, width), np.float32)
            self._wsum = np.zeros((height, width), np.float32)

    @property
    def uses_memmap(self) -> bool:
        """True when the accumulators are disk memmaps."""
        return self._tmp is not None

    def add(self, spec: TileSpec, values: npt.NDArray[Any], valid: BoolArray | None = None) -> None:
        """Accumulate one tile prediction.

        Parameters
        ----------
        spec
            Where the tile sits (from :func:`~geoinfer.tiling.tiler.tile_grid`).
        values
            ``(bands, T, T)`` or, for one band, ``(T, T)``.
        valid
            ``(T, T)``; invalid pixels get zero weight.
        """
        t = self.tile
        v = values[None] if values.ndim == 2 else values
        if v.shape != (self.bands, t, t):
            raise ValueError(f"tile shape {values.shape} != {(self.bands, t, t)}")
        if valid is not None and valid.shape != (t, t):
            raise ValueError(f"valid shape {valid.shape} != {(t, t)}")
        ty0, tx0 = max(0, -spec.y0), max(0, -spec.x0)
        ty1, tx1 = min(t, self.height - spec.y0), min(t, self.width - spec.x0)
        if ty1 <= ty0 or tx1 <= tx0:
            return
        w = self.weights[ty0:ty1, tx0:tx1]
        if valid is not None:
            w = w * valid[ty0:ty1, tx0:tx1]
        ys = slice(spec.y0 + ty0, spec.y0 + ty1)
        xs = slice(spec.x0 + tx0, spec.x0 + tx1)
        self._acc[:, ys, xs] += w * v[:, ty0:ty1, tx0:tx1].astype(np.float32)
        self._wsum[ys, xs] += w

    def weight_sum(self, window: Window | None = None) -> FloatArray:
        """Copy of the summed weights, for the whole scene or one window."""
        if window is None:
            return np.array(self._wsum)
        r, c = window.toslices()
        return np.array(self._wsum[r, c])

    def read(self, window: Window, nodata: float = float("nan")) -> FloatArray:
        """Blended values ``(bands, h, w)`` for a window; ``nodata`` where uncovered."""
        r, c = window.toslices()
        acc, wsum = self._acc[:, r, c], self._wsum[r, c]
        out = np.full(acc.shape, nodata, np.float32)
        np.divide(acc, wsum, out=out, where=np.broadcast_to(wsum > 0, acc.shape))
        return out

    def write(
        self,
        dst: str | Path,
        like: RasterInfo,
        cfg: IOConfig,
        nodata: float,
        *,
        cog: bool = True,
    ) -> None:
        """Write the blended scene as float32 on the grid of ``like``, in row blocks.

        Parameters
        ----------
        dst
            Output path.
        like
            Reference raster info (CRS, transform, size); must match the scene size.
        cfg
            I/O config.
        nodata
            Value for uncovered pixels.
        cog
            Convert to a COG on close.
        """
        if (like.height, like.width) != (self.height, self.width):
            raise ValueError("reference raster size differs from the stitched scene")
        step = cfg.cog.blocksize
        with RasterWriter(
            dst, like, cfg, count=self.bands, dtype="float32", nodata=nodata, cog=cog
        ) as wr:
            for r0 in range(0, self.height, step):
                win = Window(0, r0, self.width, min(step, self.height - r0))
                wr.write(self.read(win, nodata), win)

    def close(self) -> None:
        """Release memmaps and delete their files."""
        if self._tmp is not None:
            del self._acc, self._wsum
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
