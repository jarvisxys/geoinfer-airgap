"""Quick-look PNG rendering for large GeoTIFFs.

Reads a decimated view of the raster (rasterio uses overviews when present), so a
quick-look of a multi-GB scene never loads the full-resolution raster into RAM.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.enums import Resampling

FloatArray = npt.NDArray[np.float32]
ByteArray = npt.NDArray[np.uint8]


def percentile_stretch(
    band: FloatArray,
    valid: npt.NDArray[np.bool_],
    low: float = 2.0,
    high: float = 98.0,
) -> ByteArray:
    """Linearly map the [low, high] percentile range of valid pixels to 0..255.

    Parameters
    ----------
    band
        Single band, any float range.
    valid
        Boolean mask, True where the pixel is not nodata.
    low, high
        Percentiles used as the black and white points.

    Returns
    -------
    numpy.ndarray
        uint8 image; invalid pixels are 0.
    """
    out = np.zeros(band.shape, dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(band[valid], [low, high])
    if hi <= lo:
        out[valid] = 128
        return out
    scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
    out[valid] = scaled[valid].astype(np.uint8)
    return out


def read_decimated(
    path: str | Path,
    bands: Sequence[int] | None = None,
    max_size: int = 1024,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Read selected bands at reduced resolution so the longest side is <= ``max_size``.

    Returns
    -------
    tuple
        ``(data, valid)`` where ``data`` is float32 ``(bands, h, w)`` and ``valid``
        is a ``(h, w)`` mask that is False where every band is nodata.
    """
    with rasterio.open(path) as src:
        idx = list(bands) if bands is not None else list(range(1, min(src.count, 3) + 1))
        scale = min(1.0, max_size / max(src.width, src.height))
        h = max(1, round(src.height * scale))
        w = max(1, round(src.width * scale))
        data = src.read(idx, out_shape=(len(idx), h, w), resampling=Resampling.average)
        data = data.astype(np.float32)
        if src.nodata is not None:
            valid = ~np.all(data == src.nodata, axis=0)
        else:
            valid = np.ones((h, w), dtype=bool)
    return data, valid


def render_quicklook(
    path: str | Path,
    out_png: str | Path,
    bands: Sequence[int] | None = None,
    max_size: int = 1024,
    sar_db: bool = False,
) -> ByteArray:
    """Render a percentile-stretched quick-look PNG of a GeoTIFF.

    Parameters
    ----------
    path
        Input GeoTIFF.
    out_png
        Output PNG path (parent directories are created).
    bands
        1-based band indices; 1 band renders grey, 3 bands render RGB.
        Defaults to the first three bands (or fewer if the raster has fewer).
    max_size
        Longest side of the output in pixels.
    sar_db
        Convert linear SAR intensity to dB (``10 * log10``) before stretching.

    Returns
    -------
    numpy.ndarray
        The rendered uint8 image, ``(h, w)`` or ``(h, w, 3)`` in RGB order.
    """
    data, valid = read_decimated(path, bands, max_size)
    if sar_db:
        valid &= np.all(data > 0, axis=0)
        data = (10.0 * np.log10(np.maximum(data, 1e-6))).astype(np.float32)

    channels = [percentile_stretch(b, valid) for b in data]
    img = channels[0] if len(channels) == 1 else np.dstack(channels[:3])

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    to_write = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_png), to_write):
        raise OSError(f"could not write {out_png}")
    return img


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m geoinfer.io.quicklook in.tif out.png [--bands 3 2 1] [--sar]``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--bands", type=int, nargs="+")
    p.add_argument("--max-size", type=int, default=1024)
    p.add_argument("--sar", action="store_true", help="linear intensity -> dB before stretch")
    a = p.parse_args(argv)
    render_quicklook(a.src, a.out, a.bands, a.max_size, a.sar)


if __name__ == "__main__":
    main()
