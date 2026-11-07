"""Cloud-Optimised GeoTIFF conversion (pipeline stage 1).

SpaceNet 6 tiles and mosaics are stored one row per block with no overviews, so a 512 px
window read touches 512 full-width rows. Converting to a COG (512 px internal tiles,
DEFLATE, overviews) makes windowed reads and quick-looks cheap. GDAL streams the copy, so
memory stays bounded by ``gdal_cachemax_mb`` regardless of raster size.
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from collections.abc import Sequence
from pathlib import Path

import rasterio
import rasterio.shutil

from geoinfer.io.config import DEFAULT_IO_CONFIG, IOConfig, load_io_config
from geoinfer.io.paths import assert_writable


def to_cog(
    src: str | Path,
    dst: str | Path,
    cfg: IOConfig,
    *,
    overview_resampling: str | None = None,
) -> Path:
    """Convert a raster to a COG, writing to a temporary file and renaming on success.

    Parameters
    ----------
    src
        Input raster (any GDAL-readable format).
    dst
        Output COG path; parent directories are created.
    cfg
        I/O config (COG options, GDAL cache, read-only roots).
    overview_resampling
        Override ``cfg.cog.overview_resampling``, e.g. ``NEAREST`` for masks.

    Returns
    -------
    pathlib.Path
        ``dst``. Pixels, CRS, transform, dtype and nodata are those of ``src``.

    Raises
    ------
    PermissionError
        If ``dst`` is under a read-only root.
    ValueError
        If ``dst`` is ``src``.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError(f"refusing to convert {src} onto itself")
    assert_writable(dst, cfg.read_only_roots)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    try:
        with rasterio.Env(GDAL_CACHEMAX=cfg.gdal_cachemax_mb), rasterio.open(src) as ds:
            rasterio.shutil.copy(
                ds, part, driver="COG", **cfg.cog.creation_options(overview_resampling)
            )
        os.replace(part, dst)
    except BaseException:
        for leftover in (part, part.with_name(part.name + ".ovr.tmp")):
            leftover.unlink(missing_ok=True)
        raise
    return dst


def is_cog(path: str | Path) -> bool:
    """Return True if ``path`` is a tiled GeoTIFF with the COG layout."""
    with rasterio.open(path) as src:
        layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        return layout == "COG" and bool(src.profile.get("tiled", False))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m geoinfer.io.cog SRC DST [--config configs/io.yaml]``."""
    p = argparse.ArgumentParser(description="Convert a raster to a Cloud-Optimised GeoTIFF.")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_IO_CONFIG)
    p.add_argument("--overview-resampling", help="e.g. NEAREST for masks")
    a = p.parse_args(argv)

    t0 = time.perf_counter()
    to_cog(a.src, a.dst, load_io_config(a.config), overview_resampling=a.overview_resampling)
    elapsed = time.perf_counter() - t0
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    mib = 1024**2
    print(
        f"{a.src} ({a.src.stat().st_size / mib:.1f} MiB) -> {a.dst} "
        f"({a.dst.stat().st_size / mib:.1f} MiB) in {elapsed:.1f} s, "
        f"peak RSS {peak_mib:.0f} MiB, cog={is_cog(a.dst)}"
    )


if __name__ == "__main__":
    main()
