"""Typed view of ``configs/io.yaml`` (COG options, read limits, read-only roots)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IO_CONFIG = Path("configs/io.yaml")


@dataclass(frozen=True)
class CogOptions:
    """GDAL COG driver creation options.

    Attributes
    ----------
    blocksize
        Internal tile size in pixels (multiple of 16).
    compress
        Compression codec, e.g. ``DEFLATE``.
    predictor
        ``YES`` lets GDAL choose 2 (integers) or 3 (floats); ``NO`` disables it.
    overview_resampling
        Resampling used to build overviews, e.g. ``AVERAGE`` or ``NEAREST``.
    num_threads
        Worker threads for compression, e.g. ``ALL_CPUS``.
    """

    blocksize: int
    compress: str
    predictor: str
    overview_resampling: str
    num_threads: str

    def creation_options(self, overview_resampling: str | None = None) -> dict[str, str]:
        """Return the options as GDAL ``KEY=value`` creation options."""
        return {
            "BLOCKSIZE": str(self.blocksize),
            "COMPRESS": self.compress,
            "PREDICTOR": self.predictor,
            "OVERVIEW_RESAMPLING": overview_resampling or self.overview_resampling,
            "NUM_THREADS": self.num_threads,
            "BIGTIFF": "IF_SAFER",
        }


@dataclass(frozen=True)
class IOConfig:
    """Raster I/O settings.

    Attributes
    ----------
    cog
        COG creation options.
    gdal_cachemax_mb
        GDAL block cache size in MB used during conversions.
    max_window_px
        Largest single window (width * height) a reader will return.
    read_only_roots
        Directories that must never be written to.
    """

    cog: CogOptions
    gdal_cachemax_mb: int
    max_window_px: int
    read_only_roots: tuple[Path, ...]


def load_io_config(path: str | Path = DEFAULT_IO_CONFIG) -> IOConfig:
    """Load and validate an I/O config file.

    Parameters
    ----------
    path
        YAML file in the layout of ``configs/io.yaml``.

    Returns
    -------
    IOConfig
        Parsed config; ``~`` in read-only roots is expanded.

    Raises
    ------
    ValueError
        If a value is out of range.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    cog = CogOptions(
        blocksize=int(raw["cog"]["blocksize"]),
        compress=str(raw["cog"]["compress"]),
        predictor=str(raw["cog"]["predictor"]),
        overview_resampling=str(raw["cog"]["overview_resampling"]),
        num_threads=str(raw["cog"]["num_threads"]),
    )
    if cog.blocksize <= 0 or cog.blocksize % 16:
        raise ValueError(f"cog.blocksize must be a positive multiple of 16, got {cog.blocksize}")
    cfg = IOConfig(
        cog=cog,
        gdal_cachemax_mb=int(raw["gdal_cachemax_mb"]),
        max_window_px=int(raw["max_window_px"]),
        read_only_roots=tuple(Path(p).expanduser() for p in raw.get("read_only_roots", [])),
    )
    if cfg.gdal_cachemax_mb <= 0 or cfg.max_window_px <= 0:
        raise ValueError("gdal_cachemax_mb and max_window_px must be positive")
    return cfg
