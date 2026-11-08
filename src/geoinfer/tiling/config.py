"""Typed view of ``configs/tiling.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TILING_CONFIG = Path("configs/tiling.yaml")
STRATEGIES = ("naive", "crop", "hann", "tukey")
PAD_MODES = ("reflect", "constant")


def check_grid(tile: int, overlap: int) -> None:
    """Raise ``ValueError`` unless ``tile > 0`` and ``overlap`` is even in ``[0, tile)``."""
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must be in [0, tile), got {overlap} for tile {tile}")
    if overlap % 2:
        raise ValueError(f"overlap must be even (half goes to each side), got {overlap}")


@dataclass(frozen=True)
class TilingConfig:
    """Tiling and stitching settings.

    Attributes
    ----------
    tile, overlap
        Tile side ``T`` and overlap ``O`` in pixels.
    strategy
        Stitching strategy, one of :data:`STRATEGIES`.
    pad_mode
        How tiles crossing the scene edge are filled, one of :data:`PAD_MODES`.
    skip_nodata
        Skip tiles without any valid pixel.
    memmap_min_px
        Scenes with more pixels than this accumulate in a disk memmap.
    scratch_dir
        Directory for memmap files.
    out_nodata
        Nodata value of stitched outputs (NaN by default).
    sweep_tile, sweep_overlap
        Values swept in studies S1 and S4.
    """

    tile: int
    overlap: int
    strategy: str
    pad_mode: str
    skip_nodata: bool
    memmap_min_px: int
    scratch_dir: Path
    out_nodata: float
    sweep_tile: tuple[int, ...]
    sweep_overlap: tuple[int, ...]

    def validate(self) -> TilingConfig:
        """Check the settings; return ``self`` so calls can be chained."""
        check_grid(self.tile, self.overlap)
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.strategy == "naive" and self.overlap:
            raise ValueError("naive stitching pastes tiles; it needs overlap 0")
        if self.pad_mode not in PAD_MODES:
            raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {self.pad_mode!r}")
        if self.memmap_min_px <= 0:
            raise ValueError("memmap_min_px must be positive")
        return self


def load_tiling_config(path: Path = DEFAULT_TILING_CONFIG) -> TilingConfig:
    """Read and validate ``configs/tiling.yaml``."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return TilingConfig(
        tile=int(raw["tile"]),
        overlap=int(raw["overlap"]),
        strategy=str(raw["strategy"]),
        pad_mode=str(raw["pad_mode"]),
        skip_nodata=bool(raw["skip_nodata"]),
        memmap_min_px=int(raw["memmap_min_px"]),
        scratch_dir=Path(raw["scratch_dir"]).expanduser(),
        out_nodata=float(raw["out_nodata"]),
        sweep_tile=tuple(int(t) for t in raw["sweep"]["tile"]),
        sweep_overlap=tuple(int(o) for o in raw["sweep"]["overlap"]),
    ).validate()
