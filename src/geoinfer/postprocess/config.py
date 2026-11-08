"""Typed view of ``configs/postprocess.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POSTPROCESS_CONFIG = Path("configs/postprocess.yaml")


@dataclass(frozen=True)
class PostprocessConfig:
    """Post-processing settings.

    Attributes
    ----------
    threshold
        Probability above or at which a pixel is building.
    opening, closing
        Odd square kernel sides in pixels; 1 disables the step.
    min_area_m2
        Components smaller than this are removed.
    connectivity
        4 or 8, for component labelling and removal.
    simplify_tolerance_m
        Douglas-Peucker tolerance for polygons.
    block
        Block side in pixels for block-wise processing of large rasters.
    geojson_crs
        CRS of exported GeoJSON.
    """

    threshold: float
    opening: int
    closing: int
    min_area_m2: float
    connectivity: int
    simplify_tolerance_m: float
    block: int
    geojson_crs: str

    def validate(self) -> PostprocessConfig:
        """Check the settings; return ``self`` so calls can be chained."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        for name in ("opening", "closing"):
            size = getattr(self, name)
            if size < 1 or size % 2 == 0:
                raise ValueError(f"{name} must be an odd size >= 1, got {size}")
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity}")
        if self.min_area_m2 < 0 or self.simplify_tolerance_m < 0:
            raise ValueError("min_area_m2 and simplify_tolerance_m must be >= 0")
        if self.block < 1:
            raise ValueError(f"block must be positive, got {self.block}")
        return self


def load_postprocess_config(path: Path = DEFAULT_POSTPROCESS_CONFIG) -> PostprocessConfig:
    """Read and validate ``configs/postprocess.yaml``."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return PostprocessConfig(
        threshold=float(raw["threshold"]),
        opening=int(raw["opening"]),
        closing=int(raw["closing"]),
        min_area_m2=float(raw["min_area_m2"]),
        connectivity=int(raw["connectivity"]),
        simplify_tolerance_m=float(raw["simplify_tolerance_m"]),
        block=int(raw["block"]),
        geojson_crs=str(raw["geojson_crs"]),
    ).validate()
