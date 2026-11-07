"""Shared fixtures: tiny synthetic GeoTIFFs written to tmp_path."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from rasterio.transform import from_origin

# SpaceNet 6 Rotterdam tiles are in UTM zone 31N.
CRS = "EPSG:32631"
TRANSFORM = from_origin(594_000.0, 5_757_000.0, 0.5, 0.5)

MakeTiff = Callable[..., Path]


@pytest.fixture
def make_tiff(tmp_path: Path) -> MakeTiff:
    """Factory that writes ``array`` (bands, h, w) to a GeoTIFF and returns its path."""

    def _make(
        array: npt.NDArray[np.generic], name: str = "img.tif", nodata: float | None = None
    ) -> Path:
        path = tmp_path / name
        count, h, w = array.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=count,
            dtype=array.dtype,
            crs=CRS,
            transform=TRANSFORM,
            nodata=nodata,
        ) as dst:
            dst.write(array)
        return path

    return _make
