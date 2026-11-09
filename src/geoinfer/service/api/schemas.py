"""Request and response models of the API (plan section 3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Sensor = Literal["optical", "sar", "fused"]


class SceneIn(BaseModel):
    """Register a COG already placed in the inbox."""

    id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.-]+$")
    path: str = Field(description="file name relative to the inbox")
    sensor: Sensor
    sar_path: str | None = Field(None, description="fused scenes: the SAR COG, same grid")
    orientation: Literal[0, 1] | None = Field(
        None, description="SAR look direction (0 / 1, SN6 SAR_orientations); SAR and fused"
    )


class SceneOut(BaseModel):
    id: str
    path: str
    sar_path: str | None
    sensor: Sensor
    orientation: int | None
    gsd_m: float
    width: int
    height: int
    crs: str
    sha256: str
    bounds: list[float] = Field(description="lon_min, lat_min, lon_max, lat_max (EPSG:4326)")
    created_at: datetime


class JobIn(BaseModel):
    """Submit an analysis request."""

    scene_id: str
    aoi: dict[str, Any] = Field(description="GeoJSON Polygon in EPSG:4326")
    task: str
    sensor_mode: Sensor
    priority: int = Field(0, ge=-100, le=100)


class JobAccepted(BaseModel):
    id: str
    status: str


class JobOut(BaseModel):
    id: str
    scene_id: str
    model_id: str
    task: str
    priority: int
    status: str
    n_tiles: int | None
    worker_id: str | None
    error: str | None
    timings: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ModelOut(BaseModel):
    id: str
    sensor_mode: Sensor
    in_channels: int
    threshold: float
    sha256: str
    path: str
