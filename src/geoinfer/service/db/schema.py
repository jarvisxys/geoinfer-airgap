"""Database schema of the offline service (plan section 3), SQLAlchemy 2 + GeoAlchemy2.

Tables: ``scenes`` (registered COGs), ``models`` (registry entries), ``jobs`` (queue and
history), ``buildings`` (result polygons), ``workers`` (heartbeats for ``/readyz``).
Additions to the plan's schema: ``scenes.sar_path`` (the SAR COG of a fused scene, on the
optical grid), ``scenes.orientation`` (SAR look direction, decision D5), ``models.threshold``
and ``models.norm`` (tuned threshold and normalisation arms), ``jobs.timings`` and
``jobs.result_dir``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SENSORS = ("optical", "sar", "fused")
STATUSES = ("queued", "running", "ready", "failed")


class Base(DeclarativeBase):
    """Declarative base of the service tables."""


class Scene(Base):
    __tablename__ = "scenes"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    sar_path: Mapped[str | None] = mapped_column(Text)
    sensor: Mapped[str] = mapped_column(Text)
    footprint: Mapped[Any] = mapped_column(Geometry("POLYGON", srid=4326))
    gsd_m: Mapped[float] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    crs: Mapped[str] = mapped_column(Text)
    orientation: Mapped[int | None] = mapped_column(SmallInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        CheckConstraint(f"sensor IN {SENSORS}", name="scenes_sensor"),
        CheckConstraint("orientation IS NULL OR orientation IN (0, 1)", name="scenes_orientation"),
    )


class Model(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    sensor_mode: Mapped[str] = mapped_column(Text)
    in_channels: Mapped[int] = mapped_column(SmallInteger)
    threshold: Mapped[float] = mapped_column(Float)
    norm: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (CheckConstraint(f"sensor_mode IN {SENSORS}", name="models_sensor"),)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"))
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"))
    aoi: Mapped[Any] = mapped_column(Geometry("POLYGON", srid=4326))
    task: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(SmallInteger, server_default="0")
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    n_tiles: Mapped[int | None] = mapped_column(Integer)
    worker_id: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    timings: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    result_dir: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (CheckConstraint(f"status IN {STATUSES}", name="jobs_status"),)


# the worker's claim query scans this: queued jobs by priority, then age
Index("jobs_queue_idx", Job.status, Job.priority.desc(), Job.created_at)


class Building(Base):
    __tablename__ = "buildings"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    geom: Mapped[Any] = mapped_column(Geometry("POLYGON", srid=4326, spatial_index=True))
    area_m2: Mapped[float | None] = mapped_column(Float)
    score: Mapped[float | None] = mapped_column(Float)


class Worker(Base):
    __tablename__ = "workers"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    gpu: Mapped[str | None] = mapped_column(Text)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    jobs_done: Mapped[int] = mapped_column(Integer, server_default="0")
