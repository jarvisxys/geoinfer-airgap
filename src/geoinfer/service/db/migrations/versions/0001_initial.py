"""Initial schema: scenes, models, jobs, buildings, workers (plan section 3).

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    now = sa.text("now()")
    op.create_table(
        "scenes",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("sar_path", sa.Text),
        sa.Column("sensor", sa.Text, nullable=False),
        sa.Column("footprint", Geometry("POLYGON", srid=4326, spatial_index=False), nullable=False),
        sa.Column("gsd_m", sa.Float, nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("crs", sa.Text, nullable=False),
        sa.Column("orientation", sa.SmallInteger),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now),
        sa.CheckConstraint("sensor IN ('optical', 'sar', 'fused')", name="scenes_sensor"),
        sa.CheckConstraint(
            "orientation IS NULL OR orientation IN (0, 1)", name="scenes_orientation"
        ),
    )
    op.create_table(
        "models",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("sensor_mode", sa.Text, nullable=False),
        sa.Column("in_channels", sa.SmallInteger, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("norm", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now),
        sa.CheckConstraint("sensor_mode IN ('optical', 'sar', 'fused')", name="models_sensor"),
    )
    op.create_table(
        "jobs",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("scene_id", sa.Text, sa.ForeignKey("scenes.id"), nullable=False),
        sa.Column("model_id", sa.Text, sa.ForeignKey("models.id"), nullable=False),
        sa.Column("aoi", Geometry("POLYGON", srid=4326, spatial_index=False), nullable=False),
        sa.Column("task", sa.Text, nullable=False),
        sa.Column("priority", sa.SmallInteger, nullable=False, server_default="0"),
        sa.Column("status", sa.Text, nullable=False, server_default="queued"),
        sa.Column("n_tiles", sa.Integer),
        sa.Column("worker_id", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("timings", JSONB),
        sa.Column("result_dir", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=now),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'ready', 'failed')", name="jobs_status"
        ),
    )
    op.execute("CREATE INDEX jobs_queue_idx ON jobs (status, priority DESC, created_at)")
    op.create_table(
        "buildings",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("geom", Geometry("POLYGON", srid=4326, spatial_index=False), nullable=False),
        sa.Column("area_m2", sa.Float),
        sa.Column("score", sa.Float),
    )
    op.execute("CREATE INDEX buildings_geom_idx ON buildings USING GIST (geom)")
    op.create_table(
        "workers",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("gpu", sa.Text),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False, server_default=now),
        sa.Column("jobs_done", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    for t in ("workers", "buildings", "jobs", "models", "scenes"):
        op.drop_table(t)
