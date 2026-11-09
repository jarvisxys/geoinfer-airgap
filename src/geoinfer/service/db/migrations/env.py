"""Alembic environment: the URL comes from the caller (``migrate.upgrade``) or
``GEOINFER_DB_URL``; never from a committed file."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine

from geoinfer.service.db.schema import Base

target_metadata = Base.metadata
url = context.config.get_main_option("sqlalchemy.url") or os.environ["GEOINFER_DB_URL"]
engine = create_engine(url)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
