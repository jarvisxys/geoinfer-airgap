"""Run the Alembic migrations shipped inside the package (no alembic.ini needed)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS = Path(__file__).with_name("migrations")


def _config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    """Migrate the database at ``url`` up to ``revision``."""
    command.upgrade(_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    """Migrate the database at ``url`` down to ``revision``."""
    command.downgrade(_config(url), revision)


def main() -> None:
    """``python -m geoinfer.service.db.migrate``: upgrade ``GEOINFER_DB_URL`` to head."""
    import os

    url = os.environ.get("GEOINFER_DB_URL")
    if not url:
        raise SystemExit("GEOINFER_DB_URL is not set")
    upgrade(url)
    print("database at head")


if __name__ == "__main__":
    main()
