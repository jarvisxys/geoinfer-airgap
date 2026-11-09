"""Service settings: ``configs/service.yaml`` with ``GEOINFER_*`` environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SERVICE_CONFIG = Path("configs/service.yaml")


@dataclass(frozen=True)
class Settings:
    """Runtime settings of the API and worker."""

    db_url: str
    inbox: Path
    outputs: Path
    models_dir: Path
    readiness_max_heartbeat_age_s: float
    tasks: tuple[str, ...]
    default_models: dict[str, str]


def load_settings(path: Path | None = None) -> Settings:
    """Read the config file; ``GEOINFER_DB_URL`` is required, the paths may be overridden."""
    p = Path(os.environ.get("GEOINFER_SERVICE_CONFIG", path or DEFAULT_SERVICE_CONFIG))
    raw: dict[str, Any] = yaml.safe_load(p.read_text())
    url = os.environ.get("GEOINFER_DB_URL")
    if not url:
        raise RuntimeError("GEOINFER_DB_URL is not set")
    return Settings(
        db_url=url,
        inbox=Path(os.environ.get("GEOINFER_INBOX", raw["inbox"])),
        outputs=Path(os.environ.get("GEOINFER_OUTPUTS", raw["outputs"])),
        models_dir=Path(os.environ.get("GEOINFER_MODELS_DIR", raw["models_dir"])),
        readiness_max_heartbeat_age_s=float(raw["readiness_max_heartbeat_age_s"]),
        tasks=tuple(raw["tasks"]),
        default_models=dict(raw["default_models"]),
    )
