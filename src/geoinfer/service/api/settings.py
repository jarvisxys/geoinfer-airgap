"""Service settings: ``configs/service.yaml`` with ``GEOINFER_*`` environment overrides."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SERVICE_CONFIG = Path("configs/service.yaml")


def _env(name: str, default: Any) -> Any:
    """``GEOINFER_<name>`` if set and non-empty (Compose passes unset variables as "")."""
    v = os.environ.get(f"GEOINFER_{name}", "")
    return v if v != "" else default


@dataclass(frozen=True)
class SchedulingSettings:
    """Tile scheduling (``worker.scheduling``; experiments E2 and E3)."""

    batching: str  # per_job | cross_job
    queue: str  # fifo | priority (with ageing)
    ageing_per_s: float  # priority points gained per second waited
    max_wait_ms: float  # hold a partial batch this long while it can grow
    max_active_jobs: int  # jobs claimed and in flight at once
    prefetch_tiles: int  # read-ahead per job (normalised tiles in RAM)


@dataclass(frozen=True)
class WorkerSettings:
    """Worker loop and inference settings (``worker:`` in the service config)."""

    worker_id: str
    poll_interval_s: float
    heartbeat_interval_s: float
    batch_size: int
    providers: tuple[str, ...]
    max_aoi_mpx: float
    scratch_dir: Path
    metrics_port: int
    scheduling: SchedulingSettings


@dataclass(frozen=True)
class UiSettings:
    """Web UI (``ui:`` in the service config)."""

    max_upload_gb: float
    prepare_gsd_m: float
    tile_cache: int


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
    stats_file: Path
    preprocess_config: Path
    worker: WorkerSettings
    ui: UiSettings


def load_settings(path: Path | None = None) -> Settings:
    """Read the config file; ``GEOINFER_DB_URL`` is required, the paths may be overridden.

    ``stats_file`` is relative to ``models_dir``. The worker id is ``GEOINFER_WORKER_ID``
    or the host name (the container name under Compose).
    """
    p = Path(os.environ.get("GEOINFER_SERVICE_CONFIG", path or DEFAULT_SERVICE_CONFIG))
    raw: dict[str, Any] = yaml.safe_load(p.read_text())
    url = os.environ.get("GEOINFER_DB_URL")
    if not url:
        raise RuntimeError("GEOINFER_DB_URL is not set")
    w = raw["worker"]
    sc = w["scheduling"]
    return Settings(
        db_url=url,
        inbox=Path(os.environ.get("GEOINFER_INBOX", raw["inbox"])),
        outputs=Path(os.environ.get("GEOINFER_OUTPUTS", raw["outputs"])),
        models_dir=Path(os.environ.get("GEOINFER_MODELS_DIR", raw["models_dir"])),
        readiness_max_heartbeat_age_s=float(raw["readiness_max_heartbeat_age_s"]),
        tasks=tuple(raw["tasks"]),
        default_models=dict(raw["default_models"]),
        stats_file=Path(raw["stats_file"]),
        preprocess_config=Path(raw["preprocess_config"]),
        worker=WorkerSettings(
            worker_id=os.environ.get("GEOINFER_WORKER_ID", socket.gethostname()),
            poll_interval_s=float(w["poll_interval_s"]),
            heartbeat_interval_s=float(w["heartbeat_interval_s"]),
            batch_size=int(_env("BATCH_SIZE", w["batch_size"])),
            providers=tuple(w["providers"]),
            max_aoi_mpx=float(w["max_aoi_mpx"]),
            scratch_dir=Path(os.environ.get("GEOINFER_SCRATCH", w["scratch_dir"])),
            metrics_port=int(w["metrics_port"]),
            # GEOINFER_BATCHING, _QUEUE, _AGEING_PER_S, _MAX_WAIT_MS, _MAX_ACTIVE_JOBS:
            # the load-test driver switches policies without editing the config
            scheduling=SchedulingSettings(
                batching=str(_env("BATCHING", sc["batching"])),
                queue=str(_env("QUEUE", sc["queue"])),
                ageing_per_s=float(_env("AGEING_PER_S", sc["ageing_per_s"])),
                max_wait_ms=float(_env("MAX_WAIT_MS", sc["max_wait_ms"])),
                max_active_jobs=int(_env("MAX_ACTIVE_JOBS", sc["max_active_jobs"])),
                prefetch_tiles=int(sc["prefetch_tiles"]),
            ),
        ),
        ui=UiSettings(
            max_upload_gb=float(raw["ui"]["max_upload_gb"]),
            prepare_gsd_m=float(raw["ui"]["prepare_gsd_m"]),
            tile_cache=int(raw["ui"]["tile_cache"]),
        ),
    )
