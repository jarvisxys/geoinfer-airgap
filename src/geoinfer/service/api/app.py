"""FastAPI application of the offline service (plan section 3).

CPU only: it validates requests, registers scenes and queues jobs in PostGIS; the GPU
worker does the imagery work. No route makes an outbound network call.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import shapely
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from geoalchemy2.shape import to_shape
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pyproj import Transformer
from shapely.geometry import box, shape
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from geoinfer.io.raster import inspect_raster
from geoinfer.service.api.schemas import (
    JobAccepted,
    JobIn,
    JobOut,
    ModelOut,
    SceneIn,
    SceneOut,
    Sensor,
)
from geoinfer.service.api.settings import Settings, load_settings
from geoinfer.service.db import queue
from geoinfer.service.db.schema import Job, Model, Scene

# which scene sensors can serve a job's sensor mode
SERVES = {"optical": {"optical", "fused"}, "sar": {"sar", "fused"}, "fused": {"fused"}}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inbox_file(inbox: Path, name: str) -> Path:
    """Resolve ``name`` inside the inbox; refuse anything that escapes it."""
    p = (inbox / name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(inbox.resolve()):
        raise HTTPException(400, f"path must be a file inside the inbox: {name}")
    if not p.is_file():
        raise HTTPException(404, f"not found in the inbox: {name}")
    return p


def _scene_out(s: Scene) -> SceneOut:
    return SceneOut(
        id=s.id,
        path=s.path,
        sar_path=s.sar_path,
        sensor=cast(Sensor, s.sensor),
        orientation=s.orientation,
        gsd_m=s.gsd_m,
        width=s.width,
        height=s.height,
        crs=s.crs,
        sha256=s.sha256,
        bounds=[round(v, 7) for v in to_shape(s.footprint).bounds],
        created_at=s.created_at,
    )


def _job_out(j: Job) -> JobOut:
    return JobOut(
        id=str(j.id),
        scene_id=j.scene_id,
        model_id=j.model_id,
        task=j.task,
        priority=j.priority,
        status=j.status,
        n_tiles=j.n_tiles,
        worker_id=j.worker_id,
        error=j.error,
        timings=j.timings,
        created_at=j.created_at,
        started_at=j.started_at,
        finished_at=j.finished_at,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API; settings default to ``load_settings()``."""
    cfg = settings or load_settings()
    engine = create_engine(cfg.db_url, pool_pre_ping=True)
    registry = CollectorRegistry()
    submitted = Counter(
        "geoinfer_jobs_submitted", "Jobs accepted", ["sensor_mode"], registry=registry
    )
    depth = Gauge("geoinfer_queue_depth", "Jobs per status", ["status"], registry=registry)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()

    app = FastAPI(title="geoinfer", version="0.3", lifespan=lifespan)

    def db() -> Iterator[Session]:
        with Session(engine) as s:
            yield s

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(s: Session = Depends(db)) -> JSONResponse:
        try:
            s.execute(text("select 1"))
            db_ok = True
            worker = queue.worker_alive(s, cfg.readiness_max_heartbeat_age_s)
        except Exception:
            db_ok, worker = False, False
        body = {"db": db_ok, "worker": worker}
        return JSONResponse(body, status_code=200 if db_ok and worker else 503)

    @app.post("/v1/scenes", status_code=201)
    def register_scene(body: SceneIn, s: Session = Depends(db)) -> SceneOut:
        if s.get(Scene, body.id) is not None:
            raise HTTPException(409, f"scene {body.id} exists")
        if body.sensor in ("sar", "fused") and body.orientation is None:
            raise HTTPException(
                422, "SAR and fused scenes need their look direction (orientation 0 or 1)"
            )
        if body.sensor == "fused" and not body.sar_path:
            raise HTTPException(422, "fused scenes need sar_path")
        path = _inbox_file(cfg.inbox, body.path)
        info = inspect_raster(path)
        if info.crs is None or not info.crs.is_projected:
            raise HTTPException(422, "scene needs a projected CRS")
        if body.sar_path:
            sar = inspect_raster(_inbox_file(cfg.inbox, body.sar_path))
            same = (sar.width, sar.height, sar.crs) == (info.width, info.height, info.crs)
            if not (same and sar.transform.almost_equals(info.transform, precision=1e-3)):
                raise HTTPException(422, "sar_path is not on the optical grid")
        t = info.transform
        footprint = box(t.c, t.f + t.e * info.height, t.c + t.a * info.width, t.f)
        to_wgs = Transformer.from_crs(info.crs, "EPSG:4326", always_xy=True)
        fp = shapely.transform(
            footprint.segmentize(abs(t.a) * 64),
            lambda xy: np.column_stack(to_wgs.transform(xy[:, 0], xy[:, 1])),
        )
        scene = Scene(
            id=body.id,
            path=body.path,
            sar_path=body.sar_path,
            sensor=body.sensor,
            footprint=f"SRID=4326;{fp.wkt}",
            gsd_m=abs(t.a),
            width=info.width,
            height=info.height,
            crs=info.crs.to_string(),
            orientation=body.orientation,
            sha256=_sha256(path),
        )
        s.add(scene)
        s.commit()
        s.refresh(scene)
        return _scene_out(scene)

    @app.get("/v1/scenes")
    def list_scenes(s: Session = Depends(db)) -> list[SceneOut]:
        return [_scene_out(x) for x in s.scalars(select(Scene).order_by(Scene.created_at))]

    @app.post("/v1/jobs", status_code=202)
    def submit_job(body: JobIn, s: Session = Depends(db)) -> JobAccepted:
        scene = s.get(Scene, body.scene_id)
        if scene is None:
            raise HTTPException(404, f"no scene {body.scene_id}")
        if body.task not in cfg.tasks:
            raise HTTPException(422, f"task must be one of {list(cfg.tasks)}")
        if scene.sensor not in SERVES[body.sensor_mode]:
            raise HTTPException(422, f"a {scene.sensor} scene cannot serve {body.sensor_mode}")
        try:
            aoi = shape(body.aoi)
        except Exception as e:
            raise HTTPException(422, f"aoi is not valid GeoJSON: {e}") from e
        if aoi.geom_type != "Polygon" or not aoi.is_valid or aoi.is_empty:
            raise HTTPException(422, "aoi must be a valid GeoJSON Polygon (EPSG:4326)")
        hit = s.execute(
            select(func.ST_Intersects(func.ST_GeomFromText(aoi.wkt, 4326), Scene.footprint)).where(
                Scene.id == scene.id
            )
        ).scalar_one()
        if not hit:
            raise HTTPException(422, "aoi does not intersect the scene footprint")
        model_id = cfg.default_models[body.sensor_mode]
        if s.get(Model, model_id) is None:
            raise HTTPException(409, f"model {model_id} for {body.sensor_mode} is not registered")
        job = Job(
            scene_id=scene.id,
            model_id=model_id,
            aoi=f"SRID=4326;{aoi.wkt}",
            task=body.task,
            priority=body.priority,
        )
        s.add(job)
        s.commit()
        submitted.labels(body.sensor_mode).inc()
        return JobAccepted(id=str(job.id), status=job.status)

    def _get_job(s: Session, job_id: str) -> Job:
        try:
            jid = uuid.UUID(job_id)
        except ValueError as e:
            raise HTTPException(404, "no such job") from e
        job = s.get(Job, jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str, s: Session = Depends(db)) -> JobOut:
        return _job_out(_get_job(s, job_id))

    @app.get("/v1/jobs/{job_id}/result", response_model=None)
    def job_result(
        job_id: str, s: Session = Depends(db), format: Literal["geojson", "cog"] = Query("geojson")
    ) -> FileResponse:
        job = _get_job(s, job_id)
        if job.status != "ready" or job.result_dir is None:
            raise HTTPException(409, f"job is {job.status}")
        folder = Path(job.result_dir).resolve()
        if not folder.is_relative_to(cfg.outputs.resolve()):
            raise HTTPException(500, "result outside the outputs folder")
        if format == "geojson":
            return FileResponse(folder / "buildings.geojson", media_type="application/geo+json")
        return FileResponse(
            folder / "mask.tif", media_type="image/tiff", filename=f"{job.id}_mask.tif"
        )

    @app.get("/v1/models")
    def list_models(s: Session = Depends(db)) -> list[ModelOut]:
        return [
            ModelOut(
                id=m.id,
                sensor_mode=cast(Sensor, m.sensor_mode),
                in_channels=m.in_channels,
                threshold=m.threshold,
                sha256=m.sha256,
                path=m.path,
            )
            for m in s.scalars(select(Model).order_by(Model.id))
        ]

    @app.get("/metrics")
    def metrics(s: Session = Depends(db)) -> Response:
        for status in ("queued", "running", "ready", "failed"):
            depth.labels(status).set(0)
        for status, n in s.execute(text("select status, count(*) from jobs group by status")):
            depth.labels(status).set(n)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> Any:
    """``uvicorn geoinfer.service.api.app:app`` entry (settings from the environment)."""
    return create_app()
