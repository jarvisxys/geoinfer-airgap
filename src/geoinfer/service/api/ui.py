"""Routes behind the web UI (``/ui``): status, inbox and uploads, map tiles, job overlays.

The page itself is static (``geoinfer/service/ui/static``) and loads nothing from the
internet. Uploads stream the request body into the inbox (local file drop through the
browser; the air gap is unchanged) and are checked with :func:`imagery.inspect_image`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from geoinfer.io.config import load_io_config
from geoinfer.io.raster import inspect_raster
from geoinfer.service.api import imagery
from geoinfer.service.api.settings import Settings
from geoinfer.service.db.schema import Job, Model, Scene

STATIC = Path(__file__).resolve().parents[1] / "ui" / "static"
MODES = {"optical": ["optical"], "sar": ["sar"], "fused": ["fused", "optical", "sar"]}
CHUNK = 4 << 20


class ConvertIn(BaseModel):
    to: str  # "lonlat" or "px"
    points: list[list[float]]


def add_ui_routes(app: FastAPI, cfg: Settings, db: Callable[[], Iterator[Session]]) -> None:
    """Mount the static UI and its API routes on ``app``."""
    inbox = cfg.inbox
    prep_state: dict[str, dict[str, Any]] = {}
    prep_lock = threading.Lock()

    @lru_cache(maxsize=64)
    def _stretch(path: str, mtime: float, layer: str) -> imagery.Stretch:
        return imagery.stretch_stats(path, layer)

    @lru_cache(maxsize=cfg.ui.tile_cache)
    def _tile(path: str, mtime: float, layer: str, z: int, x: int, y: int) -> bytes:
        return imagery.tile_png(path, z, x, y, layer, _stretch(path, mtime, layer))

    def _train_optical() -> dict[str, Any] | None:
        p = cfg.models_dir / cfg.stats_file
        try:
            return dict(json.loads(p.read_text())["optical"])
        except (OSError, KeyError, ValueError):
            return None

    def _in_inbox(name: str) -> Path:
        p = (inbox / name).resolve()
        if "/" in name or "\\" in name or not p.is_relative_to(inbox.resolve()):
            raise HTTPException(400, "file name must be a plain name inside the inbox")
        return p

    def _scene(s: Session, scene_id: str) -> Scene:
        sc = s.get(Scene, scene_id)
        if sc is None:
            raise HTTPException(404, "no such scene")
        return sc

    def _layers(sc: Scene) -> dict[str, Path]:
        if sc.sensor == "sar":
            return {"sar": inbox / sc.path}
        out = {"optical": inbox / sc.path}
        if sc.sar_path:
            out["sar"] = inbox / sc.sar_path
        return out

    # -- page --------------------------------------------------------------------------

    app.mount("/ui", StaticFiles(directory=STATIC, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    # -- status ------------------------------------------------------------------------

    @app.get("/v1/status", tags=["ui"])
    def status(s: Session = Depends(db)) -> dict[str, Any]:
        workers = [
            {"id": w, "gpu": g, "last_seen_s": round(float(a), 1), "jobs_done": n}
            for w, g, a, n in s.execute(
                text(
                    "SELECT id, gpu, extract(epoch from now() - last_seen), jobs_done "
                    "FROM workers ORDER BY last_seen DESC"
                )
            )
        ]
        queue = dict.fromkeys(("queued", "running", "ready", "failed"), 0)
        for st, n in s.execute(text("SELECT status, count(*) FROM jobs GROUP BY status")):
            queue[st] = n
        alive = any(w["last_seen_s"] < cfg.readiness_max_heartbeat_age_s for w in workers)
        return {
            "db": True,
            "worker": alive,
            "workers": workers,
            "queue": queue,
            "models": [m for (m,) in s.execute(select(Model.id).order_by(Model.id))],
            "max_aoi_px": int(cfg.worker.max_aoi_mpx * 1e6),
            "max_upload_gb": cfg.ui.max_upload_gb,
        }

    # -- inbox and uploads -------------------------------------------------------------

    @app.get("/v1/inbox", tags=["ui"])
    def list_inbox(s: Session = Depends(db)) -> list[dict[str, Any]]:
        used: dict[str, list[str]] = {}
        for sid, p, sp in s.execute(select(Scene.id, Scene.path, Scene.sar_path)):
            for f in (p, sp):
                if f:
                    used.setdefault(f, []).append(sid)
        out = []
        for fp in sorted(inbox.glob("*.tif*")):
            st = fp.stat()
            out.append(
                {
                    "name": fp.name,
                    "size_mb": round(st.st_size / 2**20, 1),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                    "scenes": used.get(fp.name, []),
                }
            )
        return out

    @app.put("/v1/inbox/{name}", tags=["ui"], status_code=201)
    async def upload(name: str, request: Request, overwrite: bool = False) -> dict[str, Any]:
        try:
            safe = imagery.safe_upload_name(name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        dst = inbox / safe
        if dst.exists() and not overwrite:
            raise HTTPException(409, f"{safe} is already in the inbox")
        limit = int(cfg.ui.max_upload_gb * 2**30)
        part = inbox / f".{safe}.part"
        size = 0
        try:
            with part.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"larger than {cfg.ui.max_upload_gb} GB")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty upload")
            os.replace(part, dst)
        finally:
            part.unlink(missing_ok=True)
        _tile.cache_clear()
        return {"name": safe, "size_mb": round(size / 2**20, 1)}

    @app.delete("/v1/inbox/{name}", tags=["ui"], status_code=204)
    def delete_file(name: str, s: Session = Depends(db)) -> Response:
        p = _in_inbox(name)
        if not p.is_file():
            raise HTTPException(404, "not in the inbox")
        hit = s.execute(
            select(Scene.id).where((Scene.path == name) | (Scene.sar_path == name))
        ).first()
        if hit is not None:
            raise HTTPException(409, f"used by scene {hit[0]}")
        p.unlink()
        return Response(status_code=204)

    @app.get("/v1/inbox/{name}/inspect", tags=["ui"])
    def inspect(name: str) -> dict[str, Any]:
        p = _in_inbox(name)
        if not p.is_file():
            raise HTTPException(404, "not in the inbox")
        return imagery.inspect_image(p, _train_optical()) | {"name": name}

    def _prepare(src: Path, dst: Path) -> None:
        try:
            imagery.prepare_image(src, dst, cfg.ui.prepare_gsd_m, load_io_config())
            with prep_lock:
                prep_state[src.name] = {"state": "done", "output": dst.name}
        except Exception as e:
            with prep_lock:
                prep_state[src.name] = {"state": "failed", "error": f"{type(e).__name__}: {e}"}

    @app.post("/v1/inbox/{name}/prepare", tags=["ui"], status_code=202)
    def prepare(name: str) -> dict[str, Any]:
        src = _in_inbox(name)
        if not src.is_file():
            raise HTTPException(404, "not in the inbox")
        dst = inbox / f"{src.stem}_prepared.tif"
        with prep_lock:
            if prep_state.get(name, {}).get("state") == "running":
                return prep_state[name]
            prep_state[name] = {"state": "running", "output": dst.name}
        threading.Thread(target=_prepare, args=(src, dst), daemon=True).start()
        return prep_state[name]

    @app.get("/v1/inbox/{name}/prepare", tags=["ui"])
    def prepare_status(name: str) -> dict[str, Any]:
        with prep_lock:
            return prep_state.get(name, {"state": "idle"})

    # -- map ---------------------------------------------------------------------------

    @app.get("/v1/scenes/{scene_id}/map", tags=["ui"])
    def scene_map(scene_id: str, s: Session = Depends(db)) -> dict[str, Any]:
        sc = _scene(s, scene_id)
        layers = _layers(sc)
        info = inspect_raster(next(iter(layers.values())))
        t = info.transform
        epsg = info.crs.to_epsg() if info.crs else None
        return {
            "id": sc.id,
            "sensor": sc.sensor,
            "orientation": sc.orientation,
            "layers": list(layers),
            "modes": MODES[sc.sensor],
            "gsd_m": sc.gsd_m,
            "crs": sc.crs,
            "epsg": epsg,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            **imagery.pyramid(info.width, info.height),
        }

    @app.get("/v1/scenes/{scene_id}/tiles/{layer}/{z}/{x}/{y}.png", tags=["ui"])
    def scene_tile(
        scene_id: str, layer: str, z: int, x: int, y: int, s: Session = Depends(db)
    ) -> Response:
        layers = _layers(_scene(s, scene_id))
        if layer not in layers:
            raise HTTPException(404, f"layer must be one of {list(layers)}")
        p = layers[layer]
        try:
            png = _tile(str(p), p.stat().st_mtime, layer, z, x, y)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/v1/scenes/{scene_id}/convert", tags=["ui"])
    def convert(scene_id: str, body: ConvertIn, s: Session = Depends(db)) -> dict[str, Any]:
        sc = _scene(s, scene_id)
        info = inspect_raster(next(iter(_layers(sc).values())))
        if body.to == "lonlat":
            pts = imagery.px_to_lonlat(body.points, info.transform, info.crs)
        elif body.to == "px":
            pts = imagery.lonlat_to_px(body.points, info.transform, info.crs)
        else:
            raise HTTPException(422, "to must be 'lonlat' or 'px'")
        return {"points": [[round(a, 9), round(b, 9)] for a, b in pts]}

    # -- jobs --------------------------------------------------------------------------

    @app.get("/v1/jobs", tags=["ui"])
    def list_jobs(
        s: Session = Depends(db), limit: int = Query(50, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        rows = s.execute(
            select(Job, Model.sensor_mode)
            .join(Model, Model.id == Job.model_id)
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        return [
            {
                "id": str(j.id),
                "scene_id": j.scene_id,
                "sensor_mode": mode,
                "priority": j.priority,
                "status": j.status,
                "n_tiles": j.n_tiles,
                "error": j.error,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "total_s": (j.timings or {}).get("total_s"),
            }
            for j, mode in rows
        ]

    @app.get("/v1/jobs/{job_id}/overlay", tags=["ui"])
    def overlay(job_id: str, s: Session = Depends(db)) -> dict[str, Any]:
        import uuid

        try:
            job = s.get(Job, uuid.UUID(job_id))
        except ValueError as e:
            raise HTTPException(404, "no such job") from e
        if job is None:
            raise HTTPException(404, "no such job")
        if job.status != "ready" or not job.result_dir:
            raise HTTPException(409, f"job is {job.status}")
        folder = Path(job.result_dir).resolve()
        if not folder.is_relative_to(cfg.outputs.resolve()):
            raise HTTPException(500, "result outside the outputs folder")
        feats = json.loads((folder / "buildings.geojson").read_text())["features"]
        sc = _scene(s, job.scene_id)
        info = inspect_raster(next(iter(_layers(sc).values())))
        rings = imagery.overlay_rings(feats, info.transform, info.crs)
        return {
            "job_id": job_id,
            "scene_id": sc.id,
            "count": len(rings),
            "total_area_m2": round(sum(r["area_m2"] or 0 for r in rings), 1),
            "buildings": rings,
        }
