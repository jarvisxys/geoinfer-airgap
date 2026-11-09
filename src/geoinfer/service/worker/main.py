"""The inference worker: claim a queued job, run it, store the result (plan section 3).

Loop: claim the next job (``FOR UPDATE SKIP LOCKED``), check the model file against its
registry hash, build the model's inputs from the scene, run
:func:`~geoinfer.service.worker.pipeline.segment_aoi`, insert the building polygons and
mark the job ready, all in one transaction; on any error the job is marked failed with
the message. A background thread sends heartbeats so ``/readyz`` stays green during long
jobs. On start-up, jobs this worker id still holds from a crashed run are re-queued.
SIGTERM / SIGINT finish the current job, then exit.

Usage: python -m geoinfer.service.worker   (settings: configs/service.yaml + GEOINFER_*)
"""

from __future__ import annotations

import json
import logging
import math
import signal
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Any

import yaml
from geoalchemy2.shape import from_shape, to_shape
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from geoinfer.io.config import IOConfig, load_io_config
from geoinfer.postprocess.config import PostprocessConfig, load_postprocess_config
from geoinfer.service.api.settings import Settings, load_settings
from geoinfer.service.db import queue
from geoinfer.service.db.schema import Building, Job, Model, Scene
from geoinfer.service.registry import sha256_file
from geoinfer.service.worker.pipeline import SceneInputs, segment_aoi
from geoinfer.service.worker.predictor import OnnxPredictor, Predictor
from geoinfer.tiling.config import TilingConfig, load_tiling_config
from geoinfer.train.dataset import Normalizer

log = logging.getLogger("geoinfer.worker")

CHANNELS = {"optical": 4, "sar": 4, "fused": 8}


def scene_inputs(scene: Scene, mode: str, inbox: Path) -> SceneInputs:
    """The rasters a ``mode`` model reads from ``scene`` (paths are inbox-relative).

    A fused scene serves all three modes; optical and SAR scenes serve their own.
    """
    if mode not in CHANNELS:
        raise ValueError(f"unknown sensor mode {mode!r}")
    if scene.sensor != "fused" and scene.sensor != mode:
        raise ValueError(f"a {scene.sensor} scene cannot serve a {mode} model")
    opt = inbox / scene.path if mode in ("optical", "fused") else None
    if mode == "optical":
        sar = None
    elif scene.sensor == "sar":
        sar = inbox / scene.path
    else:
        if scene.sar_path is None:
            raise ValueError(f"fused scene {scene.id} has no sar_path")
        sar = inbox / scene.sar_path
    return SceneInputs(mode, opt, sar, scene.orientation)


@dataclass
class _Loaded:
    sha256: str
    predictor: Predictor
    norm: Normalizer


def _gpu_name() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        finally:
            pynvml.nvmlShutdown()
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None


class Worker:
    """Job loop of one inference worker.

    Parameters
    ----------
    settings
        Service settings (``worker:`` section, paths, database URL).
    tcfg, pcfg, io_cfg
        Tiling, post-processing and I/O configs; the repo files by default.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        tcfg: TilingConfig | None = None,
        pcfg: PostprocessConfig | None = None,
        io_cfg: IOConfig | None = None,
    ) -> None:
        self.settings = settings
        self.id = settings.worker.worker_id
        self.engine = create_engine(settings.db_url, pool_pre_ping=True)
        self.tcfg = replace(tcfg or load_tiling_config(), scratch_dir=settings.worker.scratch_dir)
        self.pcfg = pcfg or load_postprocess_config()
        self.io_cfg = io_cfg or load_io_config()
        self.stop = threading.Event()
        self.gpu = _gpu_name()
        self._models: dict[str, _Loaded] = {}
        self._stats: dict[str, Any] | None = None
        self._sar_arms: dict[str, Any] | None = None

    # -- database helpers -------------------------------------------------------------

    def beat(self) -> None:
        """Send one heartbeat."""
        with Session(self.engine) as s:
            queue.heartbeat(s, self.id, self.gpu)
            s.commit()

    def requeue_orphans(self) -> int:
        """Re-queue jobs this worker id held when its previous process died."""
        with Session(self.engine) as s:
            n = queue.requeue_orphans(s, self.id)
            s.commit()
        return n

    # -- models ------------------------------------------------------------------------

    def _load(self, model: Model) -> _Loaded:
        """Predictor and normaliser of a registry entry, cached while its hash holds."""
        path = (self.settings.models_dir / model.path).resolve()
        if not path.is_relative_to(self.settings.models_dir.resolve()):
            raise ValueError(f"model path {model.path} escapes the registry")
        cached = self._models.get(model.id)
        if cached is not None and cached.sha256 == model.sha256:
            return cached
        digest = sha256_file(path)
        if digest != model.sha256:
            raise ValueError(
                f"model file {model.path} sha256 {digest[:12]}... does not match the "
                f"registry ({model.sha256[:12]}...)"
            )
        if self._stats is None or self._sar_arms is None:
            stats_path = self.settings.models_dir / self.settings.stats_file
            self._stats = json.loads(stats_path.read_text())
            pre = yaml.safe_load(self.settings.preprocess_config.read_text())
            self._sar_arms = dict(pre["sar"]["arms"])
        norm_cfg = model.norm or {}
        norm = Normalizer(
            self._stats,
            None,  # per-strip statistics are unknown for a new scene (Phase 3 decision 2)
            str(norm_cfg.get("optical", "global")),
            str(norm_cfg.get("sar", "db_lee7")),
            self._sar_arms,
        )
        pred = OnnxPredictor(path, self.settings.worker.providers)
        if pred.in_channels != model.in_channels:
            raise ValueError(
                f"{model.id}: ONNX input has {pred.in_channels} channels, "
                f"registry says {model.in_channels}"
            )
        log.info("loaded %s on %s", model.id, pred.providers[0])
        loaded = _Loaded(model.sha256, pred, norm)
        self._models[model.id] = loaded
        return loaded

    # -- jobs --------------------------------------------------------------------------

    def _run(self, s: Session, job: Job) -> None:
        scene = s.get(Scene, job.scene_id)
        model = s.get(Model, job.model_id)
        if scene is None or model is None:
            raise ValueError("job references a missing scene or model")
        if CHANNELS[model.sensor_mode] != model.in_channels:
            raise ValueError(
                f"{model.id}: {model.sensor_mode} models take "
                f"{CHANNELS[model.sensor_mode]} channels"
            )
        loaded = self._load(model)
        inputs = scene_inputs(scene, model.sensor_mode, self.settings.inbox)
        out_dir = self.settings.outputs / str(job.id)
        w = self.settings.worker
        res = segment_aoi(
            inputs,
            to_shape(job.aoi),
            loaded.predictor,
            loaded.norm,
            float(model.threshold),
            self.tcfg,
            self.pcfg,
            self.io_cfg,
            out_dir,
            batch_size=w.batch_size,
            max_aoi_px=int(w.max_aoi_mpx * 1e6),
        )
        s.add_all(
            Building(
                job_id=job.id,
                geom=from_shape(g, srid=4326),
                area_m2=float(a),
                score=None if math.isnan(sc) else float(sc),
            )
            for g, a, sc in zip(
                res.buildings.geometry,
                res.buildings["area_m2"],
                res.buildings["score"],
                strict=True,
            )
        )
        queue.finish_job(s, job, res.n_tiles, res.timings, str(out_dir))
        log.info(
            "job %s ready: %d buildings, %d tiles, %.1f s",
            job.id,
            len(res.buildings),
            res.n_tiles,
            res.timings["total_s"],
        )

    def process_one(self) -> bool:
        """Claim and run one job; False when the queue is empty."""
        with Session(self.engine) as s:
            job = queue.claim_next_job(s, self.id)
            s.commit()
            if job is None:
                return False
            jid = job.id
            log.info("claimed job %s", jid)
            try:
                self._run(s, job)
                s.commit()
            except Exception as e:
                s.rollback()
                log.exception("job %s failed", jid)
                failed = s.get(Job, jid, populate_existing=True)
                if failed is not None:
                    queue.fail_job(s, failed, f"{type(e).__name__}: {e}")
                    s.commit()
        return True

    # -- loop --------------------------------------------------------------------------

    def _heartbeats(self) -> None:
        while not self.stop.is_set():
            try:
                self.beat()
            except Exception:
                log.warning("heartbeat failed", exc_info=True)
            self.stop.wait(self.settings.worker.heartbeat_interval_s)

    def run_forever(self) -> None:
        """Serve jobs until :attr:`stop` is set."""
        n = self.requeue_orphans()
        if n:
            log.warning("re-queued %d job(s) left running by a previous process", n)
        hb = threading.Thread(target=self._heartbeats, name="heartbeat", daemon=True)
        hb.start()
        log.info("worker %s ready (gpu %s)", self.id, self.gpu)
        try:
            while not self.stop.is_set():
                try:
                    busy = self.process_one()
                except Exception:
                    log.exception("queue poll failed")
                    busy = False
                if not busy:
                    self.stop.wait(self.settings.worker.poll_interval_s)
        finally:
            self.stop.set()
            hb.join(timeout=5)

    def close(self) -> None:
        """Stop the loop and release the database pool."""
        self.stop.set()
        self.engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = Worker(load_settings())

    def _stop(signum: int, _frame: FrameType | None) -> None:
        log.info("signal %d: finishing the current job, then exiting", signum)
        worker.stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        worker.run_forever()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
