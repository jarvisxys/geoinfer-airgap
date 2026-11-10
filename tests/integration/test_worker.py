from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from geoinfer.service.api.app import create_app
from geoinfer.service.api.settings import SchedulingSettings, Settings, WorkerSettings
from geoinfer.service.db import queue
from geoinfer.service.db.migrate import downgrade, upgrade
from geoinfer.service.db.schema import Building, Job
from geoinfer.service.registry import register_model
from geoinfer.service.worker.main import Worker
from geoinfer.tiling.config import load_tiling_config
from tests.conftest import MakeTiff
from tests.unit.test_worker_pipeline import ARMS, RECTS, STATS

URL = os.environ.get("GEOINFER_TEST_DB_URL")
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not URL, reason="needs GEOINFER_TEST_DB_URL (PostGIS)"),
]
# the 400 x 300 px raster at TRANSFORM spans about 4.3679-4.3708 E, 51.9544-51.9557 N
AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [4.3679, 51.9557],
            [4.3708, 51.9557],
            [4.3708, 51.9544],
            [4.3679, 51.9544],
            [4.3679, 51.9557],
        ]
    ],
}


@pytest.fixture
def env(tmp_path: Path, make_tiff: MakeTiff, tiny_onnx: Callable[..., Path]) -> Iterator[dict]:
    assert URL
    downgrade(URL)
    upgrade(URL)
    inbox, outputs, models = tmp_path / "inbox", tmp_path / "outputs", tmp_path / "models"
    for d in (inbox, outputs, models / "stats"):
        d.mkdir(parents=True)
    a = np.full((4, 300, 400), 200, np.uint16)
    for r0, r1, c0, c1 in RECTS:
        a[:, r0:r1, c0:c1] = 1500
    make_tiff(a, name="opt.tif").rename(inbox / "opt.tif")
    (models / "stats" / "stats.json").write_text(json.dumps(STATS))
    pre = tmp_path / "preprocess.yaml"
    pre.write_text(yaml.safe_dump({"sar": {"arms": ARMS}}))
    onnx_path = tiny_onnx(channels=4, gain=4.0)
    onnx_path.rename(models / "opt.onnx")
    settings = Settings(
        db_url=URL,
        inbox=inbox,
        outputs=outputs,
        models_dir=models,
        readiness_max_heartbeat_age_s=10,
        tasks=("segment_buildings",),
        default_models={"optical": "m_opt", "sar": "m_sar", "fused": "m_fus"},
        stats_file=Path("stats/stats.json"),
        preprocess_config=pre,
        worker=WorkerSettings(
            worker_id="w-test",
            poll_interval_s=0.01,
            heartbeat_interval_s=0.05,
            batch_size=4,
            providers=("CPUExecutionProvider",),
            max_aoi_mpx=1.0,
            scratch_dir=tmp_path / "scratch",
            metrics_port=0,
            scheduling=SchedulingSettings("cross_job", "priority", 0.1, 25.0, 2, 16),
        ),
    )
    engine = create_engine(URL)
    with Session(engine) as s:
        register_model(
            s,
            "m_opt",
            models / "opt.onnx",
            models,
            "optical",
            4,
            0.5,
            {"optical": "global", "sar": "db"},
        )
        s.commit()
    tcfg = replace(load_tiling_config(), tile=128, overlap=32)
    worker = Worker(settings, tcfg=tcfg)
    with TestClient(create_app(settings)) as client:
        r = client.post("/v1/scenes", json={"id": "s1", "path": "opt.tif", "sensor": "optical"})
        assert r.status_code == 201, r.text
        yield {
            "client": client,
            "worker": worker,
            "engine": engine,
            "models": models,
            "outputs": outputs,
        }
    worker.close()
    engine.dispose()


def _submit(c: TestClient) -> str:
    r = c.post(
        "/v1/jobs",
        json={"scene_id": "s1", "task": "segment_buildings", "aoi": AOI, "sensor_mode": "optical"},
    )
    assert r.status_code == 202, r.text
    return str(r.json()["id"])


def test_worker_runs_a_job_end_to_end(env: dict) -> None:
    c, w = env["client"], env["worker"]
    jid = _submit(c)
    w.beat()  # run_forever's heartbeat thread registers the worker before any job
    assert w.process_one() is True
    j = c.get(f"/v1/jobs/{jid}").json()
    assert j["status"] == "ready", j
    assert j["n_tiles"] > 0
    assert {"prepare_s", "infer_s", "stitch_s", "postprocess_s", "write_s", "total_s"} <= set(
        j["timings"]
    )
    gj = c.get(f"/v1/jobs/{jid}/result", params={"format": "geojson"}).json()
    assert len(gj["features"]) == 2
    assert sorted(round(f["properties"]["area_m2"]) for f in gj["features"]) == [600, 1000]
    cog = c.get(f"/v1/jobs/{jid}/result", params={"format": "cog"})
    assert cog.status_code == 200
    assert cog.content[:2] in (b"II", b"MM")
    with Session(env["engine"]) as s:
        rows = s.execute(
            select(func.ST_SRID(Building.geom), Building.area_m2, Building.score).where(
                Building.job_id == jid
            )
        ).all()
        assert len(rows) == 2
        assert all(srid == 4326 and score > 0.9 for srid, _, score in rows)
        inside = s.execute(
            text(
                "SELECT bool_and(ST_Within(b.geom, ST_Buffer(j.aoi, 1e-6))) FROM buildings b "
                "JOIN jobs j ON j.id = b.job_id WHERE j.id = :j"
            ),
            {"j": jid},
        ).scalar_one()
        assert inside
        done = s.execute(text("SELECT jobs_done FROM workers WHERE id = 'w-test'")).scalar_one()
        assert done == 1
    assert w.process_one() is False  # queue empty
    from prometheus_client import generate_latest

    m = generate_latest(w.metrics.registry).decode()
    assert 'geoinfer_worker_jobs_total{status="ready"} 1.0' in m
    assert "geoinfer_worker_tiles_total" in m
    assert 'geoinfer_worker_stage_seconds_total{stage="infer"}' in m


def test_worker_fails_job_on_model_hash_mismatch(env: dict) -> None:
    c, w = env["client"], env["worker"]
    (env["models"] / "opt.onnx").write_bytes(b"tampered")
    jid = _submit(c)
    assert w.process_one() is True
    j = c.get(f"/v1/jobs/{jid}").json()
    assert j["status"] == "failed"
    assert "sha256" in j["error"]
    assert w.metrics.jobs.labels("failed")._value.get() == 1
    with Session(env["engine"]) as s:
        assert s.scalar(select(func.count()).select_from(Building)) == 0


def test_heartbeat_makes_the_service_ready(env: dict) -> None:
    c, w = env["client"], env["worker"]
    assert c.get("/readyz").status_code == 503
    w.beat()
    assert c.get("/readyz").status_code == 200


def test_requeue_orphans_after_a_crash(env: dict) -> None:
    c = env["client"]
    jid = _submit(c)
    with Session(env["engine"]) as s:
        job = queue.claim_next_job(s, "w-test")
        assert job is not None
        s.commit()
        assert queue.requeue_orphans(s, "w-other") == 0
        assert queue.requeue_orphans(s, "w-test") == 1
        s.commit()
        j = s.get(Job, job.id, populate_existing=True)
        assert j is not None
        assert j.status == "queued"
        assert j.worker_id is None
    assert c.get(f"/v1/jobs/{jid}").json()["status"] == "queued"


@pytest.mark.parametrize("batching", ["cross_job", "per_job"])
def test_scheduled_runner_runs_concurrent_jobs(env: dict, batching: str) -> None:
    from geoinfer.service.worker.runner import ScheduledRunner

    c, w = env["client"], env["worker"]
    sc = replace(w.settings.worker.scheduling, batching=batching, max_active_jobs=3)
    # batch 6 does not divide a job's 20 tiles (4 x 5 grid): each tail leaves room for another job
    w.settings = replace(w.settings, worker=replace(w.settings.worker, scheduling=sc, batch_size=6))
    w.beat()
    ids = [_submit(c) for _ in range(3)]
    runner = ScheduledRunner(w)
    seen: list[set[str]] = []
    run_batch = runner._run_batch

    def spy(batch: list) -> None:
        seen.append({aj.job_id for aj, _ in batch})
        run_batch(batch)

    runner._run_batch = spy  # type: ignore[method-assign]
    runner.run(until_idle=True)
    for jid in ids:
        j = c.get(f"/v1/jobs/{jid}").json()
        assert j["status"] == "ready", j
        gj = c.get(f"/v1/jobs/{jid}/result", params={"format": "geojson"}).json()
        assert sorted(round(f["properties"]["area_m2"]) for f in gj["features"]) == [600, 1000]
    mixed = any(len(s) > 1 for s in seen)
    assert mixed is (batching == "cross_job")
