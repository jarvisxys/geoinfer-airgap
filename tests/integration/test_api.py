from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from geoinfer.service.api.app import create_app
from geoinfer.service.api.settings import (
    SchedulingSettings,
    Settings,
    UiSettings,
    WorkerSettings,
)
from geoinfer.service.db import queue
from geoinfer.service.db.migrate import downgrade, upgrade
from geoinfer.service.db.schema import Job, Model
from tests.conftest import MakeTiff

URL = os.environ.get("GEOINFER_TEST_DB_URL")
SCHED = SchedulingSettings("cross_job", "priority", 0.1, 25.0, 2, 16)
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not URL, reason="needs GEOINFER_TEST_DB_URL (PostGIS)"),
]
# conftest TRANSFORM: origin (594000, 5757000), 0.5 m, EPSG:32631; the 300 x 200 px test
# raster spans 4.3679-4.3701 E, 51.9548-51.9557 N


def _aoi(dlon: float = 0.0) -> dict:
    lon, lat = 4.3685 + dlon, 51.9555
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon, lat],
                [lon + 0.0005, lat],
                [lon + 0.0005, lat - 0.0003],
                [lon, lat - 0.0003],
                [lon, lat],
            ]
        ],
    }


@pytest.fixture
def env(tmp_path: Path, make_tiff: MakeTiff) -> Iterator[dict]:
    assert URL
    downgrade(URL)
    upgrade(URL)
    inbox, outputs = tmp_path / "inbox", tmp_path / "outputs"
    inbox.mkdir()
    outputs.mkdir()
    rng = np.random.default_rng(0)
    opt = make_tiff(rng.integers(1, 900, (4, 200, 300)).astype(np.uint16), name="opt.tif")
    sar = make_tiff(rng.random((4, 200, 300)).astype(np.float32) + 20, name="sar.tif")
    for f in (opt, sar):
        f.rename(inbox / f.name)
    settings = Settings(
        db_url=URL,
        inbox=inbox,
        outputs=outputs,
        models_dir=tmp_path,
        readiness_max_heartbeat_age_s=10,
        tasks=("segment_buildings",),
        default_models={"optical": "m_opt", "sar": "m_sar", "fused": "m_fus"},
        stats_file=Path("stats/stats.json"),
        preprocess_config=Path("configs/preprocess.yaml"),
        worker=WorkerSettings(
            "w", 1.0, 3.0, 8, ("CPUExecutionProvider",), 150.0, tmp_path, 0, SCHED
        ),
        ui=UiSettings(max_upload_gb=0.001, prepare_gsd_m=0.5, tile_cache=64),
    )
    engine = create_engine(URL)
    with Session(engine) as s:
        s.add(
            Model(
                id="m_opt",
                path="opt.onnx",
                sha256="a" * 64,
                sensor_mode="optical",
                in_channels=4,
                threshold=0.55,
            )
        )
        s.commit()
    with TestClient(create_app(settings)) as client:
        yield {"client": client, "inbox": inbox, "outputs": outputs, "engine": engine}
    engine.dispose()


def _register(c: TestClient, **kw: object) -> object:
    body = {"id": "s1", "path": "opt.tif", "sensor": "optical"} | kw
    return c.post("/v1/scenes", json=body)


def test_health_and_readiness(env: dict) -> None:
    c = env["client"]
    assert c.get("/healthz").json() == {"status": "ok"}
    r = c.get("/readyz")
    assert r.status_code == 503
    assert r.json()["worker"] is False
    with Session(env["engine"]) as s:
        queue.heartbeat(s, "w1")
        s.commit()
    r = c.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"db": True, "worker": True}


def test_register_scene_computes_footprint_hash_and_gsd(env: dict) -> None:
    c = env["client"]
    r = _register(c)
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["gsd_m"] == pytest.approx(0.5)
    assert (s["width"], s["height"], s["crs"]) == (300, 200, "EPSG:32631")
    want = hashlib.sha256((env["inbox"] / "opt.tif").read_bytes()).hexdigest()
    assert s["sha256"] == want
    lon0, lat0, lon1, lat1 = s["bounds"]
    assert 4.3 < lon0 < lon1 < 4.4
    assert 51.9 < lat0 < lat1 < 52.0
    assert [x["id"] for x in c.get("/v1/scenes").json()] == ["s1"]
    assert _register(c).status_code == 409


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"path": "../outside.tif"}, 400),
        ({"path": "/etc/passwd"}, 400),
        ({"path": "missing.tif"}, 404),
        ({"sensor": "sar", "path": "sar.tif"}, 422),  # SAR needs its look direction (D5)
        ({"sensor": "fused"}, 422),  # fused needs sar_path
        ({"sensor": "radar"}, 422),
    ],
)
def test_register_scene_rejects_bad_input(env: dict, kw: dict, code: int) -> None:
    assert _register(env["client"], **kw).status_code == code


def test_register_fused_scene(env: dict) -> None:
    r = _register(env["client"], id="f1", sensor="fused", sar_path="sar.tif", orientation=1)
    assert r.status_code == 201, r.text
    assert (r.json()["orientation"], r.json()["sar_path"]) == (1, "sar.tif")


def test_submit_job_checks_the_aoi_against_the_footprint(env: dict) -> None:
    c = env["client"]
    _register(c)
    body = {
        "scene_id": "s1",
        "aoi": _aoi(),
        "task": "segment_buildings",
        "sensor_mode": "optical",
        "priority": 3,
    }
    r = c.post("/v1/jobs", json=body)
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    st = c.get(f"/v1/jobs/{jid}").json()
    assert (st["status"], st["priority"], st["model_id"]) == ("queued", 3, "m_opt")
    far = c.post("/v1/jobs", json=body | {"aoi": _aoi(dlon=1.0)})
    assert far.status_code == 422
    assert "does not intersect" in far.text


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"scene_id": "nope"}, 404),
        ({"sensor_mode": "sar"}, 422),  # an optical scene cannot serve SAR
        ({"task": "count_cars"}, 422),
        ({"aoi": {"type": "Point", "coordinates": [4.36, 51.95]}}, 422),
    ],
)
def test_submit_job_rejects_bad_requests(env: dict, patch: dict, code: int) -> None:
    c = env["client"]
    _register(c)
    body = {
        "scene_id": "s1",
        "aoi": _aoi(),
        "task": "segment_buildings",
        "sensor_mode": "optical",
    } | patch
    assert c.post("/v1/jobs", json=body).status_code == code


def test_job_without_a_registered_model_is_refused(env: dict) -> None:
    c = env["client"]
    _register(c, id="f1", sensor="fused", sar_path="sar.tif", orientation=0)
    body = {"scene_id": "f1", "aoi": _aoi(), "task": "segment_buildings", "sensor_mode": "fused"}
    r = c.post("/v1/jobs", json=body)
    assert r.status_code == 409
    assert "m_fus" in r.text


def test_results_only_when_ready(env: dict) -> None:
    c = env["client"]
    _register(c)
    body = {"scene_id": "s1", "aoi": _aoi(), "task": "segment_buildings", "sensor_mode": "optical"}
    jid = c.post("/v1/jobs", json=body).json()["id"]
    assert c.get(f"/v1/jobs/{jid}/result").status_code == 409
    out = env["outputs"] / jid
    out.mkdir()
    (out / "buildings.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )
    (out / "mask.tif").write_bytes(b"II*\x00fake")
    with Session(env["engine"]) as s:
        j = s.get(Job, uuid.UUID(jid))
        assert j is not None
        j.status, j.result_dir = "ready", str(out)
        s.commit()
    g = c.get(f"/v1/jobs/{jid}/result?format=geojson")
    assert g.status_code == 200
    assert g.json()["type"] == "FeatureCollection"
    m = c.get(f"/v1/jobs/{jid}/result?format=cog")
    assert m.status_code == 200
    assert m.headers["content-type"] == "image/tiff"
    assert c.get(f"/v1/jobs/{jid}/result?format=png").status_code == 422
    assert c.get(f"/v1/jobs/{uuid.uuid4()}").status_code == 404


def test_models_and_metrics(env: dict) -> None:
    c = env["client"]
    m = c.get("/v1/models").json()
    assert [(x["id"], x["sha256"], x["threshold"]) for x in m] == [("m_opt", "a" * 64, 0.55)]
    _register(c)
    c.post(
        "/v1/jobs",
        json={
            "scene_id": "s1",
            "aoi": _aoi(),
            "task": "segment_buildings",
            "sensor_mode": "optical",
        },
    )
    metrics = c.get("/metrics").text
    assert "geoinfer_jobs_submitted_total" in metrics
    assert "geoinfer_queue_depth" in metrics
    with env["engine"].connect() as conn:
        assert conn.execute(text("select count(*) from jobs")).scalar() == 1


def test_register_model_records_hash_and_refuses_outside_files(env: dict, tmp_path: Path) -> None:
    from geoinfer.service.registry import register_model

    reg = tmp_path / "registry"
    reg.mkdir()
    f = reg / "fused.onnx"
    f.write_bytes(b"onnx-bytes")
    with Session(env["engine"]) as s:
        m = register_model(s, "m_fus", f, reg, "fused", 8, 0.4, {"sar": "db_lee7"})
        s.commit()
        assert (m.path, m.sha256) == ("fused.onnx", hashlib.sha256(b"onnx-bytes").hexdigest())
        with pytest.raises(ValueError, match="registry"):
            register_model(s, "x", tmp_path / "inbox" / "opt.tif", reg, "optical", 4, 0.5)
    ids = [x["id"] for x in env["client"].get("/v1/models").json()]
    assert ids == ["m_fus", "m_opt"]


def test_sync_manifest_registers_and_checks_hashes(env: dict, tmp_path: Path) -> None:
    import yaml

    from geoinfer.service.registry import sync_manifest

    reg = tmp_path / "reg"
    (reg / "m_sar").mkdir(parents=True)
    (reg / "m_sar" / "model.onnx").write_bytes(b"sar")
    entry = {
        "id": "m_sar",
        "path": "m_sar/model.onnx",
        "sensor_mode": "sar",
        "in_channels": 4,
        "threshold": 0.2,
        "norm": {"optical": "global", "sar": "db_lee7"},
        "sha256": hashlib.sha256(b"sar").hexdigest(),
    }
    (reg / "registry.yaml").write_text(yaml.safe_dump({"models": [entry]}))
    with Session(env["engine"]) as s:
        (m,) = sync_manifest(s, reg)
        s.commit()
        assert (m.threshold, m.norm) == (0.2, entry["norm"])
    (reg / "m_sar" / "model.onnx").write_bytes(b"altered")
    with Session(env["engine"]) as s, pytest.raises(ValueError, match="sha256"):
        sync_manifest(s, reg)
