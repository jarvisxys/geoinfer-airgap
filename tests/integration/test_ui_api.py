from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from sqlalchemy.orm import Session

from geoinfer.service.db import queue
from geoinfer.service.db.schema import Job
from tests.integration.test_api import URL, _aoi, env  # noqa: F401  (pytest fixture)

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not URL, reason="needs GEOINFER_TEST_DB_URL (PostGIS)"),
]


def _register(c) -> None:  # type: ignore[no-untyped-def]
    r = c.post("/v1/scenes", json={"id": "s1", "path": "opt.tif", "sensor": "optical"})
    assert r.status_code == 201, r.text


def test_page_is_served_offline(env: dict) -> None:  # noqa: F811
    c = env["client"]
    r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui/"
    page = c.get("/ui/")
    assert page.status_code == 200
    assert "GeoInfer" in page.text
    js = c.get("/ui/app.js").text
    for text in (page.text, js, c.get("/ui/app.css").text):  # nothing loaded from the internet
        assert 'src="http' not in text
        assert 'href="http' not in text
        assert 'fetch("http' not in text
        assert "url(http" not in text


def test_status_summarises_the_service(env: dict) -> None:  # noqa: F811
    s = env["client"].get("/v1/status").json()
    assert s["db"] is True
    assert s["worker"] is False
    assert s["queue"] == {"queued": 0, "running": 0, "ready": 0, "failed": 0}
    assert s["models"] == ["m_opt"]
    assert s["max_aoi_px"] == 150_000_000


def test_upload_list_and_delete(env: dict) -> None:  # noqa: F811
    c, inbox = env["client"], env["inbox"]
    names = {f["name"] for f in c.get("/v1/inbox").json()}
    assert {"opt.tif", "sar.tif"} <= names
    data = (inbox / "opt.tif").read_bytes()
    r = c.put("/v1/inbox/My City (1).TIF", content=data)
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "My_City_1.tif"
    assert (inbox / "My_City_1.tif").read_bytes() == data
    assert c.put("/v1/inbox/My_City_1.tif", content=data).status_code == 409
    assert c.put("/v1/inbox/My_City_1.tif?overwrite=true", content=data).status_code == 201
    assert c.put("/v1/inbox/photo.jpg", content=b"x").status_code == 400
    big = c.put("/v1/inbox/big.tif", content=b"0" * (2 << 20))  # limit is 0.001 GB
    assert big.status_code == 413
    assert not (inbox / "big.tif").exists()
    assert not list(inbox.glob(".*.part"))
    assert c.delete("/v1/inbox/My_City_1.tif").status_code == 204
    _register(c)
    assert c.delete("/v1/inbox/opt.tif").status_code == 409  # used by scene s1
    assert c.delete("/v1/inbox/..%2Fsecret.tif").status_code in (400, 404)


def test_inspect_and_prepare(env: dict) -> None:  # noqa: F811
    c, inbox = env["client"], env["inbox"]
    r = c.get("/v1/inbox/opt.tif/inspect").json()
    assert r["ok"] is True
    assert r["bands"] == 4
    assert r["suggested_sensor"] == "optical"
    a = np.zeros((4, 60, 80), np.uint16)
    a[:, 20:40, 30:50] = 900
    with rasterio.open(
        inbox / "geo.tif",
        "w",
        driver="GTiff",
        height=60,
        width=80,
        count=4,
        dtype="uint16",
        crs="EPSG:4326",
        nodata=0,
        transform=from_origin(4.38, 51.92, 0.00001, 0.00001),
    ) as d:
        d.write(a)
    r = c.get("/v1/inbox/geo.tif/inspect").json()
    assert r["ok"] is False
    assert r["fixable"] is True
    assert c.post("/v1/inbox/geo.tif/prepare").status_code == 202
    for _ in range(100):
        st = c.get("/v1/inbox/geo.tif/prepare").json()
        if st["state"] != "running":
            break
        time.sleep(0.05)
    assert st == {"state": "done", "output": "geo_prepared.tif"}
    assert c.get("/v1/inbox/geo_prepared.tif/inspect").json()["ok"] is True


def test_map_tiles_and_convert(env: dict) -> None:  # noqa: F811
    c = env["client"]
    _register(c)
    m = c.get("/v1/scenes/s1/map").json()
    assert m["layers"] == ["optical"]
    assert m["modes"] == ["optical"]
    assert (m["width"], m["height"], m["max_zoom"], m["epsg"]) == (300, 200, 1, 32631)
    t = c.get("/v1/scenes/s1/tiles/optical/1/0/0.png")
    assert t.status_code == 200
    assert t.headers["content-type"] == "image/png"
    assert t.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert c.get("/v1/scenes/s1/tiles/sar/1/0/0.png").status_code == 404
    assert c.get("/v1/scenes/s1/tiles/optical/9/0/0.png").status_code == 404
    ll = c.post("/v1/scenes/s1/convert", json={"to": "lonlat", "points": [[0, 0], [300, 200]]})
    back = c.post("/v1/scenes/s1/convert", json={"to": "px", "points": ll.json()["points"]})
    np.testing.assert_allclose(back.json()["points"], [[0, 0], [300, 200]], atol=1e-3)


def test_job_list_and_overlay(env: dict) -> None:  # noqa: F811
    c = env["client"]
    _register(c)
    r = c.post(
        "/v1/jobs",
        json={
            "scene_id": "s1",
            "task": "segment_buildings",
            "aoi": _aoi(),
            "sensor_mode": "optical",
            "priority": 3,
        },
    )
    jid = r.json()["id"]
    jobs = c.get("/v1/jobs").json()
    assert jobs[0]["id"] == jid
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["sensor_mode"] == "optical"
    assert jobs[0]["priority"] == 3
    assert c.get(f"/v1/jobs/{jid}/overlay").status_code == 409
    out = env["outputs"] / jid
    out.mkdir()
    lon, lat = 4.3686, 51.9554
    poly = [
        [lon, lat],
        [lon + 0.0001, lat],
        [lon + 0.0001, lat - 0.00005],
        [lon, lat - 0.00005],
        [lon, lat],
    ]
    (out / "buildings.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1, "area_m2": 38.5, "score": 0.91},
                        "geometry": {"type": "Polygon", "coordinates": [poly]},
                    }
                ],
            }
        )
    )
    with Session(env["engine"]) as s:
        job = s.get(Job, uuid.UUID(jid))
        assert job is not None
        job.worker_id = "w1"
        queue.finish_job(s, job, 4, {"total_s": 1.0}, str(out))
        s.commit()
    ov = c.get(f"/v1/jobs/{jid}/overlay").json()
    assert ov["count"] == 1
    assert ov["total_area_m2"] == 38.5
    ring = ov["buildings"][0]["rings"][0]
    assert all(0 <= x <= 300 and 0 <= y <= 200 for x, y in ring)
    assert c.get(f"/v1/jobs/{uuid.uuid4()}/overlay").status_code == 404
