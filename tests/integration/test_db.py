from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("geoalchemy2")
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from geoinfer.service.db import queue  # noqa: E402
from geoinfer.service.db.migrate import downgrade, upgrade  # noqa: E402
from geoinfer.service.db.schema import Building, Job, Model, Scene  # noqa: E402

URL = os.environ.get("GEOINFER_TEST_DB_URL")
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not URL, reason="needs GEOINFER_TEST_DB_URL (PostGIS)"),
]
SQUARE = "SRID=4326;POLYGON((4.40 51.90, 4.41 51.90, 4.41 51.91, 4.40 51.91, 4.40 51.90))"


@pytest.fixture
def engine() -> Iterator[Engine]:
    assert URL
    eng = create_engine(URL)
    downgrade(URL)
    upgrade(URL)
    yield eng
    eng.dispose()


def _seed(s: Session) -> None:
    s.add(
        Scene(
            id="sc1",
            path="/data/inbox/a.tif",
            sensor="optical",
            footprint=SQUARE,
            gsd_m=0.5,
            width=100,
            height=100,
            sha256="a" * 64,
            crs="EPSG:32631",
        )
    )
    s.add(
        Model(
            id="m1",
            path="/models/a.onnx",
            sha256="b" * 64,
            sensor_mode="optical",
            in_channels=4,
            threshold=0.5,
        )
    )
    s.flush()


def _job(s: Session, priority: int = 0) -> uuid.UUID:
    j = Job(scene_id="sc1", model_id="m1", aoi=SQUARE, task="segment_buildings", priority=priority)
    s.add(j)
    s.flush()
    return j.id


def test_migration_creates_the_schema(engine: Engine) -> None:
    with engine.connect() as c:
        tables = set(
            c.execute(text("select tablename from pg_tables where schemaname = 'public'")).scalars()
        )
        assert {"scenes", "models", "jobs", "buildings", "workers"} <= tables
        assert c.execute(text("select postgis_version()")).scalar()
    downgrade(URL)
    with engine.connect() as c:
        tables = set(
            c.execute(text("select tablename from pg_tables where schemaname = 'public'")).scalars()
        )
        assert not {"scenes", "jobs"} & tables


def test_claim_order_is_priority_then_age(engine: Engine) -> None:
    with Session(engine) as s:
        _seed(s)
        low, high, high_later = _job(s, 0), _job(s, 5), _job(s, 5)
        s.commit()
    order = []
    for _ in range(4):
        with Session(engine) as s:
            j = queue.claim_next_job(s, "w1")
            s.commit()
            order.append(None if j is None else j.id)
    assert order == [high, high_later, low, None]
    with Session(engine) as s:
        j = s.get(Job, high)
        assert j is not None
        assert (j.status, j.worker_id) == ("running", "w1")
        assert j.started_at is not None


def test_concurrent_claims_never_share_a_job(engine: Engine) -> None:
    with Session(engine) as s:
        _seed(s)
        a, b = _job(s), _job(s)
        s.commit()
    s1, s2 = Session(engine), Session(engine)
    try:
        j1 = queue.claim_next_job(s1, "w1")  # row locked, not yet committed
        j2 = queue.claim_next_job(s2, "w2")  # must skip the locked row
        assert j1 is not None
        assert j2 is not None
        assert {j1.id, j2.id} == {a, b}
        s1.commit()
        s2.commit()
    finally:
        s1.close()
        s2.close()


def test_finish_and_fail_record_status_and_timings(engine: Engine) -> None:
    with Session(engine) as s:
        _seed(s)
        jid = _job(s)
        s.commit()
        j = queue.claim_next_job(s, "w1")
        assert j is not None
        queue.finish_job(s, j, n_tiles=12, timings={"infer_s": 1.5}, result_dir="/data/outputs/x")
        s.commit()
        j2 = s.get(Job, jid)
        assert j2 is not None
        assert (j2.status, j2.n_tiles, j2.timings["infer_s"]) == ("ready", 12, 1.5)
        assert j2.finished_at is not None
        jid2 = _job(s)
        s.commit()
        j3 = queue.claim_next_job(s, "w1")
        assert j3 is not None
        queue.fail_job(s, j3, "boom")
        s.commit()
        assert s.get(Job, jid2).status == "failed"  # type: ignore[union-attr]


def test_status_constraint_and_cascade(engine: Engine) -> None:
    with Session(engine) as s:
        _seed(s)
        jid = _job(s)
        s.add(Building(job_id=jid, geom=SQUARE, area_m2=120.0, score=0.9))
        s.commit()
        assert s.execute(text("select ST_SRID(geom) from buildings")).scalar() == 4326
        s.delete(s.get(Job, jid))
        s.commit()
        assert s.execute(text("select count(*) from buildings")).scalar() == 0
    with Session(engine) as s:  # scene and model exist from above
        s.add(Job(scene_id="sc1", model_id="m1", aoi=SQUARE, task="t", status="lost"))
        with pytest.raises(IntegrityError, match="jobs_status"):
            s.flush()


def test_heartbeat_and_readiness(engine: Engine) -> None:
    with Session(engine) as s:
        assert not queue.worker_alive(s, max_age_s=10)
        queue.heartbeat(s, "w1", gpu="RTX")
        queue.heartbeat(s, "w1", gpu="RTX")  # upsert, not a second row
        s.commit()
        assert queue.worker_alive(s, max_age_s=10)
        assert s.execute(text("select count(*) from workers")).scalar() == 1
        s.execute(text("update workers set last_seen = now() - interval '1 minute'"))
        s.commit()
        assert not queue.worker_alive(s, max_age_s=10)


def _claim_all(engine: Engine, queue_policy: str, ageing: float) -> list[uuid.UUID | None]:
    out = []
    for _ in range(3):
        with Session(engine) as s:
            j = queue.claim_next_job(s, "w1", queue_policy, ageing)
            s.commit()
            out.append(None if j is None else j.id)
    return out


def test_claim_order_fifo_ignores_priority(engine: Engine) -> None:
    with Session(engine) as s:
        _seed(s)
        low, high = _job(s, 0), _job(s, 5)
        s.execute(
            text("UPDATE jobs SET created_at = now() - interval '10 s' WHERE id = :i"), {"i": low}
        )
        s.commit()
    assert _claim_all(engine, "fifo", 0.0) == [low, high, None]


def test_claim_order_ageing_lets_an_old_low_job_overtake(engine: Engine) -> None:
    with Session(engine) as s:
        _seed(s)
        low, high = _job(s, 0), _job(s, 5)
        # waited 60 s: 0 + 0.1/s * 60 = 6 > 5 + 0.1 * ~0
        s.execute(
            text("UPDATE jobs SET created_at = now() - interval '60 s' WHERE id = :i"), {"i": low}
        )
        s.commit()
    assert _claim_all(engine, "priority", 0.1) == [low, high, None]


def test_claim_order_rejects_unknown_policy(engine: Engine) -> None:
    with Session(engine) as s, pytest.raises(ValueError, match="queue"):
        queue.claim_next_job(s, "w1", "lifo")
