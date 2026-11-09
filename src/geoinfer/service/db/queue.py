"""PostgreSQL as the job queue (plan section 3): atomic claims with SKIP LOCKED.

A worker claims the highest-priority, oldest queued job in one statement; rows locked by
another worker's open transaction are skipped, so two workers never get the same job.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from geoinfer.service.db.schema import Job

_CLAIM = text(
    """
    UPDATE jobs SET status = 'running', started_at = now(), worker_id = :worker
    WHERE id = (
        SELECT id FROM jobs WHERE status = 'queued'
        ORDER BY priority DESC, created_at
        LIMIT 1 FOR UPDATE SKIP LOCKED)
    RETURNING id
    """
)


def claim_next_job(session: Session, worker_id: str) -> Job | None:
    """Mark the next queued job ``running`` for ``worker_id`` (caller commits)."""
    jid = session.execute(_CLAIM, {"worker": worker_id}).scalar_one_or_none()
    if jid is None:
        return None
    job = session.get(Job, jid, populate_existing=True)
    return job


def finish_job(
    session: Session, job: Job, n_tiles: int, timings: dict[str, Any], result_dir: str
) -> None:
    """Mark ``job`` ready with its tile count, timings and output folder."""
    job.status, job.n_tiles, job.timings, job.result_dir = "ready", n_tiles, timings, result_dir
    job.finished_at = session.execute(text("select now()")).scalar_one()
    session.execute(
        text("UPDATE workers SET jobs_done = jobs_done + 1 WHERE id = :w"), {"w": job.worker_id}
    )


def fail_job(session: Session, job: Job, error: str) -> None:
    """Mark ``job`` failed with an error message (truncated to 2000 characters)."""
    job.status, job.error = "failed", error[:2000]
    job.finished_at = session.execute(text("select now()")).scalar_one()


def heartbeat(session: Session, worker_id: str, gpu: str | None = None) -> None:
    """Insert or refresh this worker's ``last_seen`` (caller commits)."""
    session.execute(
        text(
            """
            INSERT INTO workers (id, gpu, last_seen) VALUES (:w, :g, now())
            ON CONFLICT (id) DO UPDATE SET last_seen = now(), gpu = EXCLUDED.gpu
            """
        ),
        {"w": worker_id, "g": gpu},
    )


def worker_alive(session: Session, max_age_s: float) -> bool:
    """True if any worker sent a heartbeat in the last ``max_age_s`` seconds."""
    n = session.execute(
        text("SELECT count(*) FROM workers WHERE last_seen > now() - make_interval(secs => :s)"),
        {"s": max_age_s},
    ).scalar_one()
    return bool(n)


def requeue_orphans(session: Session, worker_id: str) -> int:
    """Put ``worker_id``'s ``running`` jobs back in the queue; returns how many.

    A worker calls this on start-up: jobs it still holds were cut off when its previous
    process died. Results are written to the database only when a job finishes, so a
    cut-off job left no rows behind (its output folder is overwritten on the rerun).
    """
    r = session.execute(
        text(
            """
            UPDATE jobs SET status = 'queued', worker_id = NULL, started_at = NULL
            WHERE status = 'running' AND worker_id = :w
            """
        ),
        {"w": worker_id},
    )
    return int(getattr(r, "rowcount", 0) or 0)
