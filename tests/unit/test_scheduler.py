from __future__ import annotations

import pytest

from geoinfer.service.worker.scheduler import ActiveJob, Policy, plan, rank, take


def _job(
    jid: str, prio: int, created: float, n: int, model: str = "m", reading: bool = False
) -> ActiveJob:
    j = ActiveJob(jid, prio, created, model)
    j.pending.extend((jid, k) for k in range(n))
    j.reading_done = not reading
    return j


def _p(**kw: object) -> Policy:
    base = {
        "batching": "cross_job",
        "queue": "priority",
        "ageing_per_s": 0.0,
        "max_batch": 4,
        "max_wait_ms": 25.0,
    }
    return Policy(**(base | kw))  # type: ignore[arg-type]


def test_rank_fifo_and_priority_with_ageing() -> None:
    old_low, new_high = _job("a", 0, 0.0, 1), _job("b", 5, 10.0, 1)
    assert [j.job_id for j in rank([new_high, old_low], _p(queue="fifo"), 10.0)] == ["a", "b"]
    assert [j.job_id for j in rank([old_low, new_high], _p(), 10.0)] == ["b", "a"]
    # 0 + 1.0/s * 20 s > 5 + 1.0/s * 10 s: the old job has aged past the new one
    assert [j.job_id for j in rank([old_low, new_high], _p(ageing_per_s=1.0), 20.0)] == ["a", "b"]


def test_per_job_batches_never_mix_jobs() -> None:
    a, b = _job("a", 1, 0.0, 2), _job("b", 0, 0.0, 5)
    sel = plan([a, b], _p(batching="per_job"), 0.0)
    assert sel.size == 2
    assert not sel.can_grow
    batch = take(sel)
    assert {job.job_id for job, _ in batch} == {"a"}
    assert a.inflight == 2
    assert not a.pending


def test_cross_job_fills_from_jobs_in_rank_order_same_model_only() -> None:
    a, b = _job("a", 2, 0.0, 2), _job("b", 1, 0.0, 5)
    c = _job("c", 0, 0.0, 5, model="other")
    batch = take(plan([c, b, a], _p(), 0.0))
    assert [job.job_id for job, _ in batch] == ["a", "a", "b", "b"]
    assert len(c.pending) == 5


def test_partial_batch_can_grow_while_a_job_is_still_reading() -> None:
    a = _job("a", 0, 0.0, 1, reading=True)
    sel = plan([a], _p(), 0.0)
    assert sel.size == 1
    assert sel.can_grow
    a.reading_done = True
    assert not plan([a], _p(), 0.0).can_grow


def test_job_is_done_after_its_last_tile_is_returned() -> None:
    a = _job("a", 0, 0.0, 2)
    batch = take(plan([a], _p(), 0.0))
    assert not a.done
    a.inflight -= len(batch)
    assert a.done


@pytest.mark.parametrize(("ageing", "starves"), [(0.0, True), (1.0, False)])
def test_low_priority_job_finishes_under_sustained_high_load(ageing: float, starves: bool) -> None:
    """One batch per second; a new high-priority job (8 tiles) arrives every 2 s forever."""
    low = _job("low", 0, 0.0, 12)
    jobs = [low]
    policy = _p(batching="per_job", ageing_per_s=ageing)
    finished_at = None
    for t in range(200):
        if t % 2 == 0:
            jobs.append(_job(f"h{t}", 10, float(t), 8))
        batch = take(plan(jobs, policy, float(t)))
        for job, _ in batch:
            job.inflight -= 1
        jobs = [j for j in jobs if not j.done or j is low]
        if low.done:
            finished_at = t
            break
    if starves:
        assert finished_at is None
    else:
        assert finished_at is not None
        assert finished_at < 40
