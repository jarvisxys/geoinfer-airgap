"""Tile scheduling across active jobs (experiments E2 and E3; plan section 3).

The worker keeps several claimed jobs active; each job's reader thread fills its
``pending`` queue with normalised tiles. The GPU loop asks :func:`plan` which tiles to
run next and :func:`take` pops them:

- ``batching``: ``per_job`` batches tiles of the top-ranked job only; ``cross_job`` fills
  the batch from every active job with the same model, in rank order (dynamic batching);
- ``queue``: ``fifo`` ranks by arrival; ``priority`` by ``priority + ageing_per_s * age``
  (seconds since the job was submitted), so a low-priority job overtakes newer
  high-priority work after waiting long enough and cannot starve;
- ``max_wait_ms``: a partial batch is held back this long while it ``can_grow`` (an
  eligible job is still reading tiles), trading latency for fuller batches.

Pure logic on in-memory state: no threads, clocks or GPU here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

Batching = Literal["per_job", "cross_job"]
QueueOrder = Literal["fifo", "priority"]
BATCHING = ("per_job", "cross_job")
QUEUES = ("fifo", "priority")


@dataclass(frozen=True)
class Policy:
    """Scheduling knobs (``worker.scheduling`` in configs/service.yaml)."""

    batching: Batching
    queue: QueueOrder
    ageing_per_s: float
    max_batch: int
    max_wait_ms: float

    def __post_init__(self) -> None:
        if self.batching not in BATCHING:
            raise ValueError(f"batching must be one of {BATCHING}")
        if self.queue not in QUEUES:
            raise ValueError(f"queue must be one of {QUEUES}")
        if self.max_batch < 1 or self.max_wait_ms < 0 or self.ageing_per_s < 0:
            raise ValueError("max_batch >= 1, max_wait_ms >= 0 and ageing_per_s >= 0")


@dataclass(eq=False)
class ActiveJob:
    """Scheduling state of one claimed job.

    Attributes
    ----------
    job_id, priority
        From the jobs table.
    created
        Submission time (epoch seconds); ageing counts from here.
    model_key
        Tiles of different models never share a batch.
    pending
        Tiles read and normalised, not yet sent to the GPU.
    reading_done
        The reader has queued every tile.
    inflight
        Tiles taken for a batch whose probabilities are not stitched yet.
    payload
        Whatever the worker attaches (the SegmentJob, the database row).
    """

    job_id: str
    priority: int
    created: float
    model_key: str
    pending: deque[Any] = field(default_factory=deque)
    reading_done: bool = False
    inflight: int = 0
    payload: Any = None

    @property
    def done(self) -> bool:
        """Every tile was read, run and stitched."""
        return self.reading_done and not self.pending and self.inflight == 0


def rank(jobs: list[ActiveJob], policy: Policy, now: float) -> list[ActiveJob]:
    """Jobs best first under the queue policy (ties: older first)."""
    if policy.queue == "fifo":
        return sorted(jobs, key=lambda j: j.created)
    return sorted(
        jobs, key=lambda j: (-(j.priority + policy.ageing_per_s * (now - j.created)), j.created)
    )


@dataclass
class Selection:
    """What :func:`take` would run: ``(job, n_tiles)`` parts, in rank order."""

    parts: list[tuple[ActiveJob, int]]
    can_grow: bool

    @property
    def size(self) -> int:
        return sum(n for _, n in self.parts)


def plan(jobs: list[ActiveJob], policy: Policy, now: float) -> Selection:
    """The next batch: the top-ranked job with tiles, plus (cross-job) others of its model."""
    ranked = rank(jobs, policy, now)
    first = next((j for j in ranked if j.pending), None)
    if first is None:
        return Selection([], False)
    if policy.batching == "per_job":
        eligible = [first]
    else:
        eligible = [j for j in ranked if j.model_key == first.model_key]
    parts, room = [], policy.max_batch
    for j in eligible:
        n = min(room, len(j.pending))
        if n:
            parts.append((j, n))
            room -= n
        if room == 0:
            break
    can_grow = room > 0 and any(not j.reading_done for j in eligible)
    return Selection(parts, can_grow)


def take(sel: Selection) -> list[tuple[ActiveJob, Any]]:
    """Pop the selected tiles; each job's ``inflight`` grows by its share."""
    out = []
    for job, n in sel.parts:
        for _ in range(n):
            out.append((job, job.pending.popleft()))
        job.inflight += n
    return out
