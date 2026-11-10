"""Concurrent job execution: reader threads, one GPU loop, a finisher (plan section 3).

- **Claim**: while fewer than ``max_active_jobs`` are in flight, the GPU loop claims the
  next queued job (``queue`` policy, with ageing) and opens it (:class:`SegmentJob`).
- **Read**: one thread per active job reads and normalises tiles into the job's
  ``pending`` queue, at most ``prefetch_tiles`` ahead, so disk I/O and normalisation
  overlap with inference.
- **Infer**: the GPU loop asks the scheduler (``batching``, ``max_wait_ms``) for the next
  batch, runs the model once, and stitches each probability into its own job. Inference
  time is shared out to the jobs by their tiles in the batch.
- **Finish**: a job whose tiles are all stitched goes to a single finisher thread (clean,
  vectorise, write, database) so the CPU-heavy tail never stalls the GPU.

A job that fails while opening, reading or finishing is marked failed; the others carry on.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sqlalchemy.orm import Session

from geoinfer.service.db import queue
from geoinfer.service.db.schema import Job
from geoinfer.service.worker.pipeline import SegmentJob
from geoinfer.service.worker.scheduler import (
    ActiveJob,
    Batching,
    Policy,
    QueueOrder,
    plan,
    take,
)

if TYPE_CHECKING:
    from geoinfer.service.worker.main import Worker, _Loaded

log = logging.getLogger("geoinfer.worker")


@dataclass
class _Payload:
    db_id: Any
    seg: SegmentJob
    loaded: _Loaded
    error: BaseException | None = None


class ScheduledRunner:
    """Run many jobs at once on one GPU (see module docstring)."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        w = worker.settings.worker
        sc = w.scheduling
        self.policy = Policy(
            cast(Batching, sc.batching),
            cast(QueueOrder, sc.queue),
            sc.ageing_per_s,
            w.batch_size,
            sc.max_wait_ms,
        )
        self.max_active, self.prefetch = sc.max_active_jobs, sc.prefetch_tiles
        self.cond = threading.Condition()
        self.active: list[ActiveJob] = []
        self.finisher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finish")
        self._finishing = 0
        self._queue_empty_at = 0.0

    # -- claim and read ------------------------------------------------------------------

    def _claim(self) -> bool:
        """Claim and open one job; False when the queue is empty."""
        wk = self.worker
        with Session(wk.engine) as s:
            job = wk.claim(s)
            s.commit()
            if job is None:
                return False
            jid, prio, created = job.id, int(job.priority), job.created_at.timestamp()
            try:
                seg, loaded, model = wk.open_job(s, job)
            except Exception as e:
                self._fail(jid, e)
                return True
        aj = ActiveJob(str(jid), prio, created, model.id, payload=_Payload(jid, seg, loaded))
        with self.cond:
            self.active.append(aj)
        threading.Thread(target=self._read, args=(aj,), name=f"read-{jid}", daemon=True).start()
        log.info("claimed job %s (priority %d)", jid, prio)
        return True

    def _read(self, aj: ActiveJob) -> None:
        p: _Payload = aj.payload
        try:
            for item in p.seg.tiles():
                with self.cond:
                    while len(aj.pending) >= self.prefetch and not self.worker.stop.is_set():
                        self.cond.wait(0.5)
                    aj.pending.append(item)
                    self.cond.notify_all()
        except Exception as e:
            p.error = e
        finally:
            with self.cond:
                aj.reading_done = True
                self.cond.notify_all()

    # -- finish ----------------------------------------------------------------------------

    def _fail(self, jid: Any, e: BaseException) -> None:
        log.error("job %s failed: %s", jid, e)
        self.worker.metrics.jobs.labels("failed").inc()
        with Session(self.worker.engine) as s:
            job = s.get(Job, jid)
            if job is not None:
                queue.fail_job(s, job, f"{type(e).__name__}: {e}")
                s.commit()

    def _finish(self, aj: ActiveJob) -> None:
        p: _Payload = aj.payload
        try:
            if p.error is not None:
                raise p.error
            res = p.seg.finish()
            with Session(self.worker.engine) as s:
                job = s.get(Job, p.db_id)
                assert job is not None
                self.worker.store(s, job, res, p.seg.out_dir)
                s.commit()
            self.worker.metrics.record(res)
        except Exception as e:
            self._fail(p.db_id, e)
        finally:
            p.seg.close()
            with self.cond:
                self._finishing -= 1
                self.cond.notify_all()

    def _retire(self) -> None:
        """Hand finished (or failed and drained) jobs to the finisher (caller holds cond)."""
        for aj in [j for j in self.active if j.done]:
            self.active.remove(aj)
            self._finishing += 1
            self.finisher.submit(self._finish, aj)

    # -- GPU loop ------------------------------------------------------------------------------

    def _run_batch(self, batch: list[tuple[ActiveJob, Any]]) -> None:
        loaded: _Loaded = batch[0][0].payload.loaded
        t0 = time.perf_counter()
        probs = loaded.predictor(np.stack([tile[1] for _, tile in batch]))
        dt = time.perf_counter() - t0
        for (aj, (spec, _, valid)), prob in zip(batch, probs, strict=True):
            p: _Payload = aj.payload
            p.seg.timings["infer_s"] += dt / len(batch)
            p.seg.add(spec, prob, valid)
        with self.cond:
            for aj, _ in batch:
                aj.inflight -= 1
            self._retire()
            self.cond.notify_all()

    def step(self) -> bool:
        """One scheduling step; False when there was nothing to do."""
        now = time.time()
        poll = self.worker.settings.worker.poll_interval_s
        with self.cond:
            n_active = len(self.active)
        if n_active < self.max_active and now - self._queue_empty_at >= poll:
            if self._claim():
                return True
            self._queue_empty_at = now
        first_wait: float | None = None
        while True:
            with self.cond:
                for aj in self.active:  # a reader failed: drop its tiles, finish as failed
                    if aj.payload.error is not None and aj.pending:
                        aj.pending.clear()
                self._retire()
                sel = plan(self.active, self.policy, time.time())
                if sel.size == 0:
                    if not self.active:
                        return False
                    self.cond.wait(0.05)
                    return True
                waited = 0.0 if first_wait is None else (time.time() - first_wait) * 1e3
                if (
                    sel.size < self.policy.max_batch
                    and sel.can_grow
                    and waited < self.policy.max_wait_ms
                ):
                    first_wait = first_wait or time.time()
                    self.cond.wait((self.policy.max_wait_ms - waited) / 1e3)
                    continue
                batch = take(sel)
                self.cond.notify_all()
            self._run_batch(batch)
            return True

    def idle(self) -> bool:
        """No active job and nothing being finished."""
        with self.cond:
            return not self.active and self._finishing == 0

    def run(self, until_idle: bool = False) -> None:
        """Loop until the worker stops (or, with ``until_idle``, the queue runs dry)."""
        stop = self.worker.stop
        while not stop.is_set():
            busy = self.step()
            if not busy:
                if until_idle and self.idle():
                    break
                stop.wait(min(0.05, self.worker.settings.worker.poll_interval_s))
        self.close()

    def close(self) -> None:
        """Wait for the jobs being finished (in-flight jobs finish before shutdown)."""
        while True:
            with self.cond:
                if not self.active:
                    break
            if not self.step() and self.worker.stop.is_set():
                # stopping with work left: let active jobs drain before leaving
                time.sleep(0.01)
        self.finisher.shutdown(wait=True)
