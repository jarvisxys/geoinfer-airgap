"""Replay a request file against the API at a target arrival rate (open loop).

Each request is submitted at ``t / rate`` seconds after the start, on its own thread, so
a slow response never delays later arrivals (a closed-loop tool such as Locust would wait
for each user, which hides queueing delay at high load). Latency is measured afterwards
from the database timestamps (``created_at`` to ``finished_at``): the API is
asynchronous, so the HTTP response time alone means nothing.

Standard library only.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def call(
    api: str, method: str, path: str, body: Any = None, timeout: float = 30
) -> tuple[int, Any]:
    """One JSON request; returns (status, decoded body)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        api + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def schedule(reqs: Sequence[dict[str, Any]], rate: float | None) -> list[float]:
    """Submission offsets in seconds: ``t / rate``; ``rate=None`` submits all at once."""
    if rate is None:
        return [0.0] * len(reqs)
    if rate <= 0:
        raise ValueError("rate must be positive")
    return [float(r["t"]) / rate for r in reqs]


def replay(api: str, reqs: Sequence[dict[str, Any]], rate: float | None) -> list[dict[str, Any]]:
    """Submit every request at its offset; returns one record per request."""
    offsets = schedule(reqs, rate)
    out: list[dict[str, Any]] = [{} for _ in reqs]
    t0 = time.monotonic()
    lock = threading.Lock()

    def submit(k: int) -> None:
        r = reqs[k]
        body = {key: r[key] for key in ("scene_id", "sensor_mode", "task", "aoi", "priority")}
        sent = time.monotonic() - t0
        st, resp = call(api, "POST", "/v1/jobs", body)
        rec = {
            "i": r["i"],
            "side_km": r["side_km"],
            "priority": r["priority"],
            "sensor_mode": r["sensor_mode"],
            "sent_s": round(sent, 3),
            "http": st,
            "job_id": resp.get("id") if st == 202 and isinstance(resp, dict) else None,
            "error": None if st == 202 else str(resp)[:200],
        }
        with lock:
            out[k] = rec

    with ThreadPoolExecutor(max_workers=16) as pool:
        for k, off in enumerate(offsets):
            delay = off - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)
            pool.submit(submit, k)
    return out


def wait_all(api: str, records: list[dict[str, Any]], timeout_s: float, poll_s: float) -> None:
    """Poll each submitted job until ready / failed (or timeout); adds its fields in place."""
    deadline = time.monotonic() + timeout_s
    todo = [r for r in records if r.get("job_id")]
    while todo and time.monotonic() < deadline:
        left = []
        for r in todo:
            st, j = call(api, "GET", f"/v1/jobs/{r['job_id']}")
            if st == 200 and j["status"] in ("ready", "failed"):
                r.update(
                    {
                        k: j.get(k)
                        for k in (
                            "status",
                            "n_tiles",
                            "timings",
                            "created_at",
                            "started_at",
                            "finished_at",
                            "error",
                            "model_id",
                        )
                    }
                )
            else:
                left.append(r)
        todo = left
        if todo:
            time.sleep(poll_s)
    for r in todo:
        r["status"] = "timeout"
