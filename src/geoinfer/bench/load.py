"""Statistics of one load-test run (plan section 10: job latency and throughput).

Latency is ``finished_at - created_at`` of each ready job (database timestamps), reported
as p50 / p95 / p99 overall, per AOI size class and per priority.

Clock jumps: on the laptop the Docker VM clock occasionally flips between UTC and local
time (IST, 5 h 30 min), so a few timestamps land about 19 800 s away from the rest of their
run. :func:`fix_clock_jumps` moves any timestamp more than an hour from the run's median
by the nearest multiple of 19 800 s and counts the corrections (reported per run).
Throughput counts ready jobs, AOI area and tiles between the first submission and the
last finish.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

CLOCK_JUMP_S = 19_800.0  # UTC <-> IST
TS_KEYS = ("created_at", "started_at", "finished_at")


def _ts(v: str) -> float:
    return datetime.fromisoformat(v).timestamp()


def fix_clock_jumps(records: Sequence[dict[str, Any]], jump_s: float = CLOCK_JUMP_S) -> int:
    """Correct timestamps off by whole clock jumps (in place); returns how many moved."""
    vals = [_ts(r[k]) for r in records for k in TS_KEYS if r.get(k)]
    if not vals:
        return 0
    med = float(np.median(vals))
    fixed = 0
    for r in records:
        for k in TS_KEYS:
            if not r.get(k):
                continue
            t = _ts(r[k])
            if abs(t - med) > 3600:
                t += jump_s * round((med - t) / jump_s)
                r[k] = datetime.fromtimestamp(t, tz=UTC).isoformat()
                fixed += 1
    return fixed


def _pct(x: Sequence[float]) -> dict[str, float | int]:
    a = np.asarray(x, float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "p50_s": round(float(np.percentile(a, 50)), 2),
        "p95_s": round(float(np.percentile(a, 95)), 2),
        "p99_s": round(float(np.percentile(a, 99)), 2),
        "mean_s": round(float(a.mean()), 2),
    }


def summarize_run(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Latency percentiles and throughput of the replayed jobs (see module docstring)."""
    records = [dict(r) for r in records]
    clock_fixes = fix_clock_jumps(records)
    ready = [r for r in records if r.get("status") == "ready"]
    out: dict[str, Any] = {
        "clock_fixes": clock_fixes,
        "submitted": sum(1 for r in records if r.get("job_id")),
        "ready": len(ready),
        "failed": sum(1 for r in records if r.get("status") == "failed"),
        "timeout": sum(1 for r in records if r.get("status") == "timeout"),
        "rejected": sum(1 for r in records if r.get("http") not in (None, 202)),
    }
    if not ready:
        return out
    created = np.array([_ts(r["created_at"]) for r in ready])
    finished = np.array([_ts(r["finished_at"]) for r in ready])
    lat = finished - created
    out["latency"] = _pct(lat)
    out["queue_wait"] = _pct(
        [_ts(r["started_at"]) - _ts(r["created_at"]) for r in ready if r.get("started_at")]
    )
    out["by_side_km"] = {
        str(s): _pct(lat[[r["side_km"] == s for r in ready]])
        for s in sorted({r["side_km"] for r in ready})
    }
    out["by_priority"] = {
        str(p): _pct(lat[[r["priority"] == p for r in ready]])
        for p in sorted({r["priority"] for r in ready})
    }
    span = float(finished.max() - created.min())
    km2 = sum(float(r["side_km"]) ** 2 for r in ready)
    tiles = sum(int(r.get("n_tiles") or 0) for r in ready)
    out |= {
        "makespan_s": round(span, 1),
        "jobs_per_min": round(len(ready) / span * 60, 3),
        "km2_per_min": round(km2 / span * 60, 3),
        "tiles_per_s": round(tiles / span, 2),
        "infer_share": round(
            sum(float((r.get("timings") or {}).get("infer_s", 0)) for r in ready)
            / max(1e-9, sum(float((r.get("timings") or {}).get("total_s", 0)) for r in ready)),
            3,
        ),
    }
    return out
