"""Statistics of one load-test run (plan section 10: job latency and throughput).

Latency is ``finished_at - created_at`` of each ready job (database timestamps), reported
as p50 / p95 / p99 overall, per AOI size class and per priority. Throughput counts ready
jobs, AOI area and tiles between the first submission and the last finish.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np


def _ts(v: str) -> float:
    return datetime.fromisoformat(v).timestamp()


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
    ready = [r for r in records if r.get("status") == "ready"]
    out: dict[str, Any] = {
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
