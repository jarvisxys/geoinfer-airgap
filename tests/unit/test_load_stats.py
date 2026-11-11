from __future__ import annotations

import pytest

from geoinfer.bench.load import summarize_run


def _rec(i: int, side: float, prio: int, created: int, started: int, finished: int) -> dict:
    t = "2026-11-10T00:00:{:02d}+00:00"
    return {
        "i": i,
        "job_id": f"j{i}",
        "http": 202,
        "side_km": side,
        "priority": prio,
        "status": "ready",
        "n_tiles": 10,
        "created_at": t.format(created),
        "started_at": t.format(started),
        "finished_at": t.format(finished),
        "timings": {"infer_s": 1.0, "total_s": 4.0},
    }


def test_summarize_run_latency_and_throughput() -> None:
    recs = [_rec(0, 1.0, 0, 0, 0, 10), _rec(1, 0.5, 5, 0, 2, 4), _rec(2, 1.0, 0, 5, 10, 20)]
    recs.append(
        {"i": 3, "job_id": "j3", "http": 202, "status": "failed", "side_km": 1.0, "priority": 0}
    )
    recs.append({"i": 4, "job_id": None, "http": 422, "side_km": 1.0, "priority": 0})
    s = summarize_run(recs)
    assert (s["submitted"], s["ready"], s["failed"], s["rejected"]) == (4, 3, 1, 1)
    assert s["latency"]["n"] == 3
    assert s["latency"]["p50_s"] == 10.0
    assert s["by_priority"]["5"]["p50_s"] == 4.0
    assert s["by_side_km"]["1.0"]["n"] == 2
    assert s["queue_wait"]["p50_s"] == 2.0  # 0, 2, 5 s
    assert s["makespan_s"] == 20.0
    assert s["km2_per_min"] == pytest.approx((1 + 0.25 + 1) / 20 * 60, abs=1e-3)
    assert s["tiles_per_s"] == 1.5
    assert s["infer_share"] == 0.25


def test_summarize_run_without_ready_jobs() -> None:
    assert summarize_run([{"job_id": None, "http": 503}])["ready"] == 0
