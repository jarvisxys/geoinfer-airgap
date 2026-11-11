from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from loadtest.replay import replay, schedule, wait_all


def test_schedule_scales_unit_rate_offsets() -> None:
    reqs = [{"t": 1.0}, {"t": 3.0}]
    assert schedule(reqs, 2.0) == [0.5, 1.5]
    assert schedule(reqs, None) == [0.0, 0.0]
    with pytest.raises(ValueError, match="rate"):
        schedule(reqs, 0.0)


@pytest.fixture
def api() -> Iterator[tuple[str, list[float]]]:
    posts: list[float] = []
    t0 = time.monotonic()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def _send(self, code: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            n = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(n))
            posts.append(time.monotonic() - t0)
            time.sleep(0.5)  # a slow API must not delay later arrivals
            self._send(202, {"id": f"j{len(posts)}", "status": "queued"})

        def do_GET(self) -> None:
            self._send(
                200,
                {
                    "status": "ready",
                    "n_tiles": 3,
                    "created_at": "a",
                    "finished_at": "b",
                    "timings": {},
                },
            )

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", posts
    srv.shutdown()


def test_replay_is_open_loop_and_collects_results(api: tuple[str, list[float]]) -> None:
    url, posts = api
    reqs = [
        {
            "i": k,
            "t": 0.05 * k,
            "scene_id": "s",
            "sensor_mode": "optical",
            "task": "segment_buildings",
            "aoi": {},
            "priority": 0,
            "side_km": 0.5,
        }
        for k in range(6)
    ]
    t0 = time.monotonic()
    recs = replay(url, reqs, rate=1.0)
    # six 0.5 s responses, arrivals 50 ms apart: open loop finishes in ~0.75 s, serial in 3 s
    assert time.monotonic() - t0 < 2.0
    assert all(r["http"] == 202 for r in recs)
    assert max(posts) < 0.6
    wait_all(url, recs, timeout_s=5, poll_s=0.01)
    assert all(r["status"] == "ready" and r["n_tiles"] == 3 for r in recs)
